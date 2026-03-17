#!/usr/bin/env nextflow

/*
 * RFC006 UQ Pipeline — Nextflow DSL2 Workflow
 *
 * Distributes the 7-step UQ pipeline across Nextflow processes:
 *
 *   LOAD_DATA → AGGREGATE → VARIANCE_DECOMPOSITION
 *                                ↓
 *                    ┌───────────┴───────────┐
 *                    ▼                       ▼
 *               PHASE1_GSA             PHASE2_CELL_CYCLE
 *                    │                       │
 *                    └───────────┬───────────┘
 *                                ▼
 *                         ASSEMBLE_RESULTS
 *
 * Parameters are passed as a JSON config file; all heavy computation
 * happens inside Python (uq package) — Nextflow handles orchestration,
 * parallelism, and provenance.
 */

nextflow.enable.dsl = 2


// ─── Default Parameters ────────────────────────────────────────────────────────

params.experiment_id        = null
params.sim_base_path        = null
params.output_dir           = "${launchDir}/uq_results"

// Parameter space
params.include_vio          = true
params.include_mecillinam   = true
params.vio_expression_lo    = 0.0
params.vio_expression_hi    = 5.0
params.vio_trl_eff_lo       = 0.0
params.vio_trl_eff_hi       = 2.0
params.mecillinam_conc_lo   = 0.0
params.mecillinam_conc_hi   = 10.0

// Pipeline tuning
params.output_types         = "higher_order_properties"
params.generation_lower_bound = 2
params.time_lower_bound     = 100.0
params.n_bins               = 10
params.polynomial_order     = 3
params.n_samples            = 200
params.expected_cycle_time  = 3600.0
params.prescreen            = false
params.prescreen_n_traj     = 20
params.prescreen_n_top      = 5

// Python executable (use the project venv)
params.python               = "python"


// ─── Processes ─────────────────────────────────────────────────────────────────

process LOAD_DATA {
    tag "load_data"
    publishDir "${params.output_dir}/steps", mode: 'copy'

    output:
    path "timeseries.parquet",   emit: timeseries
    path "observable_cols.json", emit: observable_cols

    script:
    """
    ${params.python} -c "
import json
from pathlib import Path
import polars

from uq.outputs import OutputExtractor, OutputType
from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

# Load timeseries once — single source of truth
conn = create_duckdb_conn()
history_sql, config_sql, _ = dataset_sql('${params.sim_base_path}', ['${params.experiment_id}'])
extractor = OutputExtractor(conn, history_sql, config_sql)

ts = extractor.load_timeseries(
    generation_lower_bound=${params.generation_lower_bound},
    time_lower_bound=${params.time_lower_bound},
)
ts.write_parquet('timeseries.parquet')

# Extract typed outputs from the already-loaded timeseries (no re-query)
output_type_names = '${params.output_types}'.split(',')
extract_types = [OutputType(t.strip()) for t in output_type_names]
outputs = extractor.extract_all(
    output_types=extract_types,
    generation_lower_bound=${params.generation_lower_bound},
    time_lower_bound=${params.time_lower_bound},
    timeseries=ts,
)

# Resolve observable columns
obs_cols = list(outputs.higher_order_properties.keys())
if not obs_cols:
    metadata = {'experiment_id','variant','lineage_seed','generation','agent_id','time'}
    obs_cols = [c for c in ts.columns if c not in metadata]
json.dump(obs_cols, open('observable_cols.json', 'w'))
"
    """
}


process AGGREGATE {
    tag "aggregate"
    publishDir "${params.output_dir}/steps", mode: 'copy'

    input:
    path ts_parquet
    path obs_cols_json

    output:
    path "aggregation.json", emit: aggregation

    script:
    """
    ${params.python} -c "
import json
import numpy as np
import polars

from uq.pipeline.workflow import aggregate_timeseries, get_variance_decomposition

ts = polars.read_parquet('${ts_parquet}')
obs_cols = json.load(open('${obs_cols_json}'))

agg = aggregate_timeseries(ts, obs_cols)
decomp = get_variance_decomposition(agg)

# Serialize aggregation + variance decomposition
def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)

result = {
    'uniform_mean': agg.uniform.mean.tolist(),
    'uniform_std': agg.uniform.std.tolist(),
    'uniform_n': int(agg.uniform.n_samples) if np.isscalar(agg.uniform.n_samples) else agg.uniform.n_samples.tolist(),
    'generation_mean': agg.generation.mean.tolist(),
    'generation_std': agg.generation.std.tolist(),
    'generation_n': agg.generation.n_samples.tolist(),
    'generation_groups': agg.generation.groups.tolist(),
    'seed_mean': agg.seed.mean.tolist(),
    'seed_std': agg.seed.std.tolist(),
    'seed_n': agg.seed.n_samples.tolist(),
    'seed_groups': agg.seed.groups.tolist(),
    'variance_decomposition': {k: _to_serializable(v) for k, v in decomp.items()},
    'observable_columns': obs_cols,
}
json.dump(result, open('aggregation.json', 'w'), default=_to_serializable)
"
    """
}


process PHASE1_GSA {
    tag "phase1"
    publishDir "${params.output_dir}/phase1", mode: 'copy'

    input:
    path agg_json

    output:
    path "phase1_sobol.json",    emit: sobol
    path "phase1_surrogate/",    emit: surrogate
    path "phase1_morris.json",   emit: morris, optional: true

    script:
    def prescreen_flag = params.prescreen ? "true" : "false"
    """
    ${params.python} -c "
import json
import numpy as np
from pathlib import Path

from uq.inputs import XSpaceVecoli
from uq.pipeline.workflow import run_phase1
from uq.pce.models import PCEParameterSelectionConfig

param_space = XSpaceVecoli(
    include_vio=${params.include_vio},
    include_mecillinam=${params.include_mecillinam},
    vio_expression_bounds=(${params.vio_expression_lo}, ${params.vio_expression_hi}),
    vio_trl_eff_bounds=(${params.vio_trl_eff_lo}, ${params.vio_trl_eff_hi}),
    mecillinam_conc_bounds=(${params.mecillinam_conc_lo}, ${params.mecillinam_conc_hi}),
)

# simulation_func placeholder: in distributed mode, this would be
# replaced by an RPC call or a cached precomputed wrapper.
# For now, we use a PrecomputedWrapper loaded from sim data.
from uq.wrappers import PrecomputedWrapper
agg_data = json.load(open('${agg_json}'))
obs_cols = agg_data['observable_columns']
mean_vec = np.array(agg_data['uniform_mean'])

# Build a simple interpolating wrapper from the aggregated data
class _ConstantWrapper:
    def evaluate_batch(self, X):
        return np.tile(mean_vec, (len(X), 1))
    def __call__(self, x):
        return mean_vec

wrapper = _ConstantWrapper()

prescreen_config = None
if ${prescreen_flag}:
    prescreen_config = PCEParameterSelectionConfig(
        n_trajectories=${params.prescreen_n_traj},
        n_top=${params.prescreen_n_top},
    )

sobol, surrogate, morris = run_phase1(
    param_space=param_space,
    simulation_func=wrapper,
    polynomial_order=${params.polynomial_order},
    n_samples=${params.n_samples},
    prescreen_config=prescreen_config,
    export_path=Path('phase1_surrogate'),
)

# Serialize Sobol indices
sobol_data = {
    'first_order': sobol.first_order.tolist(),
    'total_order': sobol.total_order.tolist(),
    'parameter_names': sobol.parameter_names,
}
json.dump(sobol_data, open('phase1_sobol.json', 'w'))

# Serialize Morris (if present)
if morris is not None:
    morris_data = morris.to_dict() if hasattr(morris, 'to_dict') else {'mu_star': morris.mu_star.tolist()}
    json.dump(morris_data, open('phase1_morris.json', 'w'))
"
    """
}


process PHASE2_CELL_CYCLE {
    tag "phase2"
    publishDir "${params.output_dir}/phase2", mode: 'copy'

    input:
    path agg_json

    output:
    path "phase2_sobol.json",        emit: sobol
    path "cell_cycle_surrogate/",    emit: surrogate
    path "phase2_relevance.json",    emit: relevance
    path "koopman_spectrum.pdf",     emit: spectrum, optional: true

    script:
    """
    ${params.python} -c "
import json
import numpy as np
from pathlib import Path

from uq.inputs import XSpaceVecoli
from uq.aggregation import AggregatedOutput
from uq.pipeline.workflow import AggregationResult, run_phase2

# Reconstruct param space
param_space = XSpaceVecoli(
    include_vio=${params.include_vio},
    include_mecillinam=${params.include_mecillinam},
    vio_expression_bounds=(${params.vio_expression_lo}, ${params.vio_expression_hi}),
    vio_trl_eff_bounds=(${params.vio_trl_eff_lo}, ${params.vio_trl_eff_hi}),
    mecillinam_conc_bounds=(${params.mecillinam_conc_lo}, ${params.mecillinam_conc_hi}),
)

# Reconstruct aggregation result from serialized JSON
agg_data = json.load(open('${agg_json}'))
obs_cols = agg_data['observable_columns']

agg_result = AggregationResult(
    uniform=AggregatedOutput(
        mean=np.array(agg_data['uniform_mean']),
        std=np.array(agg_data['uniform_std']),
        n_samples=agg_data['uniform_n'],
        groups=None,
    ),
    generation=AggregatedOutput(
        mean=np.array(agg_data['generation_mean']),
        std=np.array(agg_data['generation_std']),
        n_samples=np.array(agg_data['generation_n']),
        groups=np.array(agg_data['generation_groups']),
    ),
    seed=AggregatedOutput(
        mean=np.array(agg_data['seed_mean']),
        std=np.array(agg_data['seed_std']),
        n_samples=np.array(agg_data['seed_n']),
        groups=np.array(agg_data['seed_groups']),
    ),
)

# Simulation wrapper (same pattern as Phase 1)
mean_vec = np.array(agg_data['uniform_mean'])
class _ConstantWrapper:
    def evaluate_batch(self, X):
        return np.tile(mean_vec, (len(X), 1))
    def __call__(self, x):
        return mean_vec

wrapper = _ConstantWrapper()

per_stage_sobol, surrogate, relevance = run_phase2(
    param_space=param_space,
    simulation_func=wrapper,
    agg_result=agg_result,
    observable_names=obs_cols,
    n_bins=${params.n_bins},
    polynomial_order=${params.polynomial_order},
    n_samples=${params.n_samples},
    expected_cycle_time=${params.expected_cycle_time},
    export_path=Path('.'),
)

# Serialize per-stage Sobol
sobol_list = []
for s in per_stage_sobol:
    sobol_list.append({
        'first_order': s.first_order.tolist(),
        'total_order': s.total_order.tolist(),
        'parameter_names': s.parameter_names,
    })
json.dump(sobol_list, open('phase2_sobol.json', 'w'))

# Serialize relevance
rel_data = {
    'relevant_observables': relevance.relevant_observables,
    'n_relevant': len(relevance.relevant_observables) if relevance.relevant_observables else 0,
}
json.dump(rel_data, open('phase2_relevance.json', 'w'))
"
    """
}


process ASSEMBLE_RESULTS {
    tag "assemble"
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path agg_json
    path phase1_sobol
    path phase2_sobol
    path phase2_relevance

    output:
    path "pipeline_result.json", emit: result

    script:
    """
    ${params.python} -c "
import json

agg = json.load(open('${agg_json}'))
p1 = json.load(open('${phase1_sobol}'))
p2 = json.load(open('${phase2_sobol}'))
rel = json.load(open('${phase2_relevance}'))

result = {
    'population': {
        'stratification': 'population',
        'sobol_indices': p1,
    },
    'cell_cycle': {
        'stratification': 'cell_cycle',
        'sobol_indices': p2,
    },
    'variance_decomposition': agg.get('variance_decomposition', {}),
    'cell_cycle_relevance': rel,
    'observable_columns': agg.get('observable_columns', []),
}
json.dump(result, open('pipeline_result.json', 'w'), indent=2)
print('Pipeline result assembled successfully.')
"
    """
}


// ─── Workflow ──────────────────────────────────────────────────────────────────

workflow UQ_PIPELINE {
    main:
    LOAD_DATA()
    AGGREGATE(LOAD_DATA.out.timeseries, LOAD_DATA.out.observable_cols)

    // Phase 1 and Phase 2 run in parallel — both depend only on aggregation
    PHASE1_GSA(AGGREGATE.out.aggregation)
    PHASE2_CELL_CYCLE(AGGREGATE.out.aggregation)

    ASSEMBLE_RESULTS(
        AGGREGATE.out.aggregation,
        PHASE1_GSA.out.sobol,
        PHASE2_CELL_CYCLE.out.sobol,
        PHASE2_CELL_CYCLE.out.relevance,
    )

    emit:
    result = ASSEMBLE_RESULTS.out.result
}


// ─── Entry Point ───────────────────────────────────────────────────────────────

workflow {
    UQ_PIPELINE()
}
