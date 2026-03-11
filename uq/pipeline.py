from uq import InputParameterSpaceVecoli, SensitivityAnalyzer
from uq.inputs import InputParameterSpace
from uq.models import Parameter, PrescreeningConfig


def prescreen_parameters(full_space: InputParameterSpace, config: PrescreeningConfig | None = None) -> list[Parameter]:
    analyzer = SensitivityAnalyzer(full_space)
    conf = config or PrescreeningConfig()
    screening = analyzer.analyze_with_morris(n_trajectories=conf.n_trajectories)
    return screening.to_parameter_config(parameter_bounds=full_space.parameter_bounds, top_n=conf.n_top)


def prescreen_parameters_vecoli(
    vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
    vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
    mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    include_vio: bool = True,
    include_mecillinam: bool = True,
    knockout_genes: list[str] | None = None,
    prescreen_config: PrescreeningConfig | None = None,
):
    full_space = InputParameterSpaceVecoli(
        vio_expression_bounds=vio_expression_bounds,
        vio_trl_eff_bounds=vio_trl_eff_bounds,
        mecillinam_conc_bounds=mecillinam_conc_bounds,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
        knockout_genes=knockout_genes,
    )
    return prescreen_parameters(full_space=full_space, config=prescreen_config)


PARAMETER_CONFIG = [
    {
        "name": "expression_factor",
        "bounds": [0.5, 5.0],
        "default": 2.75,
        "step": 0.1,
        "description": "Gene expression multiplier (1.0 = baseline)",
    },
    {
        "name": "translation_efficiency",
        "bounds": [0.5, 2.0],
        "default": 1.25,
        "step": 0.05,
        "description": "Translation efficiency factor",
    },
    {
        "name": "inhibitor_conc",
        "bounds": [0.0, 10.0],
        "default": 5.0,
        "step": 0.5,
        "description": "Inhibitor concentration (reduces output)",
    },
]
# -------------------------------------------------------------------------
# PCE Configuration
# -------------------------------------------------------------------------
PCE_ORDER = 2  # Polynomial order for the surrogate (1, 2, or 3 recommended)

# -------------------------------------------------------------------------
# Timeseries Configuration
# -------------------------------------------------------------------------
TIMESERIES_LENGTH = 1000  # Number of timesteps
RANDOM_SEED = 42  # For reproducible noise
