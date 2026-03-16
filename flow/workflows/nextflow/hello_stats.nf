#!/usr/bin/env nextflow

/*
 * hello_stats.nf — A minimal Nextflow workflow for demonstrating NextflowProcess.
 *
 * Takes a greeting and a count, generates random numbers, computes summary
 * statistics, and writes a result JSON. Simple enough to run anywhere
 * (no external data or tools), but structured like a real pipeline:
 *
 *   GENERATE_DATA → COMPUTE_STATS → WRITE_RESULT
 */

nextflow.enable.dsl = 2

params.greeting   = "hello"
params.count      = 10
params.seed       = 42
params.output_dir = "${launchDir}/hello_results"


process GENERATE_DATA {
    tag "generate"

    output:
    path "data.csv", emit: data

    script:
    """
    python3 -c "
import random, csv
random.seed(${params.seed})
with open('data.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'value'])
    for i in range(${params.count}):
        w.writerow([i, round(random.gauss(0, 1), 6)])
"
    """
}


process COMPUTE_STATS {
    tag "stats"

    input:
    path csv_file

    output:
    path "stats.json", emit: stats

    script:
    """
    python3 -c "
import csv, json, math

values = []
with open('${csv_file}') as f:
    for row in csv.DictReader(f):
        values.append(float(row['value']))

n = len(values)
mean = sum(values) / n
variance = sum((x - mean)**2 for x in values) / (n - 1) if n > 1 else 0.0
std = math.sqrt(variance)

json.dump({
    'n': n,
    'mean': round(mean, 6),
    'std': round(std, 6),
    'min': round(min(values), 6),
    'max': round(max(values), 6),
}, open('stats.json', 'w'), indent=2)
"
    """
}


process WRITE_RESULT {
    tag "result"
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path stats_json

    output:
    path "result.json", emit: result

    script:
    """
    python3 -c "
import json

stats = json.load(open('${stats_json}'))
result = {
    'greeting': '${params.greeting}',
    'count': ${params.count},
    'seed': ${params.seed},
    'statistics': stats,
    'status': 'complete',
}
json.dump(result, open('result.json', 'w'), indent=2)
print('Result written.')
"
    """
}


workflow {
    GENERATE_DATA()
    COMPUTE_STATS(GENERATE_DATA.out.data)
    WRITE_RESULT(COMPUTE_STATS.out.stats)
}
