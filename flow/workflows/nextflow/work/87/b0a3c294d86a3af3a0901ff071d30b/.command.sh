#!/bin/bash -ue
python3 -c "
import csv, json, math

values = []
with open('data.csv') as f:
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
