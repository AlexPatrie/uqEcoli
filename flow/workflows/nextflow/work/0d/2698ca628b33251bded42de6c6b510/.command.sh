#!/bin/bash -ue
python3 -c "
import random, csv
random.seed(42)
with open('data.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'value'])
    for i in range(15):
        w.writerow([i, round(random.gauss(0, 1), 6)])
"
