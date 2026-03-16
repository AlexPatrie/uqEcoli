#!/bin/bash -ue
python3 -c "
import json

stats = json.load(open('stats.json'))
result = {
    'greeting': 'hello from bigraph',
    'count': 7,
    'seed': 123,
    'statistics': stats,
    'status': 'complete',
}
json.dump(result, open('result.json', 'w'), indent=2)
print('Result written.')
"
