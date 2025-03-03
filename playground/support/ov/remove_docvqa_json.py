import json
from copy import deepcopy

src_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files_processed_2/docvqa.json'
dst_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files_processed_2/docvqa2.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)

for item in data:
    for it, conv in enumerate(item['conversations']):
        conv['value'] = conv['value'].replace(' \n', '\n')

with open(dst_json_file, 'w+') as f:
    json.dump(data, f, indent=4)