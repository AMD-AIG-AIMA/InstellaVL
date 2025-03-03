import json
from copy import deepcopy

src_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files/chart2text.json'
dst_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files/chart2text2.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = deepcopy(item)
    for it, conv in enumerate(item['conversations']):
        if conv['from'] == 'human':
            conv['value'] = 'Provide the requested information directly.\n' + conv['value'] 

    # 
    # 
    new_item['conversations'] = item['conversations']
    new_data.append(new_item)




with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)