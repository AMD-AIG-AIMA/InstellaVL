import json
from copy import deepcopy

src_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/tabmwp.json'
dst_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/tabmwp2.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = deepcopy(item)
    for it, conv in enumerate(item['conversations']):
        if conv['from'] == 'human':
            conv['value'] = 'Hint: Please answer the question and provide the final answer at the end.\n' + conv['value'] 

    # 
    # 
    new_item['conversations'] = item['conversations']
    new_data.append(new_item)




with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)