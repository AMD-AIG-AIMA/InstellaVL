import json
from copy import deepcopy
import os
import argparse

parser = argparse.ArgumentParser(description="Process Data")

parser.add_argument('--dataset', type=str, required=True, help="Input file name")

args = parser.parse_args()


src_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/%s.json' % args.dataset
dst_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files_processed/%s.json' % args.dataset

with open(src_json_file, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = deepcopy(item)
    for it, conv in enumerate(item['conversations']):
        if conv['from'] == 'Answer':
            conv['from'] = 'gpt'
        if it==0 and conv['from'] == 'human' and not '<image>'in  conv['value'] and 'image' in item.keys():
            conv['value'] = '<image>\n' + conv['value']
            continue
        if it == 0 and conv['from'] != 'human':
            conv0 = {'from': 'human', 'value': '<image>\n'}
            item['conversations'].insert(0, conv0)
            print('insert human words')
            continue
    # 
    # 
    new_item['conversations'] = item['conversations']
    new_data.append(new_item)

print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)