import json
from copy import deepcopy
import os
import argparse

parser = argparse.ArgumentParser(description="Process Data")

parser.add_argument('--dataset', type=str, required=True, help="Input file name")
parser.add_argument('--prefix', type=str,  help="Input file name")

args = parser.parse_args()

if args.prefix is None:
    args.prefix = args.dataset

src_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/%s.json' % args.dataset
dst_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files_processed/%s.json' % args.dataset
if not os.path.exists(os.path.dirname(dst_json_file)):
    os.makedirs(os.path.dirname(dst_json_file))
with open(src_json_file, 'r') as f:
    data = json.load(f)

root_path = 'playground/data/LLaVA-Stage2-OneVision/'
count = 0
for item in data:
    # 
    # 
    if 'image' in item.keys():
        item['image'] = os.path.join(args.prefix, item['image'] )
        if not os.path.exists(os.path.join(root_path, item['image'])):
            print(item['image'] )
        count += 1

print(count, len(data))
with open(dst_json_file, 'w+') as f:
    json.dump(data, f, indent=4)