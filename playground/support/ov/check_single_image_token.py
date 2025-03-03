import json
from copy import deepcopy
import os
import argparse
import glob
import tqdm

files = glob.glob('playground/data/LLaVA-Stage2-OneVision/json_files_processed/*.json')

# files = [
#     # "playground/data/LLaVA-Stage2-Single-Image/json_files_processed/llavar.json",
#     # "playground/data/LLaVA-Stage2-Single-Image/json_files_processed/lrv_normal.json", 
#     # "playground/data/LLaVA-Stage2-Single-Image/json_files_processed/llava_v1_5_mix665k.json"
#     "playground/data/LLaVA-Stage2-Single-Image/json_files_processed_2/evol_instruct.json"
#     ]

not_passed = []
for json_file in files:

    with open(json_file, 'r') as f:
        data = json.load(f)

    flag = True
    for a, item in enumerate(data):
        if 'image' in item.keys():
            num_image = 1 if isinstance(item['image'], str) else len(item['image'])
            human_word = ''
            for it, conv in enumerate(item['conversations']):
                if conv['from'] not in ['human', 'gpt']:
                    print(conv['from'])
                if conv['from'] == 'human':
                    human_word += conv['value']
            flag = flag and (human_word.count('<image>') == num_image or not 'image' in item.keys()) 
            # if not flag:
            #     
            #     
            #     print(human_word)
    if flag:
        print('%s [PASSED]' % json_file)
    else:
        print('%s [Failed]' % json_file)
        not_passed.append(json_file)

print('-' * 50)
for no_pass in not_passed:
    print(no_pass)