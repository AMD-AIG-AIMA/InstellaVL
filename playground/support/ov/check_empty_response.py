import json
from copy import deepcopy
import os
import argparse
import glob
import tqdm

# files = glob.glob('/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/*.json')

# files = [
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/ArT/train_labels_process.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/ArT/train_task2_labels_process.json", 
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mmc_instruction/mmc_instruction_arxiv_text.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mmc_instruction/mmc_instruction_non-arxiv_text.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/Uber-Text/train_labels_process_1k.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/Uber-Text/val_labels_process_1k.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/coco_text/coco_text_process.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/hiertext/train_process.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/hiertext/validation_process.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mlt/train_labels_process.json",
# "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mlt/val_labels_process.json",
# ]

# files = [
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_train_labels_process.json",
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_val_labels_process.json",
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_train_labels_process.json",
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_val_labels_process.json",
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_train_labels_process.json",
#     "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_val_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/Paper2Fig100k/paper2fig_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/RICO-Screen2Words/RICO-Screen2Words.json",
#     "playground/data/data/LLaVA-Stage1.5/Rico_ScreenQA/Rico_ScreenQA.json",
#     "playground/data/data/LLaVA-Stage1.5/SciGraphQA-295K-train/scigraphqa_process.json",
#     "playground/data/data/LLaVA-Stage1.5/chart2text/pew_train_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/chart2text/pew_val_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/chart2text/statista_train_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/chart2text/statista_val_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/openvino/train_1_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/openvino/train_2_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/openvino/train_5_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/openvino/train_f_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/openvino/validation_labels_process.json",
#     "playground/data/data/LLaVA-Stage1.5/UniChart/unichart_labels_process.json"
# ]

files = [
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_train_labels_process.json",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_val_labels_process.json",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_train_labels_process.json",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_val_labels_process.json",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_train_labels_process.json",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_val_labels_process.json",
]

not_passed = []
for json_file in files:

    with open(json_file, 'r') as f:
        data = json.load(f)

    new_data = []
    for a, item in enumerate(tqdm.tqdm(data)):
        if 'image' in item.keys():
            num_image = 1 if isinstance(item['image'], str) else len(item['image'])
            gpt_word = ''
            for it, conv in enumerate(item['conversations']):
                if conv['from'] not in ['human', 'gpt']:
                    print(conv['from'])
                if conv['from'] == 'gpt':
                    gpt_word += conv['value']
            gpt_word = gpt_word.strip()
            if gpt_word != '':
                new_data.append(item)
        else:
            new_data.append(item)
            # if not flag:
            #     
            #     
            #     print(human_word)
    new_file = json_file + '_new'
    print(new_file, len(data), len(new_data))
    with open(new_file, 'w+') as f:
        json.dump(new_data, f, indent=4)
#     if flag:
#         print('%s [PASSED]' % json_file)
#     else:
#         print('%s [Failed]' % json_file)
#         not_passed.append(json_file)

# print('-' * 50)
# for no_pass in not_passed:
#     print(no_pass)