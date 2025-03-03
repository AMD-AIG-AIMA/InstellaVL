import json
from copy import deepcopy
import os
import argparse
import glob
import tqdm

files = [
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/LLaVA-Stage1.5.json_new", 
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/ArT/train_labels_process.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/ArT/train_task2_labels_process.json_new", 
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mmc_instruction/mmc_instruction_arxiv_text.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mmc_instruction/mmc_instruction_non-arxiv_text.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/Uber-Text/train_labels_process_1k.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/Uber-Text/val_labels_process_1k.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/coco_text/coco_text_process.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/hiertext/train_process.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/hiertext/validation_process.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mlt/train_labels_process.json_new",
"/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/mlt/val_labels_process.json_new",
]

files += [
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_val_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_val_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_val_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/Paper2Fig100k/paper2fig_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/RICO-Screen2Words/RICO-Screen2Words.json_new",
    "playground/data/data/LLaVA-Stage1.5/Rico_ScreenQA/Rico_ScreenQA.json_new",
    "playground/data/data/LLaVA-Stage1.5/SciGraphQA-295K-train/scigraphqa_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/chart2text/pew_train_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/chart2text/pew_val_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/chart2text/statista_train_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/chart2text/statista_val_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/openvino/train_1_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/openvino/train_2_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/openvino/train_5_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/openvino/train_f_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/openvino/validation_labels_process.json_new",
    "playground/data/data/LLaVA-Stage1.5/UniChart/unichart_labels_process.json_new"
]

files = [
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/analysis_val_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/cap_val_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_train_labels_process.json_new",
    "/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5/M-Paper/meta/outline_val_labels_process.json_new",
]

root_path = '/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5'
for json_file in files:
    new_data = []
    with open(json_file, 'r') as f:
        data = json.load(f)

    
    for a, item in enumerate(data):
        if 'image' in item.keys():
            flag = True
            if  isinstance(item['image'], str):
                img_path = os.path.join(root_path, item['image'])
                flag = flag and os.path.exists(img_path)
            else:
                
                for image in item['image']: 
                    img_path = os.path.join(root_path, image)
                    flag = flag and os.path.exists(img_path)
            if os.path.exists(img_path):
                new_data.append(item)
        else:
            new_data.append(item)
        
    with open(json_file, 'w+') as f:
        json.dump(new_data, f, indent=4)
    
