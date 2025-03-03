# -*- coding: utf-8 -*-
from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

split ='train'
dataset = load_dataset("ecnu-icalk/cmm-math")

train_set = dataset[split]  
dataroot = 'playground/data'
image_folder = 'cmm-math/All_Images'
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

new_data = []
for id, train_instance in enumerate(tqdm(train_set)):
    item = {}
    item['id'] = os.path.join(image_folder, '%06d.jpg'%id)

    train_instance['image'] = json.loads(train_instance['image'])
    if train_instance['image']:
        item['image']  = []
        for image in train_instance['image']:
            if not os.path.exists(os.path.join(dataroot,image_folder, image)):
                print(os.path.join(dataroot,image_folder, image))
                continue
            item['image'].append(os.path.join(image_folder, image)) 
    instruction = "请先进行分析，最后给出答案。" + '\n'
    if  train_instance['options']:
        question =  train_instance['question'] + '\n' + train_instance['options'] +'\n'
    else:
        question =  train_instance['question'] + '\n'
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' + instruction + question})
    conv.append({'from': 'gpt', 'value': train_instance['analysis']})
    item['conversations'] = conv
    new_data.append(item)

new_json_file = 'cmm_math_process.json' 
if not os.path.exists('playground/data/%s' % 'cmm-math'):
    os.makedirs('playground/data/%s' % 'cmm-math')
dst_json_file = os.path.join('playground/data/%s' % 'cmm-math', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
