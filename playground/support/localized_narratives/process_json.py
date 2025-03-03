import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from copy import deepcopy


dataset = 'localized_narratives'
split = 'train'
data = load_dataset("HuggingFaceM4/the_cauldron", dataset ,split="train")
# data = load_dataset("HuggingFaceM4/the_cauldron", config_name ,split="train")

dataroot = 'playground/data'
image_folder = '%s/images' % dataset
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

count = 0
new_data = []
for da in tqdm(data):
    json_data = {}
    
    
    if da["images"] is not None:
        assert len(da["images"]) == 1
        da["image"] = da["images"][0].convert('RGB')
    img_path = os.path.join(dataroot, image_folder, '%06d.jpg' % count)
    da["image"].save(img_path)

    json_data["id"] = os.path.join(image_folder, '%06d.jpg' % count)
    json_data["image"] = os.path.join(image_folder,'%06d.jpg' % count)
    json_data["image_size"] = da["image"].size
    
    for t_id, text in enumerate(da['texts']):
        item = deepcopy(json_data)
        item['id'] += '_%d' % t_id
        conv = []
        question = text['user']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = text['assistant']
        conv.append({'from': 'human', 'value': question})
        conv.append({'from': 'gpt', 'value': answer})
        item["conversations"] = conv
    
        new_data.append(item)
    count += 1

new_json_file = '%s_%s_process.json' % (dataset, split)
if not os.path.exists('playground/data/%s' % dataset):
    os.makedirs('playground/data/%s' % dataset)
dst_json_file = os.path.join('playground/data/%s' % dataset, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)