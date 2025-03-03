import os
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np



converted_data = []

data = json.load(open("playground/data/LVIS-Instruct4V/lvis_instruct4v_220k.json"))
root_path = "playground/data/"
image_folder = "LVIS-Instruct4V/"

for idx, da in tqdm(enumerate(tqdm(data))):
    json_data = {}
    json_data["id"] = da['id']
    if not os.path.exists(os.path.join(root_path, image_folder, da['image'])):
        print(os.path.join(root_path, image_folder, da['image']))
        continue
    
    json_data['image'] = os.path.join(image_folder, da['image'])

    json_data['conversations'] = da['conversations']
    for item in json_data['conversations']:
        item['value'] =  item['value'].replace('\n<image>', "")
    json_data['conversations'][0]['value'] = "<image>\n" + json_data['conversations'][0]['value']
    converted_data.append(json_data)



new_json_file = 'lvis_instruct_process.json'
dst_json_file = os.path.join('playground/data/LVIS-Instruct4V', new_json_file)
print(len(converted_data))
with open(dst_json_file, 'w+') as f:
    json.dump(converted_data, f, indent=4)