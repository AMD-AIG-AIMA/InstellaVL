from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

 
dataroot = 'playground/data'
image_folder = 'SA-1B/' 
json_file = os.path.join(dataroot, 'AS-Core/region_vqa_1m.jsonl')
train_dataset = [json.loads(line.strip()) for line in open(json_file)]

new_data = []
for id, train_instance in enumerate(tqdm(train_dataset)):
    item = {}
    if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['image'])):
        print(os.path.join(dataroot, image_folder, train_instance['image']))
        continue
    item['image'] = os.path.join(image_folder, train_instance['image'])
    item['id'] = item['image']
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' +  train_instance['question']})
    conv.append({'from': 'gpt', 'value': train_instance['answer']})
    item['conversations'] = conv
    new_data.append(item)

new_json_file = 'AS-Core_vqa_1m_process.json'
if not os.path.exists('playground/data/AS-Core/processed'):
    os.makedirs('playground/data/AS-Core/processed')
dst_json_file = os.path.join('playground/data/AS-Core/processed', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
