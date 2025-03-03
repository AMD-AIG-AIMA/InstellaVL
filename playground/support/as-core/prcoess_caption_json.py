from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
 
dataroot = 'playground/data'
image_folder = 'SA-1B/' 
json_file = os.path.join(dataroot, 'AS-Core/region_caption_400k.jsonl')
train_dataset = [json.loads(line.strip()) for line in open(json_file)]

prompts = ["Describe the region <box>%s</box> concisely.",
           "Provide a brief description of the given region <box>%s</box>.",
           "Offer a succinct explanation of the region <box>%s</box> presented.",
           "Summarize the visual content of the region <box>%s</box>.",
            "Give a short and clear explanation of the subsequent region <box>%s</box>.",
            "Share a concise interpretation of the region <box>%s</box> provided.",
            "Present a compact description of the region <box>%s</box> key features.",
            "Relay a brief, clear account of the region <box>%s</box> shown.",
            "Render a clear and concise summary of the region <box>%s</box>.",
            "Write a terse but informative summary of the region <box>%s</box>.",
            "Create a compact narrative representing the region <box>%s</box> presented."]


new_data = []
for id, train_instance in enumerate(tqdm(train_dataset)):
    item = {}
    if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['image'])):
        print(os.path.join(dataroot, image_folder, train_instance['image']))
        continue
    item['image'] = os.path.join(image_folder, train_instance['image'])
    item['id'] = item['image']
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' +  prompts[np.random.randint(len(prompts))] % str( train_instance['bbox']) })
    conv.append({'from': 'gpt', 'value': train_instance['caption']})
    item['conversations'] = conv
    new_data.append(item)

new_json_file = 'AS-Core_caption_1m_process.json'
if not os.path.exists('playground/data/AS-Core/processed'):
    os.makedirs('playground/data/AS-Core/processed')
dst_json_file = os.path.join('playground/data/AS-Core/processed', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
