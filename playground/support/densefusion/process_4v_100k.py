from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
 
dataroot = 'playground/data'
image_folder = 'DenseFusion-1M/images' 
json_file = os.path.join(dataroot, 'DenseFusion-1M/DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl')


prompts = ["Describe the image concisely.",
           "Provide a brief description of the given image.",
           "Offer a succinct explanation of the picture presented.",
           "Summarize the visual content of the image.",
            "Give a short and clear explanation of the subsequent image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo’s key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented."]


new_data = []
for id, line in enumerate(tqdm(open(json_file))):
    train_instance = json.loads(line.strip())
    item = {}
    if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['image_path'])):
        print(os.path.join(dataroot, image_folder, train_instance['image_path']))
        continue
    item['image'] = os.path.join(image_folder, train_instance['image_path'])
    item['id'] = train_instance['image_id']
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' +  prompts[np.random.randint(len(prompts))]})
    conv.append({'from': 'gpt', 'value': train_instance['caption']})
    item['conversations'] = conv
    new_data.append(item)

new_json_file = 'DenseFusion-4V-100k_process.json'
if not os.path.exists('playground/data/DenseFusion-1M'):
    os.makedirs('playground/data/DenseFusion-1M')
dst_json_file = os.path.join('playground/data/DenseFusion-1M', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
