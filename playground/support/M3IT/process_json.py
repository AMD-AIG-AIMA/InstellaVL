from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

split ='train'
dataset = load_dataset("HuggingFaceM4/M3IT", trust_remote_code=True)
train_set = dataset[split]  
dataroot = 'playground/data'
image_folder = 'M3IT/images/' 
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

new_data = []
for id, train_instance in enumerate(tqdm(train_set)):
    item = {}
    image_base64_str_list = train_instance["image"]  # str (base64)
    image_0 = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert('RGB')
    image_0.save(os.path.join(dataroot, image_folder, '%08d.jpg'%id), quality=95)
    item['id'] = os.path.join(image_folder, '%08d.jpg'%id)
    item['image'] = item['id'] 
    instruction = train_instance['instruction'] + '\n'
    question =  train_instance['inputs']
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' + instruction + question})
    conv.append({'from': 'gpt', 'value': train_instance['outputs']})
    item['conversations'] = conv
    new_data.append(item)

new_json_file = 'M3IT_process.json'
if not os.path.exists('playground/data/M3IT'):
    os.makedirs('playground/data/M3IT')
dst_json_file = os.path.join('playground/data/M3IT', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
