from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

split ='validation'
ds_name = "coco-itm"  # change the dataset name here
dataset = load_dataset("MMInstruction/M3IT", ds_name)
train_set = dataset[split]  
dataroot = 'playground/data'
image_folder = '%s/images/%s' % (ds_name, split)
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

new_data = []
for id, train_instance in enumerate(tqdm(train_set)):
    item = {}
    image_base64_str_list = train_instance["image_base64_str"]  # str (base64)
    image_0 = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert('RGB')
    image_0.save(os.path.join(dataroot, image_folder, '%06d.jpg'%id))
    item['id'] = os.path.join(image_folder, '%06d.jpg'%id)
    item['image'] = item['id'] 
    instruction = train_instance['instruction'] + '\n'
    question =  train_instance['inputs']
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' + instruction + question})
    conv.append({'from': 'gpt', 'value': train_instance['outputs']})
    item['conversations'] = conv
    item['image_size'] = image_0.size
    new_data.append(item)

new_json_file = '%s_%s_process.json' % (ds_name, split) 
if not os.path.exists('playground/data/%s' % ds_name):
    os.makedirs('playground/data/%s' % ds_name)
dst_json_file = os.path.join('playground/data/%s' % ds_name, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
