from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

 
dataroot = 'playground/data'
image_folder = 'sam/' 
train_dataset = json.load(open(os.path.join(dataroot, 'AS-V2/as_pretrain_10m.json')))

new_data = []
for id, train_instance in enumerate(tqdm(train_dataset)):
    item = {}
    if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['image'] )):
        print(os.path.join( image_folder, train_instance['image'] ))
        continue
    train_instance['image'] =  os.path.join( image_folder, train_instance['image'] )
    new_data.append(train_instance)

new_json_file = 'AS-V2_pretrain_process.json'
if not os.path.exists('playground/data/AS-V2'):
    os.makedirs('playground/data/AS-V2')
dst_json_file = os.path.join('playground/data/AS-V2', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
