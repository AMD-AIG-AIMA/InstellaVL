import json
import os
from tqdm import tqdm
dataroot = 'playground/data'
dataset = 'ArxivCap'
image_folder = "ArxivCap/images"
os.makedirs(os.path.join(dataroot, image_folder), exist_ok=True)

from datasets import load_dataset
data = load_dataset(os.path.join(dataroot, dataset))['train']

# sample = arxiv_qa[0]
# print(sample["image"]) # image file 
new_data = []
idx = 0
for  item in tqdm(data):
    
    for image_info in item["caption_images"]:

        new_item = {}

        new_item['id'] = "%012d" % idx

        main_caption = image_info['caption']
        new_item['image'] = []
        # 
        # 
        cli_pairs = image_info["cil_pairs"]
        # save images
        for cli_idx, cli_pair in enumerate(cli_pairs):
            iamge_name = new_item['id'] +"_%d.jpg" % cli_idx
            image = cli_pair["image"].convert("RGB")
            image.save(os.path.join(dataroot, image_folder, iamge_name), quality=95)
            new_item['image'].append(os.path.join(image_folder, iamge_name))
        
        conv = []

        if len(cli_pairs) == 1:
            conv.append({'from': 'human', 'value': "<image>\nPlease generate the caption for the given image"})
            conv.append({'from': 'gpt', 'value': main_caption})
        else:
            for cli_idx, cli_pair in enumerate(cli_pairs):
                conv.append({'from': 'human', 'value': "<image>\nPlease generate the caption for the given image"})
                conv.append({'from': 'gpt', 'value': cli_pair["sub_caption"]})
            conv.append({'from': 'human', 'value': "Please generate an overall caption for previous all images"})
            conv.append({'from': 'gpt', 'value': main_caption})
            
        new_item['conversations'] = conv
        new_data.append(new_item)
        idx += 1

new_json_file = 'arxivcap_train_process.json' 
if not os.path.exists('playground/data/ArxivCap'):
    os.makedirs('playground/data/ArxivCap')
dst_json_file = os.path.join('playground/data/ArxivCap', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)