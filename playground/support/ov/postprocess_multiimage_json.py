import json
from copy import deepcopy
import os
import tqdm

src_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/m4_instruct_annotations.json'
dst_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files_processed/m4_instruct_annotations.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)



dataset_path = 'playground/data/LLaVA-Stage2-OneVision/M4-Instruct-Data'
new_data = []
single_image = 0
no_image = 0
for datem in tqdm.tqdm(data):
    if 'image' in datem.keys():
        if isinstance(datem['image'], list):
            flag = True
            for idx, img in enumerate(datem['image']):
                img_path = os.path.join(dataset_path, img)
                datem['image'][idx] = os.path.join('M4-Instruct-Data', img)
                flag = flag and os.path.exists(img_path)
            if flag and datem['metadata']['dataset'] != 'twitter_post':
                new_data.append(datem)
        else:
            single_image += 1
            img_path = os.path.join(dataset_path, datem['image'])
            datem['image'] = img_path
            if  flag and datem['metadata']['dataset'] != 'twitter_post':
                new_data.append(datem)
    else:
        no_image += 1
print(len(new_data), len(data))
print(single_image, no_image)

with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)