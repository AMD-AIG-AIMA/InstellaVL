import json
from copy import deepcopy
import os
import tqdm

src_json_file = 'playground/data/LLaVA-Stage2-OneVision/json_files/m4_instruct_annotations.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)



dataset_path = 'playground/data/LLaVA-Stage2-OneVision/M4-Instruct-Data'
unsatisfied = []
for datem in tqdm.tqdm(data):
    if 'image' in datem.keys():
        if isinstance(datem['image'], list):
            flag = True
            for img in datem['image']:
                img_path = os.path.join(dataset_path, img)
                flag = flag and os.path.exists(img_path)
            if not flag and datem['metadata']['dataset'] != 'twitter_post':
                unsatisfied.append(datem)
        else:
            img_path = os.path.join(dataset_path, datem['image'])
            if not flag and datem['metadata']['dataset'] != 'twitter_post':
                unsatisfied.append(datem)
print(len(unsatisfied), len(data))

