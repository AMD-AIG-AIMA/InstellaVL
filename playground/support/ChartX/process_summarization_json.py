import os
from tqdm import tqdm
import json
import numpy as np



converted_data = []

data = json.load(open("playground/data/ChartX/ChartX_annotation_val.json"))
root_path = "playground/data/"
image_folder = "ChartX/ChartX_png"

for idx, da in tqdm(enumerate(tqdm(data))):
    json_data = {}
    json_data["id"] = da['imgname']
    if not os.path.exists(os.path.join(root_path, image_folder, da['img'])):
        print(os.path.join(root_path, image_folder, da['img']))
        continue
    
    json_data['image'] = os.path.join(image_folder, da['img'])
    if not da.get("summarization"):
        continue
    conv = []
    conv.append({'from': 'human', 'value': "<image>\n" + da['summarization']['input']})
    conv.append({'from': 'gpt', 'value':  da['summarization']['ouput_put']})
    converted_data.append(json_data)



new_json_file = 'ChartX_summarization_process.json'
dst_json_file = os.path.join('playground/data/ChartX', new_json_file)
print(len(converted_data))
with open(dst_json_file, 'w+') as f:
    json.dump(converted_data, f, indent=4)