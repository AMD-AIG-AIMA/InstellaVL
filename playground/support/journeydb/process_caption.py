from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

def construct(question, options):
    output = question +'\n'
    for option, option_p in options.items():
        output += option + '. ' + option_p + '\n'
    return output
 
dataroot = 'playground/data'
image_folder = 'JourneyDB_2/data/train/imgs' 
json_file = os.path.join(dataroot, 'JourneyDB_2/data/train/train_anno_realease_repath.jsonl')

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
        if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['img_path'])):
            print(os.path.join(dataroot, image_folder, train_instance['img_path']))
            continue
        item['image'] = os.path.join(image_folder, train_instance['img_path'])
        item['id'] = item['image']
        conv = []
        for key, value in train_instance['Task3'].items():
            for qa_pair in value:
                try:
                    question = qa_pair['Question']
                    options = qa_pair['Options']
                    answer = qa_pair['Answer']
                    conv.append({'from': 'human', 'value': construct(question, options)})
                    conv.append({'from': 'gpt', 'value': answer})
                except:
                    continue
        if not len(conv):
            continue
        conv[0]['value'] = '<image>\n' + conv[0]['value'] 
        item['conversations'] = conv
        new_data.append(item)



new_json_file = 'JourneyDB_2_qa_process.json'
if not os.path.exists('playground/data/JourneyDB_2'):
    os.makedirs('playground/data/JourneyDB_2')
dst_json_file = os.path.join('playground/data/JourneyDB_2', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
