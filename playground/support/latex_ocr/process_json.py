import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from copy import deepcopy
import numpy as np

dataset = 'synthetic_handwrite'
split = 'train'
data = load_dataset("linxy/LaTeX_OCR", dataset ,split="train")
# data = load_dataset("HuggingFaceM4/the_cauldron", config_name ,split="train")

dataroot = 'playground/data'
image_folder = 'LaTeX_OCR/%s/images' % dataset
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

prompts = [
        "Could you convert the equation shown in the image into LaTeX syntax?",
        "Would you mind creating the LaTeX commands needed to reproduce this equation?",
        "Can you translate the mathematical expression from the image into LaTeX code?",
        "I need the LaTeX instructions to create this equation - could you help?",
        "Please provide the LaTeX syntax that would create the equation displayed in this image.",
        "What LaTeX code would I use to produce the equation shown here?",
        "Could you write out the LaTeX commands that would generate this mathematical expression?",
        "I'd like to recreate this equation using LaTeX - what would be the correct code?",
        "Can you tell me the LaTeX syntax needed to match the equation in this image?",
        "Please help me construct the LaTeX code for reproducing this mathematical expression."
        ]

new_data = []
for count, da in enumerate(tqdm(data)):
    json_data = {}
    
    img_path = os.path.join(dataroot, image_folder, '%06d.jpg' % count)
    da["image"].convert('RGB').save(img_path)

    json_data["id"] = os.path.join(image_folder, '%06d.jpg' % count)
    json_data["image"] = os.path.join(image_folder,'%06d.jpg' % count)
    json_data["image_size"] = da["image"].size
    
    conv = []
  
    conv.append({'from': 'human', 'value': '<image>\n' + prompts[np.random.randint(len(prompts))]})
    conv.append({'from': 'gpt', 'value': da['text']})
    json_data["conversations"] = conv
    new_data.append(json_data)
    


new_json_file = 'latex_ocr_%s_process.json' % dataset
if not os.path.exists('playground/data/LaTeX_OCR/' ):
    os.makedirs('playground/data/LaTeX_OCR/')
dst_json_file = os.path.join('playground/data/LaTeX_OCR/', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)