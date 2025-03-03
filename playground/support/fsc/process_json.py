import os
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np

converted_data = []

data = load_dataset("Theonewhomadethings/fsc147-controlnet-V2-xl",  split="train")

root_path = 'playground/data'
image_folder = "fsc147/images/"
os.makedirs(os.path.join(root_path, image_folder), exist_ok=True)
prompts = ["How many significant items can you identify in this image?",
"Could you enumerate the main objects visible in this picture?",
"What is the total count of distinct objects shown in the image?",
"Can you tally up the primary elements present in this photo?",
"Would you list the number of prominent objects that appear in this picture?",
"How many key items do you observe in this visual?",
"Could you determine the quantity of notable objects captured in this image?",
"Please identify and count the main elements in this photograph.",
"What's the total number of principal objects you can spot in this picture?",
"Can you provide a count of the major items visible in this image?"]

for idx, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = idx
    
    da["image"] =  da["image"].convert('RGB')
    da["image"].save(os.path.join(root_path, image_folder, '%07d.jpg' % idx), quality=95)

    json_data["image"] = os.path.join(image_folder, '%07d.jpg' % idx)
        
    conv = []
    instruction = prompts[np.random.randint(len(prompts))]
    conv.append({'from': 'human', 'value':  '<image>\n' + instruction})
    conv.append({'from': 'gpt', 'value': da['text']})
    json_data["conversations"] = conv
    converted_data.append(json_data)
    if (idx + 1) % 100000 == 0:
        print(len(converted_data))
        with open(os.path.join(root_path, 'fsc147/fsc147_process.json'), "w+") as f:
            json.dump(converted_data, f, indent=4, ensure_ascii=False)

print(len(converted_data))
with open(os.path.join(root_path, 'fsc147/fsc147_process.json'), "w+") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)