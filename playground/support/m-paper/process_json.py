import os
from datasets import load_dataset
from tqdm import tqdm
import json


converted_data = []

data = load_dataset("HuggingFaceM4/Docmatix", 'images',  split="train")

root_path = 'playground/data'
image_folder = "Docmatix/images/"
os.makedirs(os.path.join(root_path, image_folder), exist_ok=True)

image_id = 0
for idx, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = idx
    
    if da["images"] is not None:
        image_path = []
        for image in da["images"]:
            image = image.convert('RGB')
            image.save(os.path.join(root_path, image_folder, '%07d.jpg' % image_id), quality=95)
            image_path.append(os.path.join(image_folder, '%07d.jpg' % image_id))
            image_id += 1
        json_data["image"] = image_path
        

        
    json_data["conversations"] = 
    conv = {}
    for item in da['texts']:
        conv.append({})
    json_data["conversations"][0]['value'] = "<image>\n" * len() + "Please generate detailed descriptions of the given image."
    json_data['id'] = 'LLaVA-ReCap-CC3M' + '/' + json_data['id']
    json_data['image'] = 'LLaVA-ReCap-CC3M' + '/' + json_data['image']
    converted_data.append(json_data)



with open("LLaVA-ReCap-CC3M.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)