import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Process Data")



json_file_folder = "playground/data/RICO-Screen2Words"
if not os.path.exists(json_file_folder):
        os.makedirs(json_file_folder)

converted_data = []
name = 'RICO-Screen2Words'
    
data = load_dataset("rootsautomation/RICO-Screen2Words" ,split="train")
# data = load_dataset("HuggingFaceM4/the_cauldron", config_name ,split="train")

image_folder = f"playground/data/RICO-Screen2Words/"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

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

num_prompts = len(prompts)
for da in tqdm(data):
    # 
    # 
    json_data = {}
    json_data["id"] = da["file_name"]
    
    if da["image"] is not None:
        da["image"] = da["image"].convert('RGB')
        json_data["image"] = f"{da['file_name']}"
        img_path = os.path.join(image_folder, json_data["image"])
        base_dir = os.path.dirname(img_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        da["image"].save(os.path.join(image_folder, json_data["image"]))
        json_data["image"] = f"RICO-ScreenQA/{da['file_name']}"
    for caption in da['captions']:
        num = np.random.randint(0, num_prompts)
        json_data["conversations"] = [{'from': "human", "value": "<image>\n%s" % prompts[num]},
                                    {'from': 'gpt', "value": caption}]
        converted_data.append(json_data)

print(f'{name} has {len(converted_data)} items')
with open(os.path.join(json_file_folder, f"{name}.json"), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)