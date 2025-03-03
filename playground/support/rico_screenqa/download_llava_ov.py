import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser(description="Process Data")



json_file_folder = "playground/data/LLaVA-Stage2-OneVision/json_files"
if not os.path.exists(json_file_folder):
        os.makedirs(json_file_folder)

converted_data = []
name = 'Rico_ScreenQA.json'
    
data = load_dataset("rootsautomation/RICO-ScreenQA" ,split="train")
# data = load_dataset("HuggingFaceM4/the_cauldron", config_name ,split="train")

image_folder = f"playground/data/Rico_ScreenQA/"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


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
    json_data["conversations"] = [{'from': "human", "value": "<image>\n%s" % da['question']},
                                  {'from': 'gpt', "value": da['ground_truth'][0]['full_answer']}]
    converted_data.append(json_data)

print(f'{name} has {len(converted_data)} items')
with open(os.path.join(json_file_folder, f"{name}.json"), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)