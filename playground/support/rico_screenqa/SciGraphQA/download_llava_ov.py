import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser(description="Process Data")



# args = parser.parse_args()

# print('config names:', args.config_names)
# assert len(args.config_names) == len(args.names)

json_file_folder = "playground/data/SciGraphQA-295K-train"
if not os.path.exists(json_file_folder):
        os.makedirs(json_file_folder)

converted_data = []
name = 'scrigraphqa'
    
data = load_dataset("alexshengzhili/SciGraphQA-295K-train" ,split="train")
# data = load_dataset("HuggingFaceM4/the_cauldron", config_name ,split="train")

image_folder = f"playground/data/UniChart/images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)



for da in tqdm(data):
#     # 
#     # 
    json_data = {}
#     
#     
    for k,v in da.items():
         json_data[k] = v
    
#     if da["image"] is not None:
#         da["image"] = da["image"].convert('RGB')
#         json_data["image"] = f"{da['id']}.jpg"
#         img_path = os.path.join(image_folder, json_data["image"])
#         base_dir = os.path.dirname(img_path)
#         if not os.path.exists(base_dir):
#             os.makedirs(base_dir)
#         da["image"].save(os.path.join(image_folder, json_data["image"]))
#     json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)

print(f'{name} has {len(converted_data)} items')
with open(os.path.join(json_file_folder, f"{name}.json"), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)