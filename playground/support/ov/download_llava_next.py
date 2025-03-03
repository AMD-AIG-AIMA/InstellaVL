import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train")

image_folder = "playground/data/"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        da["image"] = da["image"].convert('RGB')
        json_data["image"] = f"llava_next/{da['id']}.jpg"
        img_path = os.path.join(image_folder, json_data["image"])
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("llava_next.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
