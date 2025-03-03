import os
from PIL import Image
import yaml
import json
import tqdm
import concurrent.futures

image_file = '/home/ximensun/code/LLaVA-NeXT/scripts/train/LLaVA-MidStage_3.yaml'
image_dataroot = '/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5'

num_piles = 3

import matplotlib.pyplot as plt
base_pixel = 336
output_folder = 'image_size_plot'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process each item
def process_item(item, image_dataroot):
    if not item.get('image', None):
        return None
    if item.get('image_size', None):
        image_size = item['image_size']
    else:
        image_path = os.path.join(image_dataroot, item['image'])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None
        image_size = image.size

    scale = max(image_size) / 2000.
    image_size = [int(image_size[0] / scale), int(image_size[1] / scale)]
    image = image.resize((image_size[0], image_size[1]), 0)
    new_image_path = image_path.replace('M-Paper', 'M-Paper-2K')
    image_folder = os.path.dirname(new_image_path)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image.save(new_image_path)
    return image_size

if image_file.endswith('yaml'):
    with open(image_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    datasets = yaml_data.get("datasets")
    
    json_files = []
    for dataset in datasets:
        json_path = dataset.get("json_path")
        json_files.append(json_path)
else:
    json_files = [image_file]

# Multi-threading part
heights = []
widths = []
piles = []
max_workers = 32  # You can adjust this based on your system's capabilities

for json_path in json_files:
    with open(json_path, "r") as json_file:
        cur_data_dict = json.load(json_file)
        # Use ThreadPoolExecutor to process the items concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, item, image_dataroot) for item in cur_data_dict]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    widths.append(result[0])
                    heights.append(result[1])

