from PIL import Image
import os
import glob
import json
import numpy as np
from tqdm import tqdm

root_path = 'playground/data/'
image_path = 'coyo-hd-11m_webdataset'


def list_directories(path):
    # List all directories in the specified path
    directories = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    return directories

def get_jpg_files(path):
    # Find all .jpg and .JPG files
    jpg_files = glob.glob(os.path.join(path, '*.jpg')) + \
                glob.glob(os.path.join(path, '*.JPG'))
    return jpg_files

def get_json_files(path):
    # Find all .jpg and .JPG files
    json_files = glob.glob(os.path.join(path, '*.json'))
    return json_files

# Example usage
# path = '.'  # Current directory
dirs = list_directories(os.path.join(root_path, image_path))
# print(dirs)

new_data = []
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


for folder in tqdm(dirs):
    # print(folder)
    image_folder = os.path.join(root_path, image_path, folder)
    json_files = get_json_files(image_folder)
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        caption = data['caption']
        item = {}
        item['id'] = data['key']
        item['image'] = os.path.join(image_path, folder, item['id'] + '.jpg')
        if not os.path.exists(os.path.join(root_path, item['image'])):
            print(os.path.join(root_path, item['image']))
            continue
        conv = []
        instruction = prompts[np.random.randint(len(prompts))]
        conv.append({'from': 'human', 'value': '<image>\n' + instruction})
        conv.append({'from': 'gpt', 'value': caption})
        item['conversations'] = conv
        new_data.append(item)

json_file = '%s.json' % image_path 
if not os.path.exists(os.path.join(root_path, image_path)):
    os.makedirs(os.path.join(root_path, image_path))
dst_json_file = os.path.join(root_path, image_path, json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)

