from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

# Load dataset
dataset = load_dataset('CaptionEmporium/coyo-hd-11m-llavanext')
train_set = dataset['train']

# Paths and directories
dataroot = 'playground/data'
image_folder = 'coyo/images'
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

# Prepare the output list
new_data = []
count = 0
new_json_file = 'coyo_process.json'
if not os.path.exists('playground/data/coyo'):
    os.makedirs('playground/data/coyo')
dst_json_file = os.path.join('playground/data/coyo', new_json_file)

# Function to download and process a single image
def download_and_process(item, count):
    if item['clip_similarity_vitl14'] <= 0.28:
        return None

    new_item = {}
    url = item['url']
    try:
        response = requests.get(url, timeout=10)
        image_data = BytesIO(response.content)
        image = Image.open(image_data).convert('RGB')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

    # Save image
    image_path = os.path.join(dataroot, image_folder, f'{count:08d}.jpg')
    image.save(image_path)

    # Prepare item information
    new_item['id'] = f'{count:08d}'
    new_item['image'] = os.path.join(image_folder, f'{count:08d}.jpg')
    new_item['image_size'] = image.size
    new_item['clip_similarity_vitb32'] = item['clip_similarity_vitb32']
    new_item['clip_similarity_vitl14'] = item['clip_similarity_vitl14']
    new_item['nsfw_score_opennsfw2'] = item['nsfw_score_opennsfw2']
    new_item['nsfw_score_gantman'] = item['nsfw_score_gantman']
    new_item['watermark_score'] = item['watermark_score']
    
    conv = []
    conv.append({'from': 'human', 'value': '<image>\n' + 'Please take the following image caption and attempt to distill it into a single sentence. Remove any redundant lines or descriptions and make it a maximum of 30 words in length.\n'})
    conv.append({'from': 'gpt', 'value': item['caption_llava']})
    new_item['conversations'] = conv

    return new_item

# Use ThreadPoolExecutor for multi-threading
max_workers = 10  # Adjust the number of threads as needed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks to the executor
    future_to_count = {executor.submit(download_and_process, item, idx): idx for idx, item in enumerate(train_set)}
    
    # Use tqdm to show progress
    for future in tqdm.tqdm(as_completed(future_to_count), total=len(train_set)):
        result = future.result()
        if result is not None:
            new_data.append(result)

# Save the processed data to a JSON file
print(f"Processed {len(new_data)} items.")
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
