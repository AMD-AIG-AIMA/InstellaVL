from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os
import tqdm
import json

dataset = load_dataset('CaptionEmporium/coyo-hd-11m-llavanext')

train_set = dataset['train']
dataroot = 'playground/data'
image_folder = 'coyo/images'

if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))


new_data = []
count = 0
for item in tqdm.tqdm(train_set):
    if item['clip_similarity_vitl14'] > 0.28:
        new_item = {}
        url = item['url']
        try:
            response = requests.get(url)
            image_data = BytesIO(response.content)
            image = Image.open(image_data).convert('RGB')
        except:
            print(url)
            continue
        image.save(os.path.join(dataroot, image_folder, '%08d.jpg' % count))
        new_item['id'] = '%08d' % count
        new_item['image'] = os.path.join(image_folder,  '%08d.jpg' % count)
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
        new_data.append(new_item)
        count += 1


new_json_file = 'coyo_process.json' 
if not os.path.exists('playground/data/coyo'):
    os.makedirs('playground/data/coyo')
dst_json_file = os.path.join('playground/data/coyo', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)