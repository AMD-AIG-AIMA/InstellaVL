import json
import os
from copy import deepcopy
from tqdm import tqdm
with open('playground/data/SciGraphQA-295K-train/scrigraphqa.json') as file:
    data = json.load(file)

dataroot = 'playground/data'
image_root = 'SciGraphQA-295K-train/imgs/train'

new_data = []
for item in tqdm(data):
    
    image_path = os.path.join(image_root, item['image_file'])
    if not os.path.exists(os.path.join(dataroot, image_path)):
        print(os.path.join(dataroot, image_path))
        
        
        continue
    new_example = {'id': item['id']}
    new_example['image'] = image_path
    question = item['conversations'][0]['value'].replace('<image>', '').strip()
    question = '<image>\n' + question
    answer = item['conversations'][1]['value']
    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value':answer}]
    new_example['conversations'] = conversations
    try:
        new_data.append(new_example)
        new_example2 = deepcopy(new_example)
        multi_rounds = item['response'].split('\n\nQuestion:')
        new_conversations = []
        # 
        # 
        for r_id, round in enumerate(multi_rounds):

            question, answer = round.split('Answer:')
            question = question.replace('Question:', '').strip()
            answer = answer.strip()
            if r_id == 0:
                new_conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                new_conversations.append({'from': 'human', 'value': question})
            new_conversations.append({'from': 'gpt', 'value': answer})
        new_example2['conversations'] = new_conversations
        new_data.append(new_example2)
    except:
        continue

new_json_file = 'scigraphqa_process.json' 
dst_json_file = os.path.join('playground/data/SciGraphQA-295K-train', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
