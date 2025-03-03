import json
import os

dataroot = 'playground/data/sft_oct24'
img_folder = 'KVQA/raw'
file = open(os.path.join(dataroot, img_folder, 'dataset.json'))
dataset = json.load(file)

new_data = []
for key, item in dataset.items():
    new_item = {}
    img_path = item['imgPath']
    if not os.path.exists(os.path.join(dataroot, img_folder, img_path)):
        print(os.path.join(dataroot, img_folder, img_path))
        
        
    new_item['image'] = os.path.join(img_folder, img_path)
    new_item['id'] = key
    questions = item['Questions']
    answers = item['Answers']
    assert len(questions) == len(answers)
    conv = []
    conv_id = 0
    for question, answer in zip(questions, answers):
        if conv_id == 0:
            conv.append({'from': 'human', 'value': '<image>\n' + question})
        else:
            conv.append({'from': 'human', 'value':  question})
        conv.append({'from': 'gpt', 'value': str(answer)})
        conv_id += 1
    new_item['conversations'] = conv
    new_data.append(new_item)

new_json_file = 'kvqa_process.json' 
dst_json_file = os.path.join('playground/data/sft_oct24/KVQA', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4) 


