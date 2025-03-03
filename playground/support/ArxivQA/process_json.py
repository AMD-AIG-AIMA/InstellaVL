import json
import os

dataroot = 'playground/data'
img_folder = 'ArxivQA'

with open(os.path.join(dataroot, 'ArxivQA' ,"arxivqa.jsonl"), 'r') as fr:
    arxiv_qa = [ json.loads(line.strip()) for line in fr]

# sample = arxiv_qa[0]
# print(sample["image"]) # image file 
new_data = []
for item in arxiv_qa:
    new_item = {}
    new_item['id'] = item['id']
    if not os.path.exists(os.path.join(dataroot, img_folder, item['image'])):
        print(os.path.join(dataroot, img_folder, item['image']))
        
        
    new_item['image'] = os.path.join(img_folder, item['image'])
    question = '<image>\n' + 'First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n ' + item['question'] + '\n'
    for option in item['options']:
        question += (option + '\n')
    question = question.strip()
    answer = item['rationale'] + ' Answer: %s.' % item['label']
    conv = []
    conv.append({'from': 'human', 'value': question})
    conv.append({'from': 'gpt', 'value': answer})
    new_item['conversations'] = conv
    new_data.append(new_item)

new_json_file = 'arxivqa_val_process.json'
data_path = os.path.join(dataroot, img_folder)
if not os.path.exists(data_path):
    os.makedirs(data_path)
dst_json_file = os.path.join(data_path, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)