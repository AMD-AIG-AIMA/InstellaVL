import json
from copy import deepcopy

src_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files/docvqa.json'
dst_json_file = 'playground/data/LLaVA-Stage2-Single-Image/json_files/docvqa2.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = deepcopy(item)
    new_convs = []
    for it, conv in enumerate(item['conversations']):
        new_conv = []
        for k, v in conv.items():
            if k == 'source':
                continue
            conv = {}
            if k == 'user':
                conv['from'] ="human"
                conv['value'] = v
                if it == 0:
                    conv['value'] = '<image>\n' + conv['value'] 
                conv['value'] =  conv['value'].strip()  + " \nAnswer the question with a single word (or phrase)"
            elif k == 'assistant':
                conv['from'] = "gpt"
                conv['value'] = v 
            else:
                continue
            # conv['value'] = v
            new_conv.append(conv)
        new_convs += new_conv
        if len(new_convs) == 0:
            print(item)
    # 
    # 
    new_item['conversations'] = new_convs
    new_data.append(new_item)




with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)