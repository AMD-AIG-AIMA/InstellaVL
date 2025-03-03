import json

file = open('playground/data/data/LLaVA-Stage1.5/Rico_ScreenQA/Rico_ScreenQA.json_new')

data = json.load(file)

for item in data:
    item['image'] = item['image'].replace('RICO-ScreenQA', 'Rico_ScreenQA')


# dst_json_file = os.path.join(dataroot, new_json_file)
# print(len(new_data))
# with open(dst_json_file, 'w+') as f:
#     json.dump(new_data, f, indent=4)
with open('playground/data/data/LLaVA-Stage1.5/Rico_ScreenQA/Rico_ScreenQA.json_new', 'w') as f:
    json.dump(data, f, indent=4)