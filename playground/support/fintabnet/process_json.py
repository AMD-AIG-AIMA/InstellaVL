import os
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np

prompts = [
        "Kindly produce the HTML table markup corresponding to the provided image.",
        "Could you transform the image into an HTML table structure?",
        "Render the image's contents as an HTML table representation.",
        "Convert the visual table from the image into HTML code.",
        "Translate the image's tabular layout into HTML markup.",
        "Create an HTML table that mirrors the table depicted in the image.",
        "Develop the HTML code that replicates the table shown in the image.",
        "Construct an HTML table based on the layout in the provided image.",
        "Write out the HTML table code that matches the image's table design.",
        "Formulate the HTML table syntax reflecting the image's tabular format."
]


converted_data = []

data = load_dataset("eddtsoi/fintabnet-html", 'tc', split="validation")
root_path = "playground/data/"
image_folder = "fintabnet-html/images/en"
os.makedirs(os.path.join(root_path, image_folder), exist_ok=True)

for idx, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = "%07d" % idx
    

    da["image"].convert('RGB').save(os.path.join(root_path, image_folder, json_data['id'] + '.jpg'), quality=95)
    json_data['image'] = os.path.join(image_folder, json_data['id'] + '.jpg' )
    conv = []
    conv.append({'from': 'human', 'value': "<image>\n" +  prompts[np.random.randint(len(prompts))] })
    conv.append({'from': 'gpt', 'value':  da['html_table']})
    json_data['conversations'] = conv
    converted_data.append(json_data)



new_json_file = 'fintabnet_tc_process.json'
dst_json_file = os.path.join('playground/data/fintabnet-html', new_json_file)
print(len(converted_data))
with open(dst_json_file, 'w+') as f:
    json.dump(converted_data, f, indent=4)