import coco_text
import os
import numpy as np
import json

dataroot = 'playground/data/coco_text'
ct = coco_text.COCO_Text(os.path.join(dataroot, 'cocotext.v2.json'))

def polygon_center(points):
    points = np.array(points)
    return np.mean(points, axis=0)

# Parse and sort annotations based on y-coordinates and then x-coordinates
def sort_annotations_by_position(annotations):
    # Calculate center points for all annotations
    new_annotations = []
    for annotation in annotations:
        if annotation['transcription'] != "###":
            new_annotations.append(annotation)
    annotations = new_annotations
    
    
    for annotation in annotations:
        annotation['center'] = polygon_center(annotation['points'])

    # Sort by y (top-to-bottom), then by x (left-to-right)
    sorted_annotations = sorted(annotations, key=lambda ann: (ann['center'][1], ann['center'][0]))

    # Combine sorted transcriptions
    sentence = ','.join([ann['transcription'] for ann in sorted_annotations])
    return sentence



# 
# 
imgs = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
new_data = []
for img in imgs:
    new_example = {}
    annIds = ct.getAnnIds(imgIds=img)
    anns = ct.loadAnns(annIds)
    value = []
    for ann in anns:
        if ann['legibility'] == 'illegible' or ann['utf8_string'] == "":
            continue
        points = [[ann['bbox'][0], ann['bbox'][1]],
                  [ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1]],
                  [ann['bbox'][0] , ann['bbox'][1] + ann['bbox'][3]],
                  [ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3] ]]
        text = ann['utf8_string']
        value.append({"transcription": text, "points": points})
    image = ct.loadImgs(img)[0]
    image_name = 'train2014/%s' % image['file_name']
    if not os.path.exists(os.path.join(dataroot, image_name)):
        print(image_name)
        
        
        continue
    new_example['id'] = image_name
    new_example['image'] =  "coco_text/" + image_name
     
    sentence = sort_annotations_by_position(value)

    conversations = [{'from': 'human', 'value': '<image>\nOCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. Use a comma to split different parts of text'}, {'from': 'gpt', 'value':sentence}]
    new_example['conversations'] = conversations
    # 
    # 
    new_data.append(new_example)

dst_json_file = os.path.join(dataroot, 'coco_text_process.json')
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)

