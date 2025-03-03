import glob
import yaml

json_folder = '/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage2-Single-Image-New/json_files_processed_2'
files = glob.glob(json_folder + '/*.json')
data = {'datasets': []}
for file in files:
    sampling_strategy = 'all'
    if 'k12' in file or 'clevr' in file or 'dvqa' in file or 'figureqa' in file:
        sampling_strategy = "first:1%"
    elif 'hme100k' in file or 'tallyqa' in file:
        sampling_strategy = "first:10%"
    elif "scienceqa_nona_context" in file or "FigureQA" in file or 'IconQA' in file or 'raven' in file or 'tqa.json' in file or 'PMC-VQA' in file or 'iconqa' in file:
        sampling_strategy = "first:5%"
    elif "Evol" in file:
        sampling_strategy = "first:30%"
    elif 'magpie' in file:
        sampling_strategy = "first:50%"
    



    dataset = {'json_path': file, 'sampling_strategy': sampling_strategy}
    data['datasets'].append(dataset)

output_file  = '/home/ximensun/code/LLaVA-NeXT/scripts/train/LLaVA-Stage2-Single-Image-dataset-new.yaml'
with open(output_file, 'w+') as file:
    yaml.dump(data, file, default_flow_style=False)