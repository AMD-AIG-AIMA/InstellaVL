import glob
import yaml
import os


set_all = ["cambrian_selection", "VisualWebInstruct", "ai2d_gpt4v", "infographic_vqa", "infographic_gpt4v",
           "lrv_chart", "vision_flan", 'mathqa', "geo3k", "sharegpt4o", 'sharegpt4v', "GEOS", "ai2d_original",
           "diagram_image_to_text", "m4_instruct_annotations"]
set_10 = ["mavis_math_metagen", "mavis_math_rule_geo", 'llavar', "lrv_normal", "textocr", "ai2d_internvl",
          "textcaps", "ureader_qa_processed", "ureader_cap", "ureader_ie", "ureader_kg_processed",
          "geo170k_qa", "geo170k_align", "CLEVR-Math", "FigureQA", "Geometry3K_MathV360K", "GeoQA+",  "MapQA", "PlotQA",
          "Super-CLEVR", "TabMWP", "TQA", "UniGeo", "vizwiz", "VQA-AS", "VQA-RAD", "chart2text", "chartqa", "hateful_memes",
          "hitab", "iam", "infographic_vqa_llava_format", "intergps", "mapqa", "rendered_text", "robut_sqa", "robut_wikisql",
          "screen2words", "st_vqa", "visual7w", "visualmrc", "vqarad", "vsr", "vistext", "websight"]
set_50 = ["llava_next"]
set_30 = ["allava_instruct"]
set_20 = ['image_textualization']
set_5 = ["IconQA", "tabmwp", "tallyqa", "tqa"]
set_1 = ["PMC-VQA"]
set_end_20 = ["magpie_pro_l3_80b_mt", "magpie_pro_l3_80b_st", "magpie_pro_qwen2_72b_st"]

json_folder = '/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage2-OneVision/json_files_processed'
files = glob.glob(json_folder + '/*.json')
data = {'datasets': []}
for file in files:
    name = os.path.basename(file).split('.')[0]
    if name in set_all:
        sampling_strategy = "all"
    elif name in set_10:
        sampling_strategy = "first:10%"
    elif name in set_50:
        sampling_strategy = "first:50%"
    elif name in set_30:
        sampling_strategy = "first:30%"
    elif name in set_20:
        sampling_strategy = "first:20%"
    elif name in set_5:
        sampling_strategy = "first:5%"
    elif name in set_1:
        sampling_strategy = "first:1%"
    elif name in set_end_20:
        sampling_strategy = "end:20%"
    else:
        
        
        raise ValueError

    dataset = {'json_path': file, 'sampling_strategy': sampling_strategy}
    data['datasets'].append(dataset)

output_file  = '/home/ximensun/code/LLaVA-NeXT/scripts/train/LLaVA-Stage2-OneVision.yaml'
with open(output_file, 'w+') as file:
    yaml.dump(data, file, default_flow_style=False)