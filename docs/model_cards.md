---
license: openrail
---
# Instella-VL-1B ✨
Welcome to the official repository for **Instella-VL-1B**, AMD's first ever Vision-Language Model (VLM). This repository provides a detailed guide for training and inference with **Instella-VL-1B**. Developed from AMD's **Instella-1B** (previously known as [AMD OLMo 1B SFT](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html) LLM), this model is fully open-source, with both model weights and training code available for AMD GPUs (MI300). Its compact size aims to make it accessible to a broad spectrum of researchers, developers, and enthusiasts, enabling them to build upon, modify, and integrate it into their own projects.

[[GitHub](https://github.com/AMD-AIG-AIMA/InstellaVL)][[Blog](https://github.com/AMD-AIG-AIMA/InstellaVL/blog/blog-final.md)]

## Main Results
We compare our model with models which only releases the model weights (with * in the below table) and also models which releases weights, data curvation and all training details.

<table class="tg"><thead>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow">DeepSeek-VL-1.3B *</td>
    <td class="tg-c3ow">InternVL2-1B *</td>
    <td class="tg-c3ow">InternVL2.5-1B *</td>
    <td class="tg-c3ow">TinyLLaVA-2.4B</td>
    <td class="tg-c3ow">TinyLLaVA-1.5B</td>
    <td class="tg-c3ow">llava-onevision-1b</td>
    <td class="tg-c3ow">MiniCPM-V-2</td>
    <td class="tg-c3ow">Instella-VL-1B</td>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">GQA</td>
    <td class="tg-c3ow">--</td>
    <td class="tg-c3ow">55.06</td>
    <td class="tg-c3ow">56.66</td>
    <td class="tg-c3ow">61.58</td>
    <td class="tg-c3ow">60.28</td>
    <td class="tg-c3ow">57.95</td>
    <td class="tg-c3ow">--</td>
    <td class="tg-c3ow">61.52</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SQA</td>
    <td class="tg-c3ow">64.52</td>
    <td class="tg-c3ow">89.54</td>
    <td class="tg-c3ow">93.90</td>
    <td class="tg-c3ow">64.30</td>
    <td class="tg-c3ow">59.69</td>
    <td class="tg-c3ow">59.25</td>
    <td class="tg-c3ow">76.10</td>
    <td class="tg-c3ow">83.74</td>
  </tr>
  <tr>
    <td class="tg-c3ow">POPE</td>
    <td class="tg-c3ow">85.80</td>
    <td class="tg-c3ow">87.40</td>
    <td class="tg-c3ow">89.95</td>
    <td class="tg-c3ow">85.66</td>
    <td class="tg-c3ow">84.77</td>
    <td class="tg-c3ow">87.17</td>
    <td class="tg-c3ow">86.56</td>
    <td class="tg-c3ow">86.73</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MM-Bench</td>
    <td class="tg-c3ow">64.34</td>
    <td class="tg-c3ow">61.70</td>
    <td class="tg-c3ow">68.40</td>
    <td class="tg-c3ow">58.16</td>
    <td class="tg-c3ow">51.28</td>
    <td class="tg-c3ow">44.60</td>
    <td class="tg-c3ow">70.44</td>
    <td class="tg-c3ow">69.17</td>
  </tr>
  <tr>
    <td class="tg-c3ow">seedbench</td>
    <td class="tg-c3ow">65.94</td>
    <td class="tg-c3ow">65.90</td>
    <td class="tg-c3ow">71.30</td>
    <td class="tg-c3ow">63.30</td>
    <td class="tg-c3ow">60.04</td>
    <td class="tg-c3ow">65.43</td>
    <td class="tg-c3ow">66.90</td>
    <td class="tg-c3ow">68.47</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MMMU</td>
    <td class="tg-c3ow">28.67</td>
    <td class="tg-c3ow">32.40</td>
    <td class="tg-c3ow">35.60</td>
    <td class="tg-c3ow">32.11</td>
    <td class="tg-c3ow">29.89</td>
    <td class="tg-c3ow">30.90</td>
    <td class="tg-c3ow">38.55</td>
    <td class="tg-c3ow">29.30</td>
  </tr>
  <tr>
    <td class="tg-c3ow">realworldqa</td>
    <td class="tg-c3ow">50.20</td>
    <td class="tg-c3ow">51.90</td>
    <td class="tg-c3ow">58.30</td>
    <td class="tg-c3ow">52.42</td>
    <td class="tg-c3ow">46.67</td>
    <td class="tg-c3ow">51.63</td>
    <td class="tg-c3ow">55.03</td>
    <td class="tg-c3ow">58.82</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mmstar</td>
    <td class="tg-c3ow">38.30</td>
    <td class="tg-c3ow">46.18</td>
    <td class="tg-c3ow">47.93</td>
    <td class="tg-c3ow">37.17</td>
    <td class="tg-c3ow">31.87</td>
    <td class="tg-c3ow">37.38</td>
    <td class="tg-c3ow">40.93</td>
    <td class="tg-c3ow">43.21</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:bold">Average</span></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">61.26</td>
    <td class="tg-c3ow">65.26</td>
    <td class="tg-c3ow">56.84</td>
    <td class="tg-c3ow">53.06</td>
    <td class="tg-c3ow">54.29</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">62.62</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ocrbench</td>
    <td class="tg-c3ow">41.40</td>
    <td class="tg-c3ow">74.40</td>
    <td class="tg-c3ow">74.20</td>
    <td class="tg-c3ow">28.90</td>
    <td class="tg-c3ow">34.40</td>
    <td class="tg-c3ow">43.00</td>
    <td class="tg-c3ow">60.00</td>
    <td class="tg-c3ow">67.90</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TextVQA</td>
    <td class="tg-c3ow">57.54</td>
    <td class="tg-c3ow">69.60</td>
    <td class="tg-c3ow">72.96</td>
    <td class="tg-c3ow">47.05</td>
    <td class="tg-c3ow">49.54</td>
    <td class="tg-c3ow">49.54</td>
    <td class="tg-c3ow">74.23</td>
    <td class="tg-c3ow">71.23</td>
  </tr>
  <tr>
    <td class="tg-c3ow">AI2D</td>
    <td class="tg-c3ow">51.13</td>
    <td class="tg-c3ow">62.40</td>
    <td class="tg-c3ow">67.58</td>
    <td class="tg-c3ow">49.58</td>
    <td class="tg-c3ow">43.10</td>
    <td class="tg-c3ow">57.35</td>
    <td class="tg-c3ow">64.40</td>
    <td class="tg-c3ow">66.65</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ChartQA</td>
    <td class="tg-c3ow">47.40</td>
    <td class="tg-c3ow">71.52</td>
    <td class="tg-c3ow">75.76</td>
    <td class="tg-c3ow">12.96</td>
    <td class="tg-c3ow">15.24</td>
    <td class="tg-c3ow">61.24</td>
    <td class="tg-c3ow">59.80</td>
    <td class="tg-c3ow">72.52</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DocVQA</td>
    <td class="tg-c3ow">35.70</td>
    <td class="tg-c3ow">80.94</td>
    <td class="tg-c3ow">82.76</td>
    <td class="tg-c3ow">25.82</td>
    <td class="tg-c3ow">30.38</td>
    <td class="tg-c3ow">71.22</td>
    <td class="tg-c3ow">69.54</td>
    <td class="tg-c3ow">80.30</td>
  </tr>
  <tr>
    <td class="tg-c3ow">InfoVQA</td>
    <td class="tg-c3ow">20.52</td>
    <td class="tg-c3ow">46.30</td>
    <td class="tg-c3ow">53.62</td>
    <td class="tg-c3ow">21.35</td>
    <td class="tg-c3ow">24.46</td>
    <td class="tg-c3ow">41.18</td>
    <td class="tg-c3ow">38.24</td>
    <td class="tg-c3ow">46.40</td>
  </tr>
  <tr>
    <td class="tg-c3ow">OCR Average</td>
    <td class="tg-c3ow">42.28</td>
    <td class="tg-c3ow">67.53</td>
    <td class="tg-c3ow">71.15</td>
    <td class="tg-c3ow">30.94</td>
    <td class="tg-c3ow">32.85</td>
    <td class="tg-c3ow">53.92</td>
    <td class="tg-c3ow">61.04</td>
    <td class="tg-c3ow">67.50</td>
  </tr>
</tbody></table>

## Inference

> [!NOTE]
> Follow [inference_requirements.txt](../inference_requirements.txt) for setting up the environment.

### Loading from locally saved checkpoint

> [!NOTE]
> Additionally do `pip install -e . --no-deps` to register/include for InstellaVL repo as `instellavl` package into python package list.

``` python
import torch

# Import essential modules
from instellavl.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from instellavl.conversation import conv_templates, SeparatorStyle
from instellavl.model.builder import load_pretrained_model
from instellavl.utils import disable_torch_init
from instellavl.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from io import BytesIO

# Login into HF Hub
from huggingface_hub import login
login(token = "<Your HFtoken id>") # Enter your token 

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

#
# ========= CHANGE IMAGE and Query only HERE ============
image_file = '/path/to/Instella-VL-repo/assets/images/example2_dog.jpg' # Enter the test image path
query = 'Describe this image.'
# =======================================================

disable_torch_init()
conv_mode = 'instella'

# Model loading
model_path = '<path/to/model-checkpoint-saved-locally>' # Enter your model path, should contain instellavl substring in the name.
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, False, False)
model.eval()
model = model.to('cuda') # change to 'cpu' if not 'cuda'

# Image pre-processing
image = load_image(image_file)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(model.dtype)

# Text pre-processing - follow the below logic too when there is no Image:
# if images is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in text:
#     question = DEFAULT_IMAGE_TOKEN + "\n" + text
# else:
#     question = text
query = query.replace(DEFAULT_IMAGE_TOKEN, "").strip()
question = DEFAULT_IMAGE_TOKEN + "\n" + query
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Final arrangements required
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
keywords = [conv.sep]
image_sizes = [image.size]
stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("|||IP_ADDRESS|||")]

with torch.inference_mode():
    output_ids = model.generate(input_ids.to(model.device), images=image_tensor.to(model.device), image_sizes=image_sizes, do_sample=True, num_beams=1, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=stopping_criteria, eos_token_id=terminators)

outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
print("InstellaVL: ", outputs)
``` 
### Load Model from Huggingface

```python
import os
os.environ['HF_TOKEN']="<Your HFtoken id>"
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM

from PIL import Image
import requests
from io import BytesIO

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


config = AutoConfig.from_pretrained("AIG-GenAI/Instella-VL-1B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("AIG-GenAI/Instella-VL-1B", config=config, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("AIG-GenAI/Instella-VL-1B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("AIG-GenAI/Instella-VL-1B", trust_remote_code=True).to('cuda') # or 'cpu'
model.eval()
  
# For single image and text
query="Describe the image."
image=load_image("path/to/your_image")
out = processor.encode(query, image, model.get_vision_tower().image_processor, tokenizer, config)
inputs = {k: v.to(model.device) for k, v in out.items() if isinstance(v, torch.Tensor)}
with torch.inference_mode():
    output_ids = model.generate(inputs["input_ids"], images=inputs['image_tensor'], image_sizes=out['image_sizes'], do_sample=True, num_beams=1, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=out['stopping_criteria'], eos_token_id=out['eos_token_id'])
outputs = processor.decode(output_ids)
print("InstellaVL: ", outputs)

# For batch of images and text.
query=["Describe the image.", "What is the color of the dog?"]
image=[load_image("../assets/images/instellavl.png"), load_image("../assets/images/example2_dog.jpg")]
outs = processor.batch_encode(query, image, model.get_vision_tower().image_processor, tokenizer, config)

for idx, o in enumerate(outs):
    ins = {k: v.to(model.device) for k, v in o.items() if isinstance(v, torch.Tensor)}
    with torch.inference_mode():
        output_ids = model.generate(ins["input_ids"],
                                    images=ins['image_tensor'],
                                    image_sizes=o['image_sizes'],
                                    do_sample=True,
                                    num_beams=1,
                                    temperature=0.2,
                                    max_new_tokens=1024,
                                    use_cache=True,
                                    stopping_criteria=o['stopping_criteria'],
                                    eos_token_id=o['eos_token_id'])
    outputs = processor.decode(output_ids)
    print("Query: ", query[idx])
    print("InstellaVL: ", outputs)
```

## Model Architecture

| Parts        | Parameter size   | Number of layers  | Number of heads	| Hidden size	| Patch Size  |
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| Vision Encoder | 300M | 24|  16 | 1024 | 14 |
| MLP | 6.3M | 2 | - | 2048 | - |
| LM | 1.2B | 16 |	16 |	2048 |	- |

We initialize the vision encoder from [CLIP-ViT-L/14@336](https://huggingface.co/openai/clip-vit-large-patch14-336) and initialize LM from [AMD OLMo 1B SFT](https://huggingface.co/AIG-GenAI/AMD-OLMo-1B-SFT)

## Training Stages

| Stages        | MLP Warmup           | Pretraining  | Instruction Tuning  |
| ------------- |:-------------:|:-----:|:-----:|
| Tunable Parts | Adapter | Entire Model | Entire Model |

## Hardware
Training was conducted with up to 4 nodes, totaling 32 GPUs. Each node comprises [8 AMD Instinct™ MI300X GPUs](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) 

**MLP warmup**: 1 node  
**Pretraining**: 2 nodes  
**Finetune**: 4 nodes 

## Datasets

### MLP Warmup
[BLIP558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

### Pretraining Stage

| Domain        | Datasets           | Num of Examples  |
| ------------- |:-------------:| -----:|
| Image Captions     | [BLIP150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), [COCO118K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), [CC3M-Recap](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M),  [Pixmo_Cap](https://huggingface.co/datasets/allenai/pixmo-cap)  | 3.52M |
| OCR      | [SynthDog_EN](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data), [SynthDog_ZH](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data), [UReader](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data), [ART](https://rrc.cvc.uab.es/?ch=14&com=downloads), [COCO-Text](https://bgshih.github.io/cocotext/), [HierText](https://github.com/google-research-datasets/hiertext), [Uber-Text](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html), [TextOCR](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [OpenVINO](https://github.com/openvinotoolkit/cvat), [MLT-17](https://rrc.cvc.uab.es/?ch=8&com=downloads)  |   913K  |
| Doc |  [DocVQA](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [DocStruct4M](https://huggingface.co/datasets/mPLUG/DocStruct4M)  | 410K |
| Table & Chart & Plot | [Chart2Text](https://github.com/vis-nlp/Chart-to-text/tree/main/pew_dataset/dataset/imgs), [UniChart](https://huggingface.co/datasets/ahmed-masry/unichart-pretrain-data), [PlotQA](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [WidgetCaption](https://huggingface.co/datasets/rootsautomation/RICO-WidgetCaptioning?row=0), [Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words), [SciGraphQA-295K](https://huggingface.co/datasets/alexshengzhili/SciGraphQA-295K-train), [Paper2Fig100K](https://zenodo.org/records/7299423#.Y2lzonbMKUl), [MMC Instruction](https://huggingface.co/datasets/xywang1/MMC/viewer/MMC-Instruction), [M-Paper](https://huggingface.co/datasets/mPLUG/M-Paper)  |  1.97M  |
| Text Only | [Evol-Instruct-GPT-4](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/evol_instruct)      |  70K |


### Instruction-tuning Stage
| Domain        | Datasets           | Num of Examples  |
| ------------- |:-------------:| -----:|
|General     |  [AOKVQA, CLEVR, Hateful Memes, Image Textualization, OKVQA, ScienceQA, ShareGPT-4V, TallyQA, Visual7W, VizWiz, VQAv2, WebSight, ALLaVA Instruct, Cambrian, COCO Caption, IconQA, LLaVA-158K, LLaVAR, RefCOCO, ShareGPT-4O, Vision FLAN, VisText, VQARAD, VSR, InterGPS](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [Image-Paragraph-Captioning, ImageNet, COCO-GOI, COCO-ITM, Visual Dialog, SNLI-VE](https://huggingface.co/datasets/MMInstruction/M3IT), [Web-Landmark, Web-Celebrity, SAM, LAION-GPT-4V-Dataset, OODVQA]( https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/tree/main), [Pixmo_Cap](https://huggingface.co/datasets/allenai/pixmo-cap), [Pixmo_Count](https://huggingface.co/datasets/allenai/pixmo-count), [Pixmo_Points](https://huggingface.co/datasets/allenai/pixmo-points), [Pixmo_Ask_Model_Anything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything),   [SVIT_Core_150K](https://huggingface.co/datasets/BAAI/SVIT), [Localized Narratives](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)  |  2.66M |
| Table & Chart & Screen | [AI2D, ChartQA, DocVQA, FigureQA, InfographicVQA, RoBUT-SQA, RoBUT-WTQ, TQA, UReader IE, UReader QA, Chart2Text, , Diagram Image2Text, DVQA, HiTab, LRV Chart, RoBUT WikiSQL, Screen2Words, UReader Caption, UReader KG, VisualMRC](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [TinyChartData](https://huggingface.co/datasets/mPLUG/TinyChartData) | 866K | 
| Doc | [ArxivQA](https://huggingface.co/datasets/MMInstruction/ArxivQA), [DocDownstream-1.0](https://huggingface.co/datasets/mPLUG/DocDownstream-1.0), [DocReason25K](https://huggingface.co/datasets/mPLUG/DocReason25K), [DocStruct4M](https://huggingface.co/datasets/mPLUG/DocStruct4M), [Pixmo_Docs](https://huggingface.co/datasets/allenai/pixmo-docs) | 522K |
| General OCR   | [ChromeWriting, IIIT5K, K12 Printing, Rendered Text, TextCaps, HME100K, IAM, TextOCR-GPT-4V](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [SynthDog-EN](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data)   | 84K |
| Math & Reasoning      |  [MAVIS Manual Collection, CLEVR-Math, Geo170K QA, GEOS, GeoMVerse, MapQA, Super-CLEVR, UniGeo, LRV Normal, Visual Genome, MAVIS Data Engine, Geo170K Align, Geometry3K, GeoQA+, TabMWP, GQA, RAVEN, MathVision, KVQA, VCR](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), [FinQA](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), [Design2Code, IDK](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/) | 460K |
| Others | 	[IQA, MOCHEG, Shapes](https://huggingface.co/datasets/MMInstruction/M3IT), [ALFWorld, Q-Instruct-DB](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/) | 479K |
| Text Only    | [MathQA, Magpie Pro (L3 MT), Magpie Pro (Qwen2 ST), Magpie Pro (L3 ST)](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)    | 480K |


For the details of training hyperparameters, please check [our github repo](https://github.com/AMD-AIG-AIMA/AMD-LLaVA-NeXT/tree/instellavl)

## Contributors
**Core contributors:** [Ximeng Sun](https://sunxm2357.github.io/), [Aditya Kumar Singh](https://rodosingh.github.io), [Gowtham Ramesh](https://www.linkedin.com/in/gowtham1/), [Zicheng Liu](https://zicliu.wixsite.com/mysite) 

**Contributors:** [Pratik Brahma](https://www.linkedin.com/in/pratik-p-brahma/), [Ze Wang](https://www.linkedin.com/in/ze-wang-1379601a5/), [Jiang Liu](https://joellliu.github.io/), [Jialian Wu](https://jialianwu.com/), [Prakamya Mishra](https://prakamya-mishra.github.io/), [Xiaodong Yu](https://www.xiaodongyu.me/), [Yusheng Su](https://yushengsu-thu.github.io/), [Sudhanshu Ranjan](https://www.linkedin.com/in/sudhanshu-ranjan-33a216124), [Emad Barsoum](https://www.linkedin.com/in/ebarsoum/)

##  Bias, Risks, and Limitations
This model is made accessible without any safety guarantees. Users should be aware that the model may generate outputs that are sensitive, inaccurate, harmful, biased, or otherwise objectionable based on user prompts. It is crucial for users to conduct comprehensive safety evaluations, implement safety filtering, and verify the model's outputs to mitigate these risks.

##  License
Please refer to our license [here](https://huggingface.co/AIG-GenAI/Instella-VL-1B/blob/main/LICENSE)

##  Citing

```bibtex
@misc{Instella-VL-1B, 
    title = {Instella-VL-1B-1.0: AMD’s first Vision language model}, 
    url = {https://huggingface.co/AIG-GenAI/Instella-VL-1B}, 
    author = {Ximeng Sun, Aditya Singh, Gowtham Ramesh, Jiang Liu, Ze Wang, Sudhanshu Ranjan, Pratik Brahma, Prakamya Mishra,  Jialian Wu, Xiaodong Yu, Yusheng Su, Emad Barsoum, Zicheng Liu}, 
    month = {February}, 
    year = {2025} 
} 
```