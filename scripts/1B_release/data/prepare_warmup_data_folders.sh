#!/bin/bash
# Data, both image folder (in .zip) and instruction (in .json) can be found at https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main
mkdir -p playground/data/warmup_data/instella_alignment
# Download the file using wget
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true -O playground/data/warmup_data/instella_alignment/images.zip
# Alternatively use curl
# curl -L "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true" -o playground/data/warmup_data/instella_alignment/images.zip

# Unzip the file
unzip playground/data/warmup_data/instella_alignment/images.zip -d playground/data/warmup_data/instella_alignment

# Download the instruction (.json)
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json -O playground/data/warmup_data/instella_alignment/blip_laion_cc_sbu_558k.json
# curl -L "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json" -o playground/data/warmup_data/instella_alignment/blip_laion_cc_sbu_558k.json
