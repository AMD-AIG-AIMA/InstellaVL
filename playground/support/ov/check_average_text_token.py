import json
from copy import deepcopy
import os
import argparse
import glob
import tqdm
import transformers 

files = glob.glob('playground/data/LLaVA-Stage2-OneVision/json_files_processed/*.json')

tokenizer = transformers.AutoTokenizer.from_pretrained(
            "AIG-GenAI/AMD-OLMo-1B-SFT",
            cache_dir=None,
            model_max_length=32768,
            token="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",
        )

num_tokens_all = []

for json_file in files:
    print(json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)

    for a, item in enumerate(data):
        if 'conversations' in item.keys():
            human_word = ""
            for it, conv in enumerate(item['conversations']):
                if conv['from'] not in ['human', 'gpt']:
                    print(conv['from'])
                if conv['from'] == 'human':
                    human_word += conv['value']
            human_word = human_word.replace('<image>', '')
            tokens = tokenizer.tokenize(human_word)

            # Number of tokens
            num_tokens = len(tokens)
            num_tokens_all.append(num_tokens)

average = sum(num_tokens_all) / float(len(num_tokens_all))

# Print the result
print(f"The average is: {average}")