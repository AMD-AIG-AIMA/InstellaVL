import torch
import argparse
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from huggingface_hub import login
login(token = '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')

from instellavl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from instellavl.conversation import conv_templates, SeparatorStyle
from instellavl.model.builder import load_pretrained_model
from instellavl.utils import disable_torch_init
from instellavl.mm_utils import tokenizer_image_token,  KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base=None, model_name=args.model_name, load_4bit=args.load_8bit, load_8bit=args.load_4bit)

    conv_mode = 'instella'
 

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()

    roles = conv.roles

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.INSTELLA else conv.sep2
        if 'instellavl' in args.model_name.lower():
            keywords = []
        else:
            keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("|||IP_ADDRESS|||")]
        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria], eos_token_id=terminators)

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="AIG-GenAI/Instella-VL-1B")
    parser.add_argument("--model-name", type=str, default="instellavl-1b")

    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
