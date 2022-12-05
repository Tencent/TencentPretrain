"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse

import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.layers import *
from tencentpretrain.encoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.model_loader import load_model
from tencentpretrain.opts import model_opts, tokenizer_opts
from scripts.generate_lm import top_k_top_p_filtering
from tencentpretrain.utils.image_tokenizer import *

class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.target = Target()
        self.target.update(LmTarget(args, len(args.tokenizer.vocab)), "lm")

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.lm.output_layer(output)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)



    # Model options.
    model_opts(parser)

    # Inference options.
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--config_path", type=str, required=True,
                    help="Path of the config file.")
    parser.add_argument("--seq_length", type=int, default=48,
                        help="Sequence length.")

    parser.add_argument("--samples_num", type=int, default=10,
                        help="Number of iterations for sampling.")
    parser.add_argument("--prompt", choices=["to_attributes", "to_caption", "to_image"], default="to_attributes",
                        help="Prompt that indicates the output format.")
    parser.add_argument("--text_prefix_path",  type=str, default=None,
                        help="Text prefix for to_attributes and to_image.")
    parser.add_argument("--image_prefix_path",  type=str, default=None,
                        help="Input image path.")
    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)

    tokenizer_opts(parser)

    args = parser.parse_args()

    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.text_prefix = None
    if args.text_prefix_path is not None:
        with open(args.text_prefix_path, "r") as f:
            args.text_prefix = f.readline()

    def preprocess_vqgan(x):
        x = 2.*x - 1.
        return x

    def convert_color(image, channel):
        if channel == 3:
            if image.mode == "RGBA":
                r, g, b, a = image.split()
                image = Image.merge("RGB", (r, g, b))
            elif image.mode != "RGB":
                image = image.convert("RGBA")
                r, g, b, a = image.split()
                image = Image.merge("RGB", (r, g, b))
        elif channel == 1:
            image = image.convert("L")

        return image

    transform = transforms.Compose([
        transforms.Lambda(lambda img: convert_color(img, 3)),
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: preprocess_vqgan(x)),
    ])

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    model = model.to(args.device)
    model.eval()

    vqgan = build_vqgan_model(args)
    vqgan = vqgan.to(args.device)

    prompt = " ".join(args.prompt.split("_"))

    PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 101, 102, 103
    if args.image_prefix_path is None:
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(prompt) + [SEP_TOKEN] +
                                                   args.tokenizer.tokenize(args.text_prefix) + [SEP_TOKEN])
        if len(src) > 64:
            src = src[:64]
        seg = [1] * len(src)
    else:
        image = Image.open(args.image_prefix_path)
        image = transform(image).to(args.device)
        image_token = image_tokenize(vqgan, image)

        p_src =  args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(prompt) + [SEP_TOKEN])
        src = p_src + [i + args.tokenizer.vocab_bias for i in image_token] + [SEP_ID]
        seg = [1] * len(src)

        if args.text_prefix is not None:
            attr_prompt_src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(args.text_prefix))
            src = src + attr_prompt_src
            seg = seg + [2] * len(attr_prompt_src)


    beginning_length = len(src)

    for r in range(args.samples_num):
        src_tensor, seg_tensor = torch.LongTensor([src]), torch.LongTensor([seg])
        for i in range(args.seq_length - beginning_length):
            src_tensor = src_tensor.to(args.device)
            seg_tensor = seg_tensor.to(args.device)
            with torch.no_grad():
                output = model(src_tensor, seg_tensor)

            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[2]], device=args.device)], dim=1)

        if args.image_prefix_path is not None:
            text_id = [str(token_id.item()) for token_id in src_tensor[0][beginning_length:]]
            text_id = " ".join(text_id)
            text_id = text_id.split(str(SEP_ID))[0].strip().split(" ")
            generated_sentence = " ".join(
                args.tokenizer.convert_ids_to_tokens([int(i) for i in text_id])
            )
            print("output " + str(r) + ":" + "\n")
            print(generated_sentence + "\n")
        else:
            image_id = [token_id.item() for token_id in src_tensor[0][beginning_length:]]
            img_length = (args.image_height // args.image_tokenizer["frame_size"]) ** 2
            img_seg = [i - args.tokenizer.vocab_bias for i in image_id[: img_length]]

            image_detokenize(vqgan, img_seg, args.image_tokenizer["image_vocab_size"], False, "output-" + str(r) + ".jpg")
