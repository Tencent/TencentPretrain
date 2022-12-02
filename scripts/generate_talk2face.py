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
from tencentpretrain.opts import infer_opts, tokenizer_opts
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--prompt", choices=["to_attributes", "to_caption", "to_image"], default="to_attributes",
                        help="Prompt that indicates the output format.")
    parser.add_argument("--text_prefix",  type=str, default=None,
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

    def preprocess_vqgan(x):
        x = 2.*x - 1.
        return x

    def convert_color(image, channel):
        if channel == 3:
            if image.mode == 'RGBA':
                r, g, b, a = image.split()
                image = Image.merge("RGB", (r, g, b))
            elif image.mode != 'RGB':
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

    vqgan = build_vqgan_model(args)
    vqgan = vqgan.to(args.device)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)

    model = model.to(args.device)

    model.eval()
    caption = "1"
    image_path = None

    # This person has no smile, and no eyeglasses. She is in her thirties and has no bangs.
    # This guy doesn't have any beard and has no fringe, and no glasses. He is in his middle age and has no smile.
    # She looks serious with no smile in the face. She is wearing eyeglasses.
    # This gentleman is in the thirties. He smiles with corners of his mouth turned up.

    prompt = ' '.join(args.prompt.split('_'))
    # This person is attractive and has blond hair, mouth slightly open, and arched eyebrows.
    #attribute = "wiki age : 10 ;"# gender : female ; blond hair : true ;"
    #image_path = "/apdcephfs/share_1157269/yudongli/UER_dev/vision_lm/TencentPretrain/ft_local/oba.png" # 202599.jpg 000051.jpg
    #image_path = "/apdcephfs/share_1157269/yudongli/UER_dev/vision_lm/TencentPretrain/output-1.jpg"


    PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 101, 102, 103
    if args.image_prefix_path is None:
        # to image

        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(prompt) + [SEP_TOKEN] +
                                                   args.tokenizer.tokenize(args.text_prefix) + [SEP_TOKEN])
        if len(src) > 64:
            src = src[:64]
        seg = [1] * len(src)
    else:
        # to attributes and to caption
        image = Image.open(args.image_prefix_path)
        image = transform(image).to(args.device)
        image_token = image_tokenize(vqgan, image)

        p_src =  args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(prompt) + [SEP_TOKEN])
        src = p_src + [i + len(args.tokenizer.vocab) for i in image_token] + [SEP_ID]
        seg = [1] * len(src)

        if args.text_prefix is not None:
            attr_prompt_src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(args.text_prefix))
            src = src + attr_prompt_src
            seg = seg + [2] * len(attr_prompt_src)


    beginning_length = len(src)

    src_tensor, seg_tensor = torch.LongTensor([src]), torch.LongTensor([seg])

    for i in range(args.seq_length - beginning_length):
        src_tensor = src_tensor.to(args.device)
        seg_tensor = seg_tensor.to(args.device)
        with torch.no_grad():
            output = model(src_tensor, seg_tensor)

        #next_token_logits = output[:, -1]
        #next_token = torch.argmax(next_token_logits, dim=1).unsqueeze(1)
        next_token_logits = output[0][-1] / args.temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

        src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
        seg_tensor = torch.cat([seg_tensor, torch.tensor([[2]], device=args.device)], dim=1)

    #print(src_tensor,seg_tensor)
    if image_path is not None:
        text_id = [str(token_id.item()) for token_id in src_tensor[0][beginning_length:]]
        text_id = " ".join(text_id)
        #print(text_id)
        text_id = text_id.split(str(SEP_ID))[0].strip().split(' ')
        generated_sentence = " ".join(
            args.tokenizer.convert_ids_to_tokens([int(i) for i in text_id])
        )
        print(generated_sentence)
    else:
        image_id = [token_id.item() for token_id in src_tensor[0][beginning_length:]]
        img_length = (args.image_height // args.vqgan_frame) ** 2
        #print(" ".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in output[1:]]))
        img_seg = [i - args.text_vocab_size for i in image_id[: img_length]]
        print(len(img_seg), img_seg)

        image_detokenize(vqgan, img_seg, args.image_tokenizer['image_vocab_size'], False, 'output.jpg')
