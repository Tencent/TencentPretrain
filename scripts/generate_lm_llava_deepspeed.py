"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import imghdr
import deepspeed

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.opts import infer_opts, tokenizer_opts, log_opts, mp_opts
from tencentpretrain.opts import deepspeed_opts
from tencentpretrain.utils.logging import init_logger
from tencentpretrain.model_loader import _load_state_dict_into_model, load_model
from tencentpretrain.utils.misc import pooling, ZeroOneNormalize


class LLaVaGenerate(nn.Module):
    def __init__(self, args):
        super(LLaVaGenerate, self).__init__()
        self.args = args
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

        self.target = Target()
        self.target.update(LmTarget(args, len(args.tokenizer.vocab)), "lm")
        print("tokenizer vocab nums:", len(args.tokenizer.vocab))

    def forward(self, src_text, seg_text, src_image, seg_image, image_pos):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        src = src_text, src_image, seg_text, seg_image, image_pos
        emb = self.embedding(src, None)
        seg = torch.cat((seg_image[:,1:], seg_text), 1)
        # encoder
        output = self.encoder(emb, seg)
        # # Target.
        output = self.target.lm.output_layer(output)
        return output


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        args.logger.info("loading model from {0}".format(args.pretrained_model_path))
        keys_info = model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
        args.logger.info("missing_keys: {0}".format(keys_info.missing_keys))
        args.logger.info("unexpected_keys: {0}".format(keys_info.unexpected_keys))
        if args.vision_model_in_VL_emb_path is not None:
            args.logger.info("loading model from {0}".format(args.vision_model_in_VL_emb_path))   
            model = load_model(model, args.vision_model_in_VL_emb_path, missing_prefix="embedding.vision_language.vision_")
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vision_model_in_VL_emb_path", type=str, default=None,
                        help="Path of the vision pretrained model in the vision language embedding.")
    parser.add_argument("--instruction_template", type=str, choices=["sys0", "sys1", "sys2", "sys3", "sys4"],
                        help="The instruction type for training large language-vision model.", default="sys0")

    tokenizer_opts(parser)

    deepspeed_opts(parser)

    log_opts(parser)

    mp_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    args.logger = init_logger(args)

    args.pretrained_model_path = args.load_model_path

    # Load or initialize parameters.
    if args.enable_zero3:
        print("enable_zero3:", args.enable_zero3)
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = LLaVaGenerate(args)
            if args.pretrained_model_path:
                model = _load_state_dict_into_model(model, args.pretrained_model_path)
            if args.vision_model_in_VL_emb_path is not None:
                model = _load_state_dict_into_model(model, args.vision_model_in_VL_emb_path, missing_prefix="embedding.vision_language.vision_")
    else:
        model = LLaVaGenerate(args)
        load_or_initialize_parameters(args, model)

    deepspeed.init_distributed()
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    image_height = args.vision_language_emb["vision_encoder"]["image_height"]
    image_width = args.vision_language_emb["vision_encoder"]["image_width"]
    patch_size = args.vision_language_emb["vision_encoder"]["patch_size"]

    transform = transforms.Compose([
        transforms.Resize(min(image_height, image_width)),
        transforms.CenterCrop((image_height, image_width)),
        ZeroOneNormalize(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    prompt_template = {
        "sys0": "",
        "sys1": "<<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n",
        "sys2": "<<SYS>>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<</SYS>>\n\n",
        "sys3": "<SYS> You are a helpful language and vision assistant. </SYS> \n",
        "sys4": "[INST]<<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"
    }
    role1, role2 = "USER", "ASSISTANT"
    im_start, im_end = "<Image>", "</Image>"
    num_image_tokens = int(image_width / patch_size) * int(image_height / patch_size) + 1 # 336/14-14 --> 576 dim + 1
    seq_text = args.seq_length - num_image_tokens
    outf = open(args.prediction_path, mode="w", encoding="utf-8")
    input_f = open(args.test_path, mode="r", encoding="utf-8")
    datas = json.load(input_f)
    try:
        prompt_overall = prompt_template[args.instruction_template]
    except:
        args.logger.info("unsupported prompt template!")
        NotImplementedError
    for line_id, item in enumerate(datas):
        try:
            id = item["id"]
            image_path = "datasets/llava/" + item["image"]
            if not os.path.isfile(image_path):
                continue
            if imghdr.what(image_path) != 'jpeg' and imghdr.what(image_path) != 'png':
                continue
            image = read_image(image_path, ImageReadMode.RGB)
            image = image.to(device)
            src_image = transform(image)
        except:
            print("sth wrong with item{}".format(item))
            continue

        prompt_before_image = prompt_overall + " " + role1 + ": "
        ground_truth = []
        prompt_answer_id = []
        if "conversations" in item:
            conversations = item["conversations"]
            for i, conv in enumerate(conversations):
                # 1 round
                if i > 1:
                    continue
                if i == 0:
                    prompt = conv["value"]
                    if prompt.endswith("<image>"):
                        prompt_before_image = prompt_before_image + prompt.replace("<image>", im_start)
                        prompt_after_image = im_end + "\n" + role2 + ": "
                    elif prompt.startswith("<image>"):
                        prompt_before_image = prompt_before_image + im_start
                        prompt_after_image = prompt.replace("<image>", im_end) + "\n" + role2 + ": "
                    else:
                        prompt_before_image = prompt_before_image + im_start
                        prompt_after_image = im_end + "\n" + prompt + " " + role2 + ": "

                    prompt_before_image_id = args.tokenizer.convert_tokens_to_ids(
                        args.tokenizer.tokenize(prompt_before_image)
                    )
                    prompt_after_image_id = args.tokenizer.convert_tokens_to_ids(
                        args.tokenizer.tokenize(prompt_after_image)
                    )
                    seg_before_image = [1] * len(prompt_before_image_id)
                    seg_after_image = [1] * len(prompt_after_image_id)
                    if len(prompt_before_image_id) + len(prompt_after_image_id) > seq_text:
                        args.logger.info("promt too long, jump for now")
                        break
                    prompt_answer_id = [prompt_before_image_id + prompt_after_image_id]
                    prompt_answer_seg = [seg_before_image + seg_after_image]
                elif i % 2 == 0: # human
                    prompt = conv["value"]
                    prompt_id = args.tokenizer.convert_tokens_to_ids(
                        args.tokenizer.tokenize(role1 + ": " + prompt + " " + role2 + ": ")
                    )
                    if prompt_answer_id:
                        prompt_answer_id.append(prompt_id)
                        prompt_answer_seg.append(prompt_answer_seg + [1] * len(prompt_id))
                    else:
                        args.logger.info("no prompt, or prompt too long, jumping")
                        break
                else: # gpt
                    ground_truth.append(conv["value"])
        else:
            prompt = item["instruction"]
            prompt_before_image = prompt_before_image + im_start
            prompt_after_image = im_end + "\n" + prompt + "\n" + role2 + ": "
            prompt_before_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_before_image)
            )
            prompt_after_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_after_image)
            )
            seg_before_image = [1] * len(prompt_before_image_id)
            seg_after_image = [1] * len(prompt_after_image_id)
            if len(prompt_before_image_id) + len(prompt_after_image_id) > seq_text:
                args.logger.info("promt too long, jump for now")
                break
            prompt_answer_id = [prompt_before_image_id + prompt_after_image_id]
            prompt_answer_seg = [seg_before_image + seg_after_image]

        image_pos = len(prompt_before_image_id)
        
        image_tensor = torch.unsqueeze(src_image, 0).half()
        image_seg_tensor = torch.ones(1, num_image_tokens).to(device)
        image_pos = torch.LongTensor([image_pos]).to(device)
        SEP_ID = args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])
        text_tensor = None

        for i, prompt in enumerate(prompt_answer_id):
            if text_tensor is None:
                text_tensor, text_seg_tensor = torch.LongTensor([prompt]).to(device), torch.LongTensor([prompt_answer_seg[i]]).to(device)
            else:
                text_tensor = torch.cat([text_tensor, torch.LongTensor([prompt]).to(device)], dim=1)
                text_seg_tensor = torch.cat([text_seg_tensor, torch.LongTensor([prompt_answer_seg[i]]).to(device)], dim=1)

            while text_tensor.shape[1] + num_image_tokens <= args.seq_length:
                output = model(text_tensor, text_seg_tensor, image_tensor, image_seg_tensor, image_pos)
                next_token_logits = output[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                text_tensor = torch.cat([text_tensor, next_token.view(1, 1)], dim=1)
                text_seg_tensor = torch.cat([text_seg_tensor, torch.tensor([[1]]).to(device)], dim=1)
                if next_token.cpu().tolist() == SEP_ID:
                    break

        if rank == 0 and text_tensor is not None:
            tokens = [token_id.item() for token_id in text_tensor[0]]
            if args.tokenizer.sp_model is not None:
                generated_sentence = args.tokenizer.sp_model.decode(tokens)
            else:
                generated_sentence = "".join(args.tokenizer.convert_ids_to_tokens(tokens))
            print(item)
            print(generated_sentence)
            print(generated_sentence+ "\n\n", file=outf)
