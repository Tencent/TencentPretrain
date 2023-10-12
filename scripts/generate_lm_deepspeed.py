"""
  This script provides an example to use DeepSpeed for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.model_loader import _load_state_dict_into_model, load_model
from tencentpretrain.opts import deepspeed_opts
from scripts.generate_lm import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)

    tokenizer_opts(parser)

    deepspeed_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = GenerateLm(args)
            model = _load_state_dict_into_model(model, args.load_model_path)
    else:
        model = GenerateLm(args)
        model = load_model(model, args.load_model_path)
    deepspeed.init_distributed()
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(args.test_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line))
        seg = [1] * len(src)
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
    src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        for i in range(args.seq_length - beginning_length):
            output = model(src_tensor, seg_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            src_tensor = torch.cat([src_tensor, next_token.view(1, 1).to(device)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).to(device)], dim=1)
        if rank == 0:
            f.write(line + "\n")
            tokens = [token_id.item() for token_id in src_tensor[0]]
            if args.tokenizer.sp_model is not None:
                generated_sentence = args.tokenizer.sp_model.decode(tokens)
            else:
                generated_sentence = "".join(args.tokenizer.convert_ids_to_tokens(tokens))

            f.write(generated_sentence)
