import sys
import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, tencentpretrain_dir)

from tencentpretrain.opts import deepspeed_opts
from scripts.generate_seq2seq import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tgt_vocab_path", type=str,
                        help="Path of the vocabulary file.")
    tokenizer_opts(parser)
    parser.add_argument("--tgt_tokenizer", choices=[None, "bert", "char", "space", "xlmroberta"], default=None,
                        help="Specify the tokenizer for target side.")
    parser.add_argument("--tgt_seq_length", type=int, default=128,
                        help="Sequence length.")
    deepspeed_opts(parser)

    args = parser.parse_args()

    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    if args.tgt_tokenizer == None:
        args.tgt_tokenizer = args.tokenizer
    else:
        args.vocab_path = args.tgt_vocab_path
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args)
        args.tgt_vocab = args.tgt_tokenizer.vocab
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = GenerateSeq2seq(args)
            model = _load_state_dict_into_model(model, args.load_model_path)
    else:
        model = GenerateSeq2seq(args)
        model = load_model(model, args.load_model_path)
    deepspeed.init_distributed()
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    with open(args.test_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line) + [SEP_TOKEN])
        seg = [1] * len(src)
        tgt = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
    src_tensor, seg_tensor, tgt_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device), torch.LongTensor([tgt]).to(device)

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        for i in range(args.tgt_seq_length-1):
            output = model(src_tensor, seg_tensor, tgt_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            tgt_tensor = torch.cat([tgt_tensor, next_token.view(1, 1).to(device)], dim=1)
        if rank == 0:
            f.write(line + "\n")
            generated_sentence = "".join(
                args.tgt_tokenizer.convert_ids_to_tokens([token_id.item() for token_id in tgt_tensor[0]])
            )
            f.write(generated_sentence)
