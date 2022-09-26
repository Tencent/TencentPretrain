"""
This script provides an example to wrap TencentPretrain for text-to-text inference.
"""
import sys
import os
import random
import argparse
import torch

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.vocab import Vocab
from tencentpretrain.model_loader import load_model
from tencentpretrain.opts import infer_opts, tokenizer_opts
from finetune.run_text2text import Text2text
from inference.run_classifier_infer import batch_loader


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")

            if len(columns) == 2:
                text = line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]]
            else:
                text = line[columns["text_a"]]

            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)

            dataset.append((src, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--tgt_seq_length", type=int, default=32,
                        help="Output sequence length.")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Beam width.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Text2text(args)
    model = load_model(model, args.load_model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)


    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    beam_width = args.beam_width
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        f.write("\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            tgt_in_batch = torch.zeros(src_batch.size()[0], 1, dtype = torch.long, device = args.device)
            current_batch_size = tgt_in_batch.size()[0]
            for j in range(current_batch_size):
                tgt_in_batch[j][-1] = args.tokenizer.vocab.get(CLS_TOKEN)

            with torch.no_grad():
                memory_bank = model(src_batch, None, seg_batch, only_use_encoder=True)
            with torch.no_grad():
                outputs = model(src_batch, (tgt_in_batch, None, src_batch), None, memory_bank=memory_bank)

            current_level = []
            current_level.append([tgt_in_batch, [0 for j in range(current_batch_size)], 0, outputs])
            next_level = {}
            while len(current_level):
                pre_node = current_level.pop(0)
                if pre_node[2] == args.tgt_seq_length:
                    final_result = pre_node
                    break
                next_token_logits = pre_node[3][:, -1]
                next_token_prob = torch.softmax(next_token_logits, dim=-1)
                log_prob, indices = torch.topk(next_token_prob, beam_width, dim=1)

                for k in range(beam_width):
                    index = indices[:, k].unsqueeze(1)
                    log_p = log_prob[:, k].data.cpu().numpy()
                    tgt_in_batch = torch.cat([pre_node[0], index], dim=1)
                    with torch.no_grad():
                        outputs = model(src_batch, (tgt_in_batch, None, src_batch), None, memory_bank=memory_bank)

                    for j in range(current_batch_size):
                        if j not in next_level:
                            next_level[j] = []
                        next_node = [tgt_in_batch[j].unsqueeze(0), [pre_node[1][j]+log_p[j]], pre_node[2]+1, outputs[j].unsqueeze(0)]
                        next_level[j].append([pre_node[1][j]+log_p[j], next_node])
                if len(current_level) == 0:
                    for k, v in next_level.items():
                        next_level[k] = sorted(next_level[k], key=lambda d:d[0], reverse=True)
                    for k in range(beam_width):
                        utterance = []
                        for j in range(current_batch_size):
                            if j not in next_level:
                                continue
                            if k >= len(next_level[j]):
                                break
                            node = next_level[j][k][1]
                            if len(utterance):
                                utterance = [torch.cat([utterance[0], node[0]], dim=0),
                                            utterance[1]+node[1], node[2],
                                            torch.cat([utterance[3], node[3]], dim=0)]
                            else:
                                utterance = node
                        current_level.append(utterance)
                    next_level = {}
            for j in range(len(final_result[-1])):
                f.write("".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in final_result[0][j][1:]])
                        .split(SEP_TOKEN)[0])
                f.write("\n")


if __name__ == "__main__":
    main()
