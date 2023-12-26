"""
  This script provides an example to use DeepSpeed for multi-task classification inference.
"""
import sys
import os
import torch
import argparse
import torch.nn as nn
import deepspeed
import torch.distributed as dist
import json

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.opts import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.model_loader import _load_state_dict_into_model, load_model
from finetune.run_classifier_mt_deepspeed import read_dataset
from inference.run_classifier_mt_infer import MultitaskClassifier
from tencentpretrain.utils import *

def batch_loader(batch_size, src, seg, is_pad):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        is_pad_batch = is_pad[i * batch_size : (i + 1) * batch_size]
        yield src_batch, seg_batch, is_pad_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        is_pad_batch = is_pad[instances_num // batch_size * batch_size :]
        yield src_batch, seg_batch, is_pad_batch

def predict(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    is_pad = torch.LongTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size

    args.model.eval()

    result = []
    for i, (src_batch, seg_batch, is_pad_batch) in enumerate(batch_loader(batch_size, src, seg, is_pad)):
        src_batch = src_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        is_pad_batch = is_pad_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, None, seg_batch)
        logits = torch.stack(logits)
        is_pad_batch = is_pad_batch.view(1,is_pad_batch.shape[0],1).repeat(logits.shape[0],1,1)
        logits_all = torch.cat((logits, is_pad_batch), dim=-1)
        result.append(logits_all)
    result = torch.cat(result, dim=1)
    return result

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    log_opts(parser)

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    args.use_mp = False

    # Build multi-task classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    deepspeed.init_distributed()
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = MultitaskClassifier(args)
            if os.path.isdir(args.load_model_path):
                index_filename = os.path.join(args.load_model_path, "tencentpretrain_model.bin.index.json")
                with open(index_filename, "r") as f:
                    index = json.loads(f.read())
                shard_filenames = sorted(set(index["weight_map"].values()))
                shard_filenames = [os.path.join(args.load_model_path, f) for f in shard_filenames]
                for shard_file in shard_filenames:
                    model = _load_state_dict_into_model(model, shard_file, "")
            elif args.load_model_path:
                    model = _load_state_dict_into_model(model, args.load_model_path, "")
    else:
        model = MultitaskClassifier(args)
        model = load_model(model, args.load_model_path)
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]
    args.model = model

    args.rank = dist.get_rank()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = predict(args, read_dataset(args, args.test_path)[args.rank])
    output = torch.as_tensor(dataset).to(args.device)
    output_list = [torch.zeros_like(output).to(args.device) for _ in range(args.world_size)]
    dist.all_gather(output_list, output, async_op=False)

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        if args.rank == 0:
            f.write("label")
            if args.output_logits:
                f.write("\t" + "logits")
            if args.output_prob:
                f.write("\t" + "prob")
            f.write("\n")
        for logits_all in output_list:
            logits = logits_all[:,:,:-1]
            is_pad = logits_all[0,:,-1]

            pred = [torch.argmax(logits_i, dim=-1) for logits_i in logits]
            prob = [nn.Softmax(dim=-1)(logits_i) for logits_i in logits]
            logits = [x.cpu().numpy().tolist() for x in logits]
            pred   = [x.cpu().numpy().tolist() for x in pred]
            prob   = [x.cpu().numpy().tolist() for x in prob]
            pad    = [x.cpu().numpy().tolist() for x in is_pad]
            for j in range(len(pred[0])):
                if pad[j] == 1:
                    continue
                if args.rank == 0:
                    f.write("|".join([str(v[j]) for v in pred]))
                    if args.output_logits:
                        f.write("\t" + "|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in logits]))
                    if args.output_prob:
                        f.write("\t" + "|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in prob]))
                    f.write("\n")
        f.close()


if __name__ == "__main__":
    main()
