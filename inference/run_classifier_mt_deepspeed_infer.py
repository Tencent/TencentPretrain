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

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.opts import *
from inference.run_classifier_mt_infer import *
from tencentpretrain.utils.config import load_hyperparam

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

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
            model = _load_state_dict_into_model(model, args.load_model_path)
    else:
        model = MultitaskClassifier(args)
        model = load_model(model, args.load_model_path)
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: {0}".format(instances_num))

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        if rank == 0:
            f.write("label")
            if args.output_logits:
                f.write("\t" + "logits")
            if args.output_prob:
                f.write("\t" + "prob")
            f.write("\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            
            pred = [torch.argmax(logits_i, dim=-1) for logits_i in logits]
            prob = [nn.Softmax(dim=-1)(logits_i) for logits_i in logits]

            logits = [x.cpu().numpy().tolist() for x in logits]
            pred = [x.cpu().numpy().tolist() for x in pred]
            prob = [x.cpu().numpy().tolist() for x in prob]
            if rank == 0:
                for j in range(len(pred[0])):
                    f.write("|".join([str(v[j]) for v in pred]))
                    if args.output_logits:
                        f.write("\t" + "|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in logits]))
                    if args.output_prob:
                        f.write("\t" + "|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in prob]))
                    f.write("\n")
        f.close()


if __name__ == "__main__":
    main()
