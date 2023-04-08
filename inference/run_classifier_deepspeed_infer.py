"""
  This script provides an example to use DeepSpeed for classification inference.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn
import deepspeed
import torch.distributed as dist


tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)


from tencentpretrain.opts import deepspeed_opts
from inference.run_classifier_infer import *



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    deepspeed.init_distributed()
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = Classifier(args)
            model = _load_state_dict_into_model(model, args.load_model_path)
    else:
        model = Classifier(args)
        model = load_model(model, args.load_model_path)

    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

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

            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu().numpy().tolist()
            prob = nn.Softmax(dim=1)(logits)
            logits = logits.cpu().numpy().tolist()
            prob = prob.cpu().numpy().tolist()
            if rank == 0:
                for j in range(len(pred)):
                    f.write(str(pred[j]))
                    if args.output_logits:
                        f.write("\t" + " ".join([str(v) for v in logits[j]]))
                    if args.output_prob:
                        f.write("\t" + " ".join([str(v) for v in prob[j]]))
                    f.write("\n")


if __name__ == "__main__":
    main()
