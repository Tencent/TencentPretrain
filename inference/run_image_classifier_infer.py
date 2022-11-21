"""
  This script provides an example to wrap TencentPretrain for image classification inference.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.model_loader import load_model
from tencentpretrain.opts import infer_opts, tokenizer_opts
from tencentpretrain.utils.misc import ZeroOneNormalize
from finetune.run_classifier import Classifier


def data_loader(args, path):

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])

    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        src_batch, seg_batch = [], []
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            path = line[columns["path"]]
            image = read_image(path, ImageReadMode.RGB)
            image =  image.to(args.device)
            src = transform(image)
            seg = [1] * ((src.size()[1] // args.patch_size) * (src.size()[2] // args.patch_size) + 1)

            src_batch.append(src)
            seg_batch.append(seg)

            if len(src_batch) == args.batch_size:
                yield torch.stack(src_batch, 0), \
                      torch.LongTensor(seg_batch)
                src_batch, seg_batch = [], []

        if len(src_batch) > 0:
            yield torch.stack(src_batch, 0), \
                  torch.LongTensor(seg_batch)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer["virtual"](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\n")
        for i, (src_batch, seg_batch) in enumerate(data_loader(args, args.test_path)):
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)

            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu().numpy().tolist()
            prob = nn.Softmax(dim=1)(logits)
            logits = logits.cpu().numpy().tolist()
            prob = prob.cpu().numpy().tolist()

            for j in range(len(pred)):
                f.write(str(pred[j]))
                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits[j]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob[j]]))
                f.write("\n")


if __name__ == "__main__":
    main()
