"""
This script provides an exmaple to wrap TencentPretrain for image classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dest
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.layers import *
from tencentpretrain.encoders import *
from tencentpretrain.utils.vocab import Vocab
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.misc import ZeroOneNormalize, count_lines
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.model_saver import save_model
from tencentpretrain.opts import finetune_opts
from finetune.run_classifier import *


def data_loader(args, path):

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])

    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        src_batch, tgt_batch, seg_batch = [], [], []
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])
            path = line[columns["path"]]
            image = read_image(path, ImageReadMode.RGB)
            image = image.to(args.device)
            src = transform(image)
            seg = [1] * ((src.size()[1] // args.patch_size) * (src.size()[2] // args.patch_size) + 1)

            src_batch.append(src)
            tgt_batch.append(tgt)
            seg_batch.append(seg)

            if len(src_batch) == args.batch_size:
                yield torch.stack(src_batch, 0), \
                      torch.LongTensor(tgt_batch), \
                      torch.LongTensor(seg_batch)
                src_batch, tgt_batch, seg_batch = [], [], []

        if len(src_batch) > 0:
            yield torch.stack(src_batch, 0), \
                  torch.LongTensor(tgt_batch), \
                  torch.LongTensor(seg_batch)


def evaluate(args, dataset_path):

    correct, instances_num = 0, 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch) in enumerate(data_loader(args, dataset_path)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
        instances_num += len(pred)

    args.logger.info("Confusion matrix:")
    args.logger.info(confusion)
    args.logger.info("Report precision, recall, and f1:")

    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))
    return correct / instances_num, confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    args.soft_targets, args.soft_alpha = False, 0

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)
    instances_num = count_lines(args.train_path) - 1


    # Build tokenizer.
    args.tokenizer = str2tokenizer["virtual"](args)
    set_seed(args.seed)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))
    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(data_loader(args, args.train_path)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, args.dev_path)
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, args.test_path)

if __name__ == "__main__":
    main()
