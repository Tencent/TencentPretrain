"""
This script provides an example to use DeepSpeed for multi-task classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import deepspeed
import torch.distributed as dist

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.opts import *
from finetune.run_classifier_deepspeed import *
from finetune.run_classifier_mt import MultitaskClassifier

def pack_dataset(dataset, dataset_id, batch_size):
    packed_dataset = []
    src_batch, tgt_batch, seg_batch, is_pad_batch = [], [], [], []
    for i, sample in enumerate(dataset):
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])
        seg_batch.append(sample[2])
        is_pad_batch.append(sample[3])
        if (i + 1) % batch_size == 0:
            packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch), torch.LongTensor(is_pad_batch)))
            src_batch, tgt_batch, seg_batch = [], [], []
            continue
    if len(src_batch) > 0:
        packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch), torch.LongTensor(is_pad_batch)))

    return packed_dataset

def predict(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    is_pad = torch.LongTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size

    args.model.eval()

    result = []
    for _, (src_batch, tgt_batch, seg_batch, is_pad_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg, is_pad)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        is_pad_batch = is_pad_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        pad  = is_pad_batch
        for j in range(pred.size()[0]):
            result.append([pred[j], gold[j], pad[j]])
    return result

def evaluate(args, output_list):
    for dataset_id, _ in enumerate(args.dataset_path_list):
        # Confusion matrix.
        correct, total = 0, 0
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
        for result in output_list:
            for pred, gold, is_pad in result.tolist()[dataset_id]:
                if is_pad == 1: continue
                confusion[pred, gold] += 1
                correct += pred == gold
                total += 1
        args.logger.info("Confusion matrix:")
        args.logger.info(confusion)
        args.logger.info("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 2 * p * r / (p + r + eps)
            args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        args.logger.info("Dataset_id: {} Acc. (Correct/Total): {:.4f} ({}/{}) ".format(dataset_id, correct / total, correct, total))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument("--output_model_path", default="models/multitask_classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    # Model options.
    model_opts(parser)

    # Tokenizer options.
    tokenizer_opts(parser)

    # Optimizer options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)

    adv_opts(parser)

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num_list = [count_labels_num(os.path.join(path, "train.tsv")) for path in args.dataset_path_list]
    
    args.datasets_num = len(args.dataset_path_list)
    
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build multi-task classification model.
    model = MultitaskClassifier(args)
    if args.pretrained_model_path:
        load_model(args, model, args.pretrained_model_path)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    # Get logger.
    args.logger = init_logger(args)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    deepspeed.init_distributed()
    rank = dist.get_rank()
    args.rank = rank

    dataset_list = [read_dataset(args, os.path.join(path, "train.tsv"))[args.rank] for path in args.dataset_path_list]
    packed_dataset_list = [pack_dataset(dataset, i, args.batch_size) for i, dataset in enumerate(dataset_list)]
    packed_dataset_all = []
    for packed_dataset in packed_dataset_list:
        packed_dataset_all += packed_dataset

    instances_num = sum([len(dataset) for dataset in dataset_list])
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    custom_optimizer, custom_scheduler = build_optimizer(args, model)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        args=args,
        optimizer=custom_optimizer,
        lr_scheduler=custom_scheduler,
        mpu=None,
        dist_init_required=False)

    args.model = model
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss, result, best_result, best_epoch = 0.0, 0.0, 0.0, 0

    if args.rank == 0:
        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info("The number of training instances: {}".format(instances_num))
        args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(packed_dataset_all)
        model.train()
        for i, (dataset_id, src_batch, tgt_batch, seg_batch, _) in enumerate(packed_dataset_all):
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_id)
            else:
                model.change_dataset(dataset_id)
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, None)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0 and args.rank == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0
        output = []
        for dataset_id, path in enumerate(args.dataset_path_list):
            args.labels_num = args.labels_num_list[dataset_id]
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_id)
            else:
                model.change_dataset(dataset_id)
            result = predict(args, read_dataset(args, os.path.join(path, "dev.tsv"))[args.rank])
            output.append(result)
        output = torch.as_tensor(output).to(args.device)
        output_list = [torch.zeros_like(output).to(args.device) for _ in range(args.world_size)]
        dist.all_gather(output_list, output)
        if args.rank == 0:
            evaluate(args, output_list)
        model.save_checkpoint(args.output_model_path, str(epoch))
        
if __name__ == "__main__":
    main()