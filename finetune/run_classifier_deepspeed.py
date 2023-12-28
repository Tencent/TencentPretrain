"""
This script provides an example to use DeepSpeed for classification.
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

from tencentpretrain.opts import deepspeed_opts
from tencentpretrain.model_loader import *
from finetune.run_classifier import *


def read_tsv(args, path):
    datas = []
    column = {}
    error_num = 0
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    column[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            if len(column) == len(line):
                datas.append(line)
            else:
                error_num += 1
    args.logger.info("{}: read {} lines with {} errors...".format(path, len(datas) - 1, error_num))
    return datas, column

def read_dataset(args, path):
    dataset, columns = [], {}
    for i in range(args.world_size):
        dataset.append([])
    datas, columns = read_tsv(args, path)
    num_of_rank    = math.ceil(1.0 * len(datas) / args.world_size)
    index = 0
    for line_id, line in enumerate(datas):
        tgt = int(line[columns["label"]])
        if args.soft_targets and "logits" in columns.keys():
            soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
        if "text_b" not in columns:  # Sentence classification.
            text_a = line[columns["text_a"]]
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
            seg = [1] * len(src)
        else:  # Sentence-pair classification.
            text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
            src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
            src = src_a + src_b
            seg = [1] * len(src_a) + [2] * len(src_b)

        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
        while len(src) < args.seq_length:
            src.append(PAD_ID)
            seg.append(0)
        if args.soft_targets and "logits" in columns.keys():
            dataset[index].append((src, tgt, seg, 0, soft_tgt))
        else:
            dataset[index].append((src, tgt, seg, 0))
        if (line_id+1) % num_of_rank == 0:
            index += 1
    for i in range(args.world_size):
        while len(dataset[i]) < num_of_rank:
            dataset[i].append(tuple([1 if j == 3 else dataset[0][-1][j] for j in range(len(dataset[0][-1]))]))
    return dataset

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    model.backward(loss)

    model.step()

    return loss

def batch_loader(batch_size, src, tgt, seg, is_pad, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        is_pad_batch = is_pad[i * batch_size : (i + 1) * batch_size]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, is_pad_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, is_pad_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        is_pad_batch = is_pad[instances_num // batch_size * batch_size :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, is_pad_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, is_pad_batch, None

def predict(args, dataset):
    src = torch.LongTensor([example[0] for example in dataset])
    tgt = torch.LongTensor([example[1] for example in dataset])
    seg = torch.LongTensor([example[2] for example in dataset])
    is_pad = torch.LongTensor([example[3] for example in dataset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[4] for example in dataset])
    else:
        soft_tgt = None

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
    # Confusion matrix.
    correct, total = 0, 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    for result in output_list:
        for pred, gold, is_pad in result.tolist():
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
    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / total, correct, total))
    return correct / total, confusion

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build multi-task classification model.
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = Classifier(args)
            model = load_state_dict_block_model(model, args.pretrained_model_path)
    else:
        model = Classifier(args)
        # Load or initialize parameters.
        model = load_block_model(model, args.pretrained_model_path)

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

    trainset = read_dataset(args, args.train_path)[args.rank]
    random.shuffle(trainset)
    instances_num = len(trainset)
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
    
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    is_pad = torch.LongTensor([example[3] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[4] for example in trainset])
    else:
        soft_tgt = None

    args.model = model
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss, result, best_result, best_epoch = 0.0, 0.0, 0.0, 0

    result_tensor = torch.tensor(result).to(args.device)
    if args.rank == 0:
        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info("The number of training instances: {}".format(instances_num))
        args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, is_pad_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, is_pad, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0 and args.rank == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0
        output = predict(args, read_dataset(args, args.dev_path)[args.rank])
        output = torch.as_tensor(output).to(args.device)
        output_list = [torch.zeros_like(output).to(args.device) for _ in range(args.world_size)]
        dist.all_gather(output_list, output)
        if args.rank == 0:
            result = evaluate(args, output_list)
            result_tensor = torch.tensor(result[0]).to(args.device)
        dist.broadcast(result_tensor, 0, async_op=False)
        if result_tensor.float() >= best_result:
            best_result = result_tensor.float().item()
            best_epoch = epoch
        model.save_checkpoint(args.output_model_path, str(epoch))

if __name__ == "__main__":
    main()
