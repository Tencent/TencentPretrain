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
import json

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.opts import *
from tencentpretrain.model_loader import _load_state_dict_into_model
from finetune.run_classifier import *

def read_dataset(args, path):
    dataset, columns = [], {}
    for i in range(args.world_size):
        dataset.append([])
    index = 0
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
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
            index += 1
            if index == args.world_size:
                index = 0
    max_data_num_rank_index = 0
    max_data_num = len(dataset[0])
    for i in range(args.world_size):
        if len(dataset[i]) > max_data_num:
            max_data_num_rank_index = i
            max_data_num = len(dataset[i])
    for i in range(args.world_size):
        if len(dataset[i]) < max_data_num:
            dataset[i].append(tuple([1 if j == 3 else dataset[max_data_num_rank_index][-1][j] for j in range(len(dataset[max_data_num_rank_index][-1]))]))
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

class MultitaskClassifier(nn.Module):
    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])

        self.dataset_id = 0

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layers_1[self.dataset_id](output))
        logits = self.output_layers_2[self.dataset_id](output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits

    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id

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

    args.use_mp = False
    
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build multi-task classification model.
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = MultitaskClassifier(args)
            if os.path.isdir(args.pretrained_model_path):
                index_filename = os.path.join(args.pretrained_model_path, "tencentpretrain_model.bin.index.json")
                with open(index_filename, "r") as f:
                    index = json.loads(f.read())
                shard_filenames = sorted(set(index["weight_map"].values()))
                shard_filenames = [os.path.join(args.pretrained_model_path, f) for f in shard_filenames]
                for shard_file in shard_filenames:
                    model = _load_state_dict_into_model(model, shard_file, "")
            elif args.pretrained_model_path:
                    model = _load_state_dict_into_model(model, args.pretrained_model_path, "")
    else:
        model = MultitaskClassifier(args)

        # Load or initialize parameters.
        load_or_initialize_parameters(args, model)

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
        for i, (dataset_id, src_batch, tgt_batch, seg_batch, is_pad_batch) in enumerate(packed_dataset_all):
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
        dist.all_gather(output_list, output, async_op=False)
        if args.rank == 0:
            evaluate(args, output_list)
        model.save_checkpoint(args.output_model_path, str(epoch))
        
if __name__ == "__main__":
    main()