"""
This script provides an example to wrap TencentPretrain for multi-task classification.
"""
import sys
import os
import random, json
import argparse
import torch
import torch.nn as nn

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.utils.logging import init_logger
from tencentpretrain.utils.misc import pooling
from tencentpretrain.model_saver import save_model
from tencentpretrain.opts import *
from finetune.run_classifier import count_labels_num, batch_loader, build_optimizer, load_or_initialize_parameters, train_model
from finetune.run_classifier import evaluate as evaluate_cls
from finetune.run_classifier import read_dataset as read_dataset_cls
from finetune.run_classifier_multi_label import evaluate as evaluate_ml
from finetune.run_classifier_multi_label import read_dataset as read_dataset_ml
from finetune.run_classifier_multi_label import count_labels_num as count_ml_labels_num
from finetune.run_ner import evaluate as evaluate_ner
from finetune.run_ner import read_dataset as read_dataset_ner


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

        if self.type in ["cls", "ml"]:
            # Target.
            output = pooling(output, seg, self.pooling_type)
            output = torch.tanh(self.output_layers_1[self.dataset_id](output))
            logits = self.output_layers_2[self.dataset_id](output)

            if tgt is not None:
                if self.type == "ml":
                    probs_batch = nn.Sigmoid()(logits)
                    loss = nn.BCELoss()(probs_batch, tgt)
                else:
                    loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                return loss, logits
            else:
                return None, logits

        elif self.type == "ner":
            logits = self.output_layers_2[self.dataset_id](output)
            tgt_mask = seg.contiguous().view(-1).float()
            logits = logits.contiguous().view(-1, self.labels_num)
            pred = logits.argmax(dim=-1)
            if tgt is not None:
                tgt = tgt.contiguous().view(-1, 1)
                one_hot = torch.zeros(tgt.size(0), self.labels_num). \
                    to(torch.device(tgt.device)). \
                    scatter_(1, tgt, 1.0)
                numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-6
                loss = numerator / denominator
                return loss, pred
            else:
                return None, pred

    def change_dataset(self, args, dataset_id):
        self.type = args.id2type[dataset_id]
        self.dataset_id = dataset_id
        if self.type == "ner":
            info_id = dataset_id - len(args.cls_dataset_path_list + args.clsml_dataset_path_list)
            args.l2i = args.ner_info[info_id][0]
            args.begin_ids = args.ner_info[info_id][1]
            self.labels_num = args.ner_info[info_id][2]


def pack_dataset(args, dataset, dataset_id, batch_size):
    packed_dataset = []
    src_batch, tgt_batch, seg_batch = [], [], []
    for i, sample in enumerate(dataset):
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])
        seg_batch.append(sample[2])
        if (i + 1) % batch_size == 0:
            if args.id2type[dataset_id] == "ml":
                packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.tensor(tgt_batch, dtype=torch.float), torch.LongTensor(seg_batch)))
            else:
                packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))
            src_batch, tgt_batch, seg_batch = [], [], []
            continue
    if len(src_batch) > 0:
        if args.id2type[dataset_id] == "ml":
            packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.tensor(tgt_batch, dtype=torch.float), torch.LongTensor(seg_batch)))
        else:
            packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))

    return packed_dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--cls_dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument("--clsml_dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument("--ner_dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument("--output_model_path", default="models/multitask_classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Tokenizer options.
    tokenizer_opts(parser)

    # Optimizer options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)

    adv_opts(parser)

    args = parser.parse_args()

    args.soft_targets = False

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of classification labels.
    args.cls_labels_num_list = [count_labels_num(os.path.join(path, "train.tsv")) for path in args.cls_dataset_path_list]
    args.ml_labels_num_list = [count_ml_labels_num(os.path.join(path, "train.tsv")) for path in args.clsml_dataset_path_list]

    # Count the number of NER labels.
    args.ner_labels_num_list = []
    args.ner_info = {}
    for i, path in enumerate(args.ner_dataset_path_list):
        begin_ids = []

        with open(os.path.join(path, "label2id.json"), mode="r", encoding="utf-8") as f:
            l2i = json.load(f)
            l2i["[PAD]"] = len(l2i)
            for label in l2i:
                if label.startswith("B"):
                    begin_ids.append(l2i[label])
        args.ner_info[i] = [l2i, begin_ids, len(l2i)]
        args.ner_labels_num_list.append(len(l2i))


    args.datasets_num = len(args.cls_labels_num_list + args.ml_labels_num_list + args.ner_labels_num_list)
    args.labels_num_list = args.cls_labels_num_list + args.ml_labels_num_list + args.ner_labels_num_list
    print(args.labels_num_list)

    # Build tokenizer.
    args.cls_tokenizer = CharTokenizer(args)
    args.ner_tokenizer = SpaceTokenizer(args)
    args.tokenizer = args.cls_tokenizer
    # Build multi-task classification model.
    model = MultitaskClassifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    # Training phase.
    args.tokenizer = args.cls_tokenizer
    cls_dataset_list = [read_dataset_cls(args, os.path.join(path, "train.tsv")) for path in args.cls_dataset_path_list]

    ml_dataset_list = []
    for i, path in enumerate(args.clsml_dataset_path_list):
        args.labels_num = args.ml_labels_num_list[i]
        ml_dataset_list.append(read_dataset_ml(args, os.path.join(path, "train.tsv")))

    ner_dataset_list = []
    args.tokenizer = args.ner_tokenizer
    for i, path in enumerate(args.ner_dataset_path_list):
        args.labels_num = args.ner_labels_num_list[i]
        args.l2i = args.ner_info[i][0]
        ner_dataset_list.append(read_dataset_ner(args, os.path.join(path, "train.tsv")))


    args.id2type = {}
    for i in range(len(cls_dataset_list)):
        args.id2type[i] = "cls"
    for i in range(len(args.id2type.keys()), len(args.id2type.keys()) + len(ml_dataset_list)):
        args.id2type[i] = "ml"
    for i in range(len(args.id2type.keys()), len(args.id2type.keys()) + len(ner_dataset_list)):
        args.id2type[i] = "ner"

    print(args.id2type)

    packed_dataset_list = [pack_dataset(args, dataset, i, args.batch_size) for i, dataset in enumerate(cls_dataset_list + ml_dataset_list + ner_dataset_list)]

    packed_dataset_all = []
    for packed_dataset in packed_dataset_list:
        packed_dataset_all += packed_dataset

    instances_num = sum([len(dataset) for dataset in cls_dataset_list + ml_dataset_list + ner_dataset_list])
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(packed_dataset_all)
        model.train()
        for i, (dataset_id, src_batch, tgt_batch, seg_batch) in enumerate(packed_dataset_all):
            if hasattr(model, "module"):
                model.module.change_dataset(args, dataset_id)
            else:
                model.change_dataset(args, dataset_id)
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, None)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

    save_model(model, args.output_model_path)

    for dataset_id, path in enumerate(args.cls_dataset_path_list + args.clsml_dataset_path_list + args.ner_dataset_path_list):
        args.logger.info(str(dataset_id) + ' --------- ' +path)
        args.labels_num = args.labels_num_list[dataset_id]
        if hasattr(model, "module"):
            model.module.change_dataset(args, dataset_id)
        else:
            model.change_dataset(args, dataset_id)
        if args.id2type[dataset_id] == "cls":
            args.tokenizer = args.cls_tokenizer
            evaluate_cls(args, read_dataset_cls(args, os.path.join(path, "dev.tsv")))
        elif args.id2type[dataset_id] == "ml":
            args.tokenizer = args.cls_tokenizer
            evaluate_ml(args, read_dataset_ml(args, os.path.join(path, "dev.tsv")))
        elif args.id2type[dataset_id] == "ner":
            args.tokenizer = args.ner_tokenizer
            args.l2i = args.ner_info[dataset_id - len(args.cls_labels_num_list + args.ml_labels_num_list)][0]
            evaluate_ner(args, read_dataset_ner(args, os.path.join(path, "dev.tsv")))



if __name__ == "__main__":
    main()
