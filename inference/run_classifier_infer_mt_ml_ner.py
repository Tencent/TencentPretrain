"""
  This script provides an example to wrap TencentPretrain for classification inference.
"""
import sys
import os
import torch
import argparse, json
import collections
import torch.nn as nn

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.model_loader import load_model
from tencentpretrain.opts import infer_opts, tokenizer_opts
from tencentpretrain.utils.misc import pooling


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

        self.cls_task_num = len(args.cls_labels_num_list)
        self.ml_task_num = len(args.clsml_labels_num_list)
        self.ner_task_num = len(args.ner_labels_num_list)
        self.labels_num_list = args.labels_num_list

    def forward(self, src, tgt, seg):
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
        last_hidden = pooling(output, seg, self.pooling_type)
        cls_logits, ml_logits, ner_pred = [], [], []
        for dataset_id in range(0, self.cls_task_num):
            output_1 = torch.tanh(self.output_layers_1[dataset_id](last_hidden))
            cls_logits.append(self.output_layers_2[dataset_id](output_1))

        for dataset_id in range(self.cls_task_num, self.cls_task_num + self.ml_task_num):
            output_1 = torch.tanh(self.output_layers_1[dataset_id](last_hidden))
            ml_logits.append(self.output_layers_2[dataset_id](output_1))

        for dataset_id in range(self.cls_task_num + self.ml_task_num, self.cls_task_num + self.ml_task_num + self.ner_task_num):
            logits = self.output_layers_2[dataset_id](output)
            logits = logits.contiguous().view(-1, self.labels_num_list[dataset_id])
            pred = logits.argmax(dim=-1)
            ner_pred.append(pred)

        return cls_logits, ml_logits, ner_pred, last_hidden


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, seg_batch


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                line = line.rstrip("\r\n").split("\t")
                for i, column_name in enumerate(line):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                seg = [1] * len(src)
            else:  # Sentence pair classification.
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
            dataset.append((src, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--output_last_hidden", action="store_true", help="Write hidden state of last layer to output file.")
    parser.add_argument("--cls_labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    parser.add_argument("--clsml_labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    parser.add_argument("--ner_labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = CharTokenizer(args)
    args.labels_num_list = args.cls_labels_num_list + args.clsml_labels_num_list + args.ner_labels_num_list

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = MultitaskClassifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:

        for _, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            output_list = [{"cls_labels": [],
                            "cls_logits": [],
                            "cls_probs": [],
                            "clsml_labels": [],
                            "clsml_logits": [],
                            "clsml_probs": [],
                            "ner_labels": []
                            } for _ in range(src_batch.size(0))]


            with torch.no_grad():
                cls_logits, ml_logits, ner_pred, last_hidden = model(src_batch, None, seg_batch)


            for dataset_id in range(len(args.cls_labels_num_list)):

                pred = torch.argmax(cls_logits[dataset_id], dim=1)
                pred = pred.cpu().numpy().tolist()
                prob = nn.Softmax(dim=1)(cls_logits[dataset_id])
                logits_np = cls_logits[dataset_id].cpu().numpy().tolist()
                prob = prob.cpu().numpy().tolist()

                for j in range(len(pred)):
                    output_list[j]["cls_labels"].append(str(pred[j]))
                    output_list[j]["cls_logits"].append(" ".join([str(v) for v in logits_np[j]]))
                    output_list[j]["cls_probs"].append(prob[j])

            for ml_id in range(len(args.clsml_labels_num_list)):
                prob = nn.Sigmoid()(ml_logits[ml_id])
                prob = prob.cpu().numpy().tolist()
                logits = ml_logits[ml_id].cpu().numpy().tolist()

                for i, p in enumerate(prob):
                    label = list()
                    for j in range(len(p)):
                        if p[j] > 0.5:
                            label.append(str(j))
                    output_list[i]["clsml_labels"].append(",".join(label))
                    output_list[i]["clsml_logits"].append(" ".join([str(v) for v in logits[i]]))
                    output_list[i]["clsml_probs"].append( " ".join([str(v) for v in p]))

            for ner_id in range(len(args.ner_labels_num_list)):
                pred = ner_pred[ner_id]
                seq_length_batch = []
                for seg in seg_batch.cpu().numpy().tolist():
                    for j in range(len(seg) - 1, -1, -1):
                        if seg[j] != 0:
                            break
                    seq_length_batch.append(j+1)
                pred = pred.cpu().numpy().tolist()
                for i, j in enumerate(range(0, len(pred), args.seq_length)):
                    ner_labels = []
                    for label_id in pred[j: j + seq_length_batch[j // args.seq_length]]:
                        ner_labels.append(str(label_id))
                    output_list[i]["ner_labels"].append(ner_labels)

            for l in output_list:
                f.write(json.dumps(l) + '\n')


if __name__ == "__main__":
    main()
