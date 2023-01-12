"""
  This script provides an example to wrap TencentPretrain for classification inference.
"""
import sys
import os
import torch
import argparse
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

    def forward(self, src, tgt, seg, labels_num_list):
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
        output_logits = []
        for dataset_id in range(len(labels_num_list)):
            output_1 = torch.tanh(self.output_layers_1[dataset_id](last_hidden))
            output_logits.append(self.output_layers_2[dataset_id](output_1))

        return output_logits, last_hidden


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
    parser.add_argument("--labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

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

        for dataset_id in range(len(args.labels_num_list)):
            if dataset_id == 0:
                f.write("label-" + str(dataset_id))
            else:
                f.write("\t" + "label-" + str(dataset_id))

            if args.output_logits:
                f.write("\t" + "logits-" + str(dataset_id))
            if args.output_prob:
                f.write("\t" + "prob-" + str(dataset_id))

        f.write("\n")

        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            labels_list, logits_list, prob_list = [[] for _ in range(len(args.labels_num_list))], \
                                                  [[] for _ in range(len(args.labels_num_list))], \
                                                  [[] for _ in range(len(args.labels_num_list))]


            with torch.no_grad():
                output_logits, last_hidden = model(src_batch, None, seg_batch, args.labels_num_list)


            for dataset_id in range(len(args.labels_num_list)):

                pred = torch.argmax(output_logits[dataset_id], dim=1)
                pred = pred.cpu().numpy().tolist()
                prob = nn.Softmax(dim=1)(output_logits[dataset_id])
                logits_np = output_logits[dataset_id].cpu().numpy().tolist()
                prob = prob.cpu().numpy().tolist()

                for j in range(len(pred)):
                    labels_list[dataset_id].append(str(pred[j]))
                    logits_list[dataset_id].append(logits_np[j])
                    prob_list[dataset_id].append(prob[j])

            for x in range(len(pred)):
                for y in range(len(args.labels_num_list)):
                    if y == 0:
                        f.write(labels_list[y][x])
                    else:
                        f.write("\t" + labels_list[y][x])
                    if args.output_logits:
                        f.write("\t" + " ".join([str(v) for v in logits_list[y][x]]))
                    if args.output_prob:
                        f.write("\t" + " ".join([str(v) for v in prob_list[y][x]]))

                f.write("\n")


if __name__ == "__main__":
    main()
