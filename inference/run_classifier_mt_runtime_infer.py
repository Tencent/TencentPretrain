"""
  This script provides an example to wrap TencentPretrain for multi-task classification runtime inference.
"""
import sys
import os

import argparse

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch.nn as nn

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.misc import pooling
from tencentpretrain.model_loader import *
from tencentpretrain.opts import tokenizer_opts, log_opts, model_opts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskClassifier(nn.Module):
    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layers_1 = nn.ModuleList(
            [nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList(
            [nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])

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
        memory_bank = self.encoder(emb, seg)
        # Target.
        memory_bank = pooling(memory_bank, seg, self.pooling_type)
        logits = []
        for i in range(len(self.output_layers_1)):
            output_i = torch.tanh(self.output_layers_1[i](memory_bank))
            logits_i = self.output_layers_2[i](output_i)
            logits.append(logits_i)

        return None, logits


def infer_opts(parser):
    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str, help="Path of the input model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file.")
    parser.add_argument("--test_path", type=str, required=False, help="Path of the testset.")
    parser.add_argument("--prediction_path", type=str, required=False, help="Path of the prediction file.")

    # Model options.
    model_opts(parser)

    # Inference options.
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")


def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")

    parser.add_argument("--text_a", required=False, type=str, help="Data of the input text_a.")
    parser.add_argument("--text_b", required=False, type=str, help="Data of the input text_b.")
    parser.add_argument("--infer_framework", default="torch", type=str, help="Framework of inference.")

    log_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False

    return args


def init_model(args):
    model = MultitaskClassifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{0} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.eval()
    return model


def pre_process(args, text_a, text_b: str = None, is_batch: bool = False):
    if text_b is None:  # Sentence classification.
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
        seg = [1] * len(src)
    else:  # Sentence pair classification.
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

    if is_batch:
        return src, seg

    src = torch.LongTensor([src])
    seg = torch.LongTensor([seg])
    return src, seg


def post_process(logits):
    pred = [torch.argmax(logits_i, dim=-1) for logits_i in logits]
    prob = [nn.Softmax(dim=-1)(logits_i) for logits_i in logits]

    logits = [x.cpu().numpy().tolist() for x in logits]
    pred = [x.cpu().numpy().tolist() for x in pred]
    prob = [x.cpu().numpy().tolist() for x in prob]

    pred_outputs, prob_outputs, logits_outputs = [], [], []

    for j in range(len(pred[0])):
        pred_outputs.append("|".join([str(v[j]) for v in pred]))
        logits_outputs.append("|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in logits]))
        prob_outputs.append("|".join([" ".join(["{0:.4f}".format(w) for w in v[j]]) for v in prob]))

    if args.output_logits:
        return pred_outputs, logits_outputs
    elif args.output_prob:
        return pred_outputs, prob_outputs
    else:
        return pred_outputs, prob_outputs


def inference_torch(model, src, seg):
    with torch.no_grad():
        _, logits = model(src.to(device), None, seg.to(device))
    return logits


def inference_onnx(ort_session, src, seg):
    outputs = ort_session.run(None, {"input": src.numpy(), "seg": seg.numpy()})
    logits = [torch.from_numpy(np.array(output)) for output in outputs]
    return logits


def transfer_torch_2_onnx(src, seg, model, onnx_path: str = "your_onnx_model.onnx"):
    input_names = ["input", "seg"]
    output_names = ["output"]
    dummy_input = (src.to(device), None, seg.to(device))
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes={"input": {0: "batch_size"}, "seg": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"ONNX model saved at: {onnx_path}")


def save_torch_as_onnx(onnx_model_path: str = "your_onnx_model.onnx"):
    """
    Transfer Torch model to ONNX format
    """
    src, seg = pre_process(args, "Hello, World")
    model = init_model(args)
    transfer_torch_2_onnx(src, seg, model, onnx_model_path)


def classify_torch(text_a, text_b: str = None):
    src, seg = pre_process(args, text_a, text_b)
    model = init_model(args)
    logits = inference_torch(model, src, seg)
    return post_process(logits)


def classify_onnx(text_a, text_b: str = None):
    src, seg = pre_process(args, text_a, text_b)
    ort_session = ort.InferenceSession(args.load_model_path)
    logits = inference_onnx(ort_session, src, seg)
    return post_process(logits)


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        yield src_batch, seg_batch


def batch_classify():
    global model, ort_session

    df = pd.read_csv(args.test_path)
    dataset = []
    columns = df.columns.values.tolist()
    for idx, item in enumerate(df.itertuples()):
        text_a = getattr(item, "text_a")
        text_b = getattr(item, "text_b") if "text_b" in columns else None
        src, seg = pre_process(args, text_a, text_b, True)
        dataset.append((src, seg))

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]
    print("The number of prediction instances: {0}".format(instances_num))

    # Load model
    if args.infer_framework == "torch":
        model = init_model(args)
    elif args.infer_framework == "onnx":
        ort_session = ort.InferenceSession(args.load_model_path)
    else:
        print("Not supported format: %s" % args.infer_framework)
        return

    results = []
    for _, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
        if args.infer_framework == "torch":
            logits = inference_torch(model, src_batch, seg_batch)
        else:
            logits = inference_onnx(ort_session, src_batch, seg_batch)
        result = post_process(logits)
        results += [(pred, score) for pred, score in zip(result[0], result[1])]

    # Save as local file
    columns = ["label", "logits" if args.output_logits else "prob"]
    pd.DataFrame(results, columns=columns).to_csv(args.prediction_path, sep="\t", encoding="utf-8", index=False)


args = init_args()


def main():
    if args.test_path is not None:
        batch_classify()
        return
    if args.infer_framework == "torch":
        print(classify_torch(args.text_a, args.text_b))
    elif args.infer_framework == "onnx":
        print(classify_onnx(args.text_a, args.text_b))
    else:
        print("Not supported format: %s" % args.infer_framework)


if __name__ == "__main__":
    main()
