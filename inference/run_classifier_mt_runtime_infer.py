"""
  This script provides an example to wrap TencentPretrain for multi-task classification runtime inference.
"""
import sys
import os

import argparse

import numpy as np
import onnxruntime as ort
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

    parser.add_argument("--input_text", required=True, type=str, help="Data of the input text.")
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


def pre_process(args, text):
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
    seg = [1] * len(src)
    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    while len(src) < args.seq_length:
        src.append(PAD_ID)
        seg.append(0)

    src = torch.LongTensor([src])
    seg = torch.LongTensor([seg])
    return src, seg


def post_process(logits):
    pred = [torch.argmax(logits_i, dim=-1) for logits_i in logits]
    prob = [nn.Softmax(dim=-1)(logits_i) for logits_i in logits]

    logits = [x.cpu().numpy().tolist()[0] for x in logits]
    pred = [x.cpu().numpy().tolist()[0] for x in pred]
    prob = [x.cpu().numpy().tolist()[0] for x in prob]

    pred_output = "|".join([str(v) for v in pred])
    prob_output = "|".join([" ".join(["{0:.4f}".format(w) for w in v]) for v in prob])
    logits_output = "|".join([" ".join(["{0:.4f}".format(w) for w in v]) for v in logits])
    if args.output_logits:
        return pred_output, logits_output
    elif args.output_prob:
        return pred_output, prob_output
    else:
        return pred_output, prob_output


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
                      input_names=input_names, output_names=output_names)
    print(f"ONNX model saved at: {onnx_path}")


def classify_torch(text):
    src, seg = pre_process(args, text)
    model = init_model(args)
    if not os.path.exists("your_onnx_model.onnx"):
        transfer_torch_2_onnx(src, seg, model)
    logits = inference_torch(model, src, seg)
    return post_process(logits)


def classify_onnx(text):
    src, seg = pre_process(args, text)
    ort_session = ort.InferenceSession(args.load_model_path)
    logits = inference_onnx(ort_session, src, seg)
    return post_process(logits)


args = init_args()

if __name__ == "__main__":
    if args.infer_framework == "torch":
        print(classify_torch(args.input_text))
    elif args.infer_framework == "onnx":
        print(classify_onnx(args.input_text))
    else:
        print("Not supported format: %s" % args.infer_framework)
