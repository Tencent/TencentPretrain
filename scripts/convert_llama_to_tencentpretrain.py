import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=12)

args = parser.parse_args()

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = collections.OrderedDict()

output_model["embedding.word.embedding.weight"] = input_model["tok_embeddings.weight"]

for i in range(args.layers_num):

    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
        input_model["layers." + str(i) + ".attention.wq.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
        input_model["layers." + str(i) + ".attention.wk.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
        input_model["layers." + str(i) + ".attention.wv.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        input_model["layers." + str(i) + ".attention.wo.weight"]

    output_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        input_model["layers." + str(i) + ".attention_norm.weight"]

    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"] = \
        input_model["layers." + str(i) + ".feed_forward.w1.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        input_model["layers." + str(i) + ".feed_forward.w3.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        input_model["layers." + str(i) + ".feed_forward.w2.weight"]

    output_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
        input_model["layers." + str(i) + ".ffn_norm.weight"]

output_model["encoder.layer_norm.weight"] = input_model["norm.weight"]
output_model["target.lm.output_layer.weight"] = input_model["output.weight"]

torch.save(output_model, args.output_model_path)
