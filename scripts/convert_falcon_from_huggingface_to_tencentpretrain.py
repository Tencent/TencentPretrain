import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/falcon-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/falcon-7b.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=32,
                    help=".")

args = parser.parse_args()

files = os.listdir(args.input_model_path)
model_files = [f for f in files if f[-4:] == ".bin"]
input_models = {f: torch.load(os.path.join(args.input_model_path, f), map_location="cpu") for f in model_files}

with open(os.path.join(args.input_model_path, "pytorch_model.bin.index.json")) as f:
    model_index = json.load(f)
    weight_map = model_index["weight_map"]


output_model = collections.OrderedDict()

def get_weight_from_name(layer_name):
    return input_models[weight_map[layer_name]][layer_name]


output_model["embedding.word.embedding.weight"] = get_weight_from_name("transformer.word_embeddings.weight")

for i in range(args.layers_num):

    output_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        get_weight_from_name("transformer.h." + str(i) + ".input_layernorm.weight")
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.bias"] = \
        get_weight_from_name("transformer.h." + str(i) + ".input_layernorm.bias")

    output_model["encoder.transformer." + str(i) + ".self_attn.query_key_value.weight"] = \
        get_weight_from_name("transformer.h." + str(i) + ".self_attention.query_key_value.weight")
    output_model["encoder.transformer." + str(i) + ".self_attn.dense.weight"] = \
        get_weight_from_name("transformer.h." + str(i) + ".self_attention.dense.weight")

    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        get_weight_from_name("transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        get_weight_from_name("transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight")

output_model["encoder.layer_norm.weight"] = get_weight_from_name("transformer.ln_f.weight")
output_model["encoder.layer_norm.bias"] = get_weight_from_name("transformer.ln_f.bias")
output_model["target.lm.output_layer.weight"] = get_weight_from_name("lm_head.weight")

torch.save(output_model, args.output_model_path)
