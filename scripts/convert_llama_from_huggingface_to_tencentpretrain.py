import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/llama-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/llama-7b.bin",
                    help=".")
parser.add_argument("--type", choices=["7B", "13B", "33B", "65B"], default="7B")

args = parser.parse_args()

model_config = {"7B" : [32, 4096, 32],
              "13B": [40, 5120, 40],
              "33B": [60, 6656, 52],
              "65B": [80, 8192, 64]
              }

layers_num, dim, n_heads = model_config[args.type]

files = os.listdir(args.input_model_path)
model_files = [f for f in files if f[-4:] == ".bin"]
input_models = {f: torch.load(os.path.join(args.input_model_path, f), map_location="cpu") for f in model_files}

with open(os.path.join(args.input_model_path, "pytorch_model.bin.index.json")) as f:
    model_index = json.load(f)
    weight_map = model_index["weight_map"]


output_model = collections.OrderedDict()

def get_weight_from_name(layer_name):
    return input_models[weight_map[layer_name]][layer_name]

def unpermute(w):
    return w.reshape(n_heads, 2, dim // n_heads // 2, dim).transpose(2, 1).reshape(dim, dim)

output_model["embedding.word.embedding.weight"] = get_weight_from_name("model.embed_tokens.weight")

for i in range(layers_num):

    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
        unpermute(get_weight_from_name("model.layers." + str(i) + ".self_attn.q_proj.weight"))
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
        unpermute(get_weight_from_name("model.layers." + str(i) + ".self_attn.k_proj.weight"))

    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".self_attn.v_proj.weight")
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".self_attn.o_proj.weight")

    output_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".input_layernorm.weight")

    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".mlp.gate_proj.weight")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".mlp.up_proj.weight")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".mlp.down_proj.weight")

    output_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
        get_weight_from_name("model.layers." + str(i) + ".post_attention_layernorm.weight")

output_model["encoder.layer_norm.weight"] = get_weight_from_name("model.norm.weight")
output_model["target.lm.output_layer.weight"] = get_weight_from_name("lm_head.weight")

torch.save(output_model, args.output_model_path)
