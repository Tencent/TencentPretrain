import argparse
import collections
import torch
import os
import json
from safetensors import safe_open


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/llama-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/llama-7b.bin",
                    help=".")
parser.add_argument("--type", choices=["3B", "7B", "13B", "33B", "65B", "70B"], default="7B")


args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()



model_config = {"3B" : [26, 3200, 32],
                "7B" : [32, 4096, 32],
                "13B": [40, 5120, 40],
                "33B": [60, 6656, 52],
                "65B": [80, 8192, 64],
                "70B": [80, 8192, 64]
                }

layers_num, dim, n_heads = model_config[args.type]

if args.gqa == "70B":
    dim2 = dim // 8
    kv_heads = 8
else:
    dim2 = dim
    kv_heads = n_heads

def permute_q(w):
    return w.reshape(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

def permute_k(w):
    return w.reshape(kv_heads, dim // kv_heads // kv_heads // 2, 2, dim).transpose(1, 2).reshape(dim2, dim)

dims_per_head = dim // n_heads


inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

output_model["model.embed_tokens.weight"] = input_model["embedding.word.embedding.weight"]

for i in range(layers_num):

    output_model["model.layers." + str(i) + ".self_attn.q_proj.weight"] = \
        permute_q(input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"])

    output_model["model.layers." + str(i) + ".self_attn.k_proj.weight"] = \
        permute_k(input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"])

    output_model["model.layers." + str(i) + ".self_attn.v_proj.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
    output_model["model.layers." + str(i) + ".self_attn.o_proj.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]

    output_model["model.layers." + str(i) + ".input_layernorm.weight"] = \
        input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]

    output_model["model.layers." + str(i) + ".mlp.gate_proj.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]
    output_model["model.layers." + str(i) + ".mlp.up_proj.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
    output_model["model.layers." + str(i) + ".mlp.down_proj.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]

    output_model["model.layers." + str(i) + ".post_attention_layernorm.weight"] = \
        input_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"]

    output_model["model.layers." + str(i) + ".self_attn.rotary_emb.inv_freq"] = inv_freq

output_model["model.norm.weight"] = input_model["encoder.layer_norm.weight"]
output_model["lm_head.weight"] = input_model["target.lm.output_layer.weight"]

os.system('mkdir ' + args.output_model_path)


byte_size = 10 * 500000000

param_count, file_count, filename_count = 0, 0, 0
index_dict = {"weight_map": {}}

state_dict = collections.OrderedDict()
filename = f"pytorch_model-0.bin"
for k, v in output_model.items():
    state_dict[k] = v.bfloat16()
    index_dict["weight_map"][k] = filename
    param_count += v.numel()
    file_count += v.numel()
    if file_count > byte_size:
        torch.save(state_dict, os.path.join(args.output_model_path, filename))
        state_dict = collections.OrderedDict()
        filename_count += 1
        filename = f"pytorch_model-"+str(filename_count)+".bin"
        file_count = 0

if len(state_dict) > 0:
    torch.save(state_dict, os.path.join(args.output_model_path, filename))

index_dict["metadata"] = {"total_size": param_count * 2}
with open(os.path.join(args.output_model_path, "pytorch_model.bin.index.json"), "w") as f:
    json.dump(index_dict, f)

