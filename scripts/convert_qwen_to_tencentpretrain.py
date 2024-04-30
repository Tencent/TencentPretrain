import argparse
import os
import collections
from safetensors.torch import load_file
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=12)

args = parser.parse_args()

input_model = {}
for file_name in os.listdir(args.input_model_path):
    if os.path.splitext(file_name)[-1][1:] == "safetensors":
        dict = load_file(filename=os.path.join(args.input_model_path, file_name))
        input_model.update(dict)

output_model = collections.OrderedDict()
emb_size = input_model["transformer.h." + str(0) + ".attn.c_attn.weight"].shape[1]

output_model["embedding.word.embedding.weight"] = input_model["transformer.wte.weight"]


for i in range(args.layers_num):
    for j in range(3):
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".weight"] = \
            input_model["transformer.h." + str(i) + ".attn.c_attn.weight"][j*emb_size:(j+1)*emb_size, :]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".bias"] = \
            input_model["transformer.h." + str(i) + ".attn.c_attn.bias"][j*emb_size:(j+1)*emb_size]

    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        input_model["transformer.h." + str(i) + ".attn.c_proj.weight"]
    
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        input_model["transformer.h." + str(i) + ".ln_1.weight"]
    
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"] = \
        input_model["transformer.h." + str(i) + ".mlp.w1.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        input_model["transformer.h." + str(i) + ".mlp.w2.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        input_model["transformer.h." + str(i) + ".mlp.c_proj.weight"]
    
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
        input_model["transformer.h." + str(i) + ".ln_2.weight"]

output_model["encoder.layer_norm.gamma"] = input_model["transformer.ln_f.weight"]
output_model["target.lm.output_layer.weight"] = input_model["lm_head.weight"]

torch.save(output_model, args.output_model_path)
