import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str,
                    help="Input file path")
parser.add_argument("--output_model_path", type=str,
                    help="Output folder path")
parser.add_argument("--block_size", type=int, default=10,
                    help="Disi size (GB) of each block.")


args = parser.parse_args()

os.system('mkdir ' + args.output_model_path)

input_model = torch.load(args.input_model_path)

byte_size = args.block_size * 500000000

param_count, file_count, filename_count = 0, 0, 0
index_dict = {"weight_map": {}}

state_dict = collections.OrderedDict()
filename = f"tencentpretrain_model-0.bin"
for k, v in input_model.items():
    state_dict[k] = v
    index_dict["weight_map"][k] = filename
    param_count += v.numel()
    file_count += v.numel()
    if file_count > byte_size:
        torch.save(state_dict, os.path.join(args.output_model_path, filename))
        state_dict = collections.OrderedDict()
        filename_count += 1
        filename = f"tencentpretrain_model-"+str(filename_count)+".bin"
        file_count = 0

if len(state_dict) > 0:
    torch.save(state_dict, os.path.join(args.output_model_path, filename))

index_dict["metadata"] = {"total_size": param_count * 2}
with open(os.path.join(args.output_model_path, "tencentpretrain_model.bin.index.json"), "w") as f:
    json.dump(index_dict, f)

