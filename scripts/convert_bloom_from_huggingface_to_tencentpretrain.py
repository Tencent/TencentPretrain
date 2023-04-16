import argparse
import collections
import torch
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/llama-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/llama-7b.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, required=True)
parser.add_argument("--hidden_size", type=int, required=True)
parser.add_argument("--head_num", type=int, required=True)

args = parser.parse_args()

files = os.listdir(args.input_model_path)
model_files = [f for f in files if f[-4:] == ".bin"]
output_model = collections.OrderedDict()
output_model_mapping = collections.OrderedDict()

output_model_mapping['embedding.word.embedding.weight'] = 'word_embeddings.weight'
output_model_mapping['embedding.layer_norm.gamma'] = 'word_embeddings_layernorm.weight'
output_model_mapping['embedding.layer_norm.beta'] = 'word_embeddings_layernorm.bias'

for i in range(args.layers_num):
    # attention ln
    output_model_mapping["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = \
        "h." + str(i) + ".input_layernorm.weight"
    output_model_mapping["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = \
        "h." + str(i) + ".input_layernorm.bias"

    # attention weight
    output_model_mapping["encoder.transformer." + str(i) + ".self_attn.linear_layers.weight"] = \
        'h.' + str(i) + '.self_attention.query_key_value.weight'

    # attention bias
    output_model_mapping["encoder.transformer." + str(i) + ".self_attn.linear_layers.bias"] = \
        'h.' + str(i) + '.self_attention.query_key_value.bias'

    # attention output
    output_model_mapping['encoder.transformer.' + str(i) + '.self_attn.final_linear.weight'] = \
        'h.' + str(i) + '.self_attention.dense.weight'
    output_model_mapping['encoder.transformer.' + str(i) + '.self_attn.final_linear.bias'] = \
        'h.' + str(i) + '.self_attention.dense.bias'

    # FFN ln
    output_model_mapping["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = \
        'h.' + str(i) + '.post_attention_layernorm.weight'
    output_model_mapping["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = \
        'h.' + str(i) + '.post_attention_layernorm.bias'

    # FFN
    output_model_mapping['encoder.transformer.' + str(i) + '.feed_forward.linear_1.weight'] = \
        'h.' + str(i) + '.mlp.dense_h_to_4h.weight'
    output_model_mapping['encoder.transformer.' + str(i) + '.feed_forward.linear_1.bias'] = \
        'h.' + str(i) + '.mlp.dense_h_to_4h.bias'
    output_model_mapping['encoder.transformer.' + str(i) + '.feed_forward.linear_2.weight'] = \
        'h.' + str(i) + '.mlp.dense_4h_to_h.weight'
    output_model_mapping['encoder.transformer.' + str(i) + '.feed_forward.linear_2.bias'] = \
        'h.' + str(i) + '.mlp.dense_4h_to_h.bias'

output_model_mapping['encoder.layer_norm.gamma'] = 'ln_f.weight'
output_model_mapping['encoder.layer_norm.beta'] = 'ln_f.bias'

input_model_mapping = {v: k for k, v in output_model_mapping.items()}
head_per_size = args.hidden_size // args.head_num

for f in model_files:
    checkpoint = torch.load(os.path.join(args.input_model_path, f), map_location='cpu')
    for name, parm in checkpoint.items():
        if 'query_key_value' in name:
            module_name = input_model_mapping[name].split('.')
            if 'weight' in name:
                parm = parm.reshape((args.head_num, head_per_size * 3, args.hidden_size))
                q, k, v = torch.split(parm, head_per_size, dim=-2)
                output_model['.'.join(module_name[:-1]) + '.0.' + module_name[-1]] = q.reshape((args.hidden_size, args.hidden_size))
                output_model['.'.join(module_name[:-1]) + '.1.' + module_name[-1]] = k.reshape((args.hidden_size, args.hidden_size))
                output_model['.'.join(module_name[:-1]) + '.2.' + module_name[-1]] = v.reshape((args.hidden_size, args.hidden_size))
            else:
                parm = parm.reshape((args.head_num, head_per_size * 3))
                q, k, v = torch.split(parm, head_per_size, dim=-1)
                output_model['.'.join(module_name[:-1]) + '.0.' + module_name[-1]] = q.reshape((args.hidden_size))
                output_model['.'.join(module_name[:-1]) + '.1.' + module_name[-1]] = k.reshape((args.hidden_size))
                output_model['.'.join(module_name[:-1]) + '.2.' + module_name[-1]] = v.reshape((args.hidden_size))
        else:
            output_model[input_model_mapping[name]] = parm

torch.save(output_model, args.output_model_path)
