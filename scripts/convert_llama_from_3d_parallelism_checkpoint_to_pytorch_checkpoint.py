import argparse
import os
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=32)
parser.add_argument("--tensor_model_parallel_size", type=int, default=4)

args = parser.parse_args()

if not os.path.exists(args.output_model_path):
    os.mkdir(args.output_model_path)

model_piece_list = []
for n in range(args.tensor_model_parallel_size):
    model_piece = collections.OrderedDict()
    model_index = str(n) if len(str(n))==2 else '0'+str(n)
    for i in range(args.layers_num+2):
        layer_index = str(i) if len(str(i))==2 else '0'+str(i)
        weight_name = f"layer_{layer_index}-model_{model_index}-model_states.pt"
        tmp_weight = torch.load(os.path.join(args.input_model_path, weight_name), map_location="cpu")
        if i == 0:
            model_piece["embedding.word.embedding.weight"] = tmp_weight["embeddings.word.embedding.weight"]
        elif i == args.layers_num+1:
            model_piece["target.lm.output_layer.weight"] = tmp_weight["target_layer.lm.output_layer.weight"]
        else:
            for j in range(3):
                model_piece["encoder.transformer." + str(i-1) + ".self_attn.linear_layers."+ str(j) +".weight"] = tmp_weight["layer.self_attn.linear_layers."+ str(j) +".weight"]
            model_piece["encoder.transformer." + str(i-1) + ".self_attn.final_linear.weight"] = tmp_weight["layer.self_attn.final_linear.weight"]
            model_piece["encoder.transformer." + str(i-1) + ".feed_forward.linear_1.weight"] = tmp_weight["layer.feed_forward.linear_1.weight"]
            model_piece["encoder.transformer." + str(i-1) + ".feed_forward.linear_2.weight"] = tmp_weight["layer.feed_forward.linear_2.weight"]
            model_piece["encoder.transformer." + str(i-1) + ".feed_forward.linear_gate.weight"] = tmp_weight["layer.feed_forward.linear_gate.weight"]
            model_piece["encoder.transformer." + str(i-1) + ".layer_norm_1.weight"] = tmp_weight["layer.layer_norm_1.weight"]
            model_piece["encoder.transformer." + str(i-1) + ".layer_norm_2.weight"] = tmp_weight["layer.layer_norm_2.weight"]
            if i == args.layers_num:
                model_piece["encoder.layer_norm.weight"] = tmp_weight["layer_norm.weight"]
    model_piece_list.append(model_piece)

output_model = model_piece_list[0]

for n in range(1, args.tensor_model_parallel_size):
    model_piece = model_piece_list[n]
    output_model["embedding.word.embedding.weight"] = torch.cat((output_model["embedding.word.embedding.weight"], model_piece["embedding.word.embedding.weight"]),dim=-2)
    
    for i in range(args.layers_num):
        for j in range(3):
            tensor_a=output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers."+ str(j) +".weight"]
            tensor_b=model_piece["encoder.transformer." + str(i) + ".self_attn.linear_layers."+ str(j) +".weight"]
            output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers."+ str(j) +".weight"]=torch.cat((tensor_a,tensor_b),dim=-2)
        
        tensor_a=output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
        tensor_b=model_piece["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
      
        output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]=torch.cat((tensor_a,tensor_b),dim=-1)
        
        tensor_a=output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        tensor_b=model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]=torch.cat((tensor_a,tensor_b),dim=-2)
        
        tensor_a=output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]
        tensor_b=model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]=torch.cat((tensor_a,tensor_b),dim=-2)
                
        tensor_a=output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        tensor_b=model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]=torch.cat((tensor_a,tensor_b),dim=-1)
    
    tensor_a=output_model["target.lm.output_layer.weight"]
    tensor_b=model_piece["target.lm.output_layer.weight"]
    output_model["target.lm.output_layer.weight"]=torch.cat((tensor_a,tensor_b),dim=-2)

torch.save(output_model, args.output_model_path)
