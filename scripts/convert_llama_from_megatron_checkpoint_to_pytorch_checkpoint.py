import argparse
import os 
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model",
                        help=".")
parser.add_argument("--layers_num", type=int, default=32)
parser.add_argument("--tensor_model_parallel_size", type=int, default=4)


args = parser.parse_args()

if not os.path.exists(args.output_model_path):
    os.mkdir(args.output_model_path)

output_model=torch.load(os.path.join(args.input_model_path,'mp_rank_00_model_states.pt'),map_location='cpu')["module"]

for n in range(1,args.tensor_model_parallel_size):
    index=str(n) if len(str(n))==2 else '0'+str(n)
    model_name=f"mp_rank_{index}_model_states.pt"
    model_piece = torch.load(os.path.join(args.input_model_path,model_name),map_location="cpu")["module"]
    output_model["embedding.word.embedding.weight"] = torch.cat((output_model["embedding.word.embedding.weight"],model_piece["embedding.word.embedding.weight"]),dim=-2)
    
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

torch.save(output_model,os.path.join(args.output_model_path,'merge_model.bin'))
 
