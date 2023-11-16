import argparse
import collections
import torch
import os 


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model",
                        help=".")
parser.add_argument("--layers_num", type=int, default=32)
parser.add_argument("--tensor_model_parallel_size", type=int, default=4)
parser.add_argument("--hidden_size", type=int, default=4096)
parser.add_argument("--feedforward_size", type=int, default=11008)

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

if not os.path.exists(args.output_model_path):
    os.mkdir(args.output_model_path)

seg_feed_size=args.feedforward_size // args.tensor_model_parallel_size
seg_hidden_size = args.hidden_size // args.tensor_model_parallel_size
seg_word_size=input_model["embedding.word.embedding.weight"].size()[0] // args.tensor_model_parallel_size

for n in range(args.tensor_model_parallel_size):
    model_piece=collections.OrderedDict()
    seg_dim=input_model["embedding.word.embedding.weight"].size()[0]//args.tensor_model_parallel_size
    model_piece["embedding.word.embedding.weight"] = input_model["embedding.word.embedding.weight"][n* seg_dim :(n+1) * seg_dim,:]
    
    for i in range(args.layers_num):
        for j in range(3):
            model_piece["encoder.transformer." + str(i) + ".self_attn.linear_layers."+str(j)+".weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers."+str(j)+".weight"][n * seg_hidden_size:(n+1) * seg_hidden_size,:]
        
        model_piece["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"][:,n * seg_hidden_size:(n+1) * seg_hidden_size]
        
        model_piece["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]
       
        model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"][n * seg_feed_size:(n+1)  * seg_feed_size,:]
        
        model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]= \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"][n * seg_feed_size:(n+1)  * seg_feed_size,:]
        
        model_piece["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"][:,n * seg_feed_size:(n+1)  * seg_feed_size]
        
        model_piece["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"]
    
    model_piece["encoder.layer_norm.weight"] = input_model["encoder.layer_norm.weight"]
    
    model_piece["target.lm.output_layer.weight"]= input_model["target.lm.output_layer.weight"][n * seg_word_size:(n+1) * seg_word_size,:]
    
    name=str(n) if len(str(n))==2 else '0'+str(n)
    torch.save(model_piece, os.path.join(args.output_model_path,"mp_rank_"+str(name)+"_model_states.pt"))
 
