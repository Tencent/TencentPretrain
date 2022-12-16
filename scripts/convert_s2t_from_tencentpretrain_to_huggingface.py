import argparse
import collections
import torch


def convert_transformer_encoder_from_huggingface_to_tencentpretrain(input_model, output_model, layers_num):
    for i in range(layers_num):
        output_model['model.encoder.layers.' + str(i) + '.self_attn.q_proj.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.0.weight']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.q_proj.bias']  = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.0.bias']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.k_proj.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.1.weight']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.k_proj.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.1.bias']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.v_proj.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.2.weight']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.v_proj.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.linear_layers.2.bias']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.out_proj.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.final_linear.weight']
        output_model['model.encoder.layers.' + str(i) + '.self_attn.out_proj.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.self_attn.final_linear.bias']
            
        output_model['model.encoder.layers.' + str(i) + '.self_attn_layer_norm.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.layer_norm_1.gamma']
        output_model['model.encoder.layers.' + str(i) + '.self_attn_layer_norm.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.layer_norm_1.beta']
            
        output_model['model.encoder.layers.' + str(i) + '.fc1.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.feed_forward.linear_1.weight']
        output_model['model.encoder.layers.' + str(i) + '.fc1.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.feed_forward.linear_1.bias']
        output_model['model.encoder.layers.' + str(i) + '.fc2.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.feed_forward.linear_2.weight']
        output_model['model.encoder.layers.' + str(i) + '.fc2.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.feed_forward.linear_2.bias']
                
        output_model['model.encoder.layers.' + str(i) + '.final_layer_norm.weight'] = \
            input_model['encoder.transformer.' + str(i) + '.layer_norm_2.gamma']
        output_model['model.encoder.layers.' + str(i) + '.final_layer_norm.bias'] = \
            input_model['encoder.transformer.' + str(i) + '.layer_norm_2.beta']
             

def convert_transformer_decoder_from_huggingface_to_tencentpretrain(input_model, output_model, layers_num):
    for i in range(layers_num):
        output_model['model.decoder.layers.' + str(i) + '.self_attn.q_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.0.weight']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.q_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.0.bias']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.k_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.1.weight']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.k_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.1.bias']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.v_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.2.weight']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.v_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.linear_layers.2.bias']
                
        output_model['model.decoder.layers.' + str(i) + '.self_attn.out_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.final_linear.weight']
        output_model['model.decoder.layers.' + str(i) + '.self_attn.out_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.self_attn.final_linear.bias']
        output_model['model.decoder.layers.' + str(i) + '.self_attn_layer_norm.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_1.gamma']
        output_model['model.decoder.layers.' + str(i) + '.self_attn_layer_norm.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_1.beta']
            
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.q_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.0.weight']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.q_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.0.bias']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.k_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.1.weight']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.k_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.1.bias']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.v_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.2.weight']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.v_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.linear_layers.2.bias']
            
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.out_proj.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.final_linear.weight']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn.out_proj.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.context_attn.final_linear.bias']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn_layer_norm.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_2.gamma']
        output_model['model.decoder.layers.' + str(i) + '.encoder_attn_layer_norm.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_2.beta']
                
        output_model['model.decoder.layers.' + str(i) + '.fc1.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.feed_forward.linear_1.weight']
        output_model['model.decoder.layers.' + str(i) + '.fc1.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.feed_forward.linear_1.bias']
        output_model['model.decoder.layers.' + str(i) + '.fc2.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.feed_forward.linear_2.weight']
        output_model['model.decoder.layers.' + str(i) + '.fc2.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.feed_forward.linear_2.bias']
            
        output_model['model.decoder.layers.' + str(i) + '.final_layer_norm.weight'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_3.gamma']
        output_model['model.decoder.layers.' + str(i) + '.final_layer_norm.bias'] = \
            input_model['decoder.transformer_decoder.' + str(i) + '.layer_norm_3.beta']
                       

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.pt",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")
    parser.add_argument("--decoder_layers_num", type=int, default=6, help=".")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path)

    output_model = collections.OrderedDict()

    for i in range(2):
        output_model["model.encoder.conv.conv_layers." + str(i) + ".weight"] = \
            input_model["embedding.speech.conv.conv_layers." + str(i) + ".0.weight"]
        output_model["model.encoder.conv.conv_layers." + str(i) + ".bias"] = \
            input_model["embedding.speech.conv.conv_layers." + str(i) + ".0.bias"]
            
    output_model['model.decoder.embed_tokens.weight'] = input_model['tgt_embedding.word.embedding.weight']

    convert_transformer_encoder_from_huggingface_to_tencentpretrain(input_model, output_model, args.layers_num)
    convert_transformer_decoder_from_huggingface_to_tencentpretrain(input_model, output_model, args.decoder_layers_num)
    output_model['model.encoder.layer_norm.weight'] = input_model['encoder.layer_norm.gamma']
    output_model['model.encoder.layer_norm.bias'] = input_model['encoder.layer_norm.beta']
    output_model['model.decoder.layer_norm.weight'] = input_model['decoder.layer_norm.gamma']
    output_model['model.decoder.layer_norm.bias'] = input_model['decoder.layer_norm.beta']

    output_model["lm_head.weight"] = input_model['target.lm.output_layer.weight']

    torch.save(output_model, args.output_model_path)

if __name__ == "__main__":
    main()
