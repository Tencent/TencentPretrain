import argparse
import collections
import torch

def convert_llama_transformer_encoder_from_tencentpretrain_to_huggingface(input_model, output_model, layers_num):
    for i in range(args.layers_num):

        output_model["layers." + str(i) + ".attention.wq.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["layers." + str(i) + ".attention.wk.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["layers." + str(i) + ".attention.wv.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["layers." + str(i) + ".attention.wo.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]

        output_model["layers." + str(i) + ".attention_norm.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]

        output_model["layers." + str(i) + ".feed_forward.w1.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]
        output_model["layers." + str(i) + ".feed_forward.w3.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["layers." + str(i) + ".feed_forward.w2.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]

        output_model["layers." + str(i) + ".ffn_norm.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                            help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                            help=".")
    parser.add_argument("--layers_num", type=int, default=12)

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    output_model["tok_embeddings.weight"] = input_model["embedding.word.embedding.weight"]

    convert_llama_transformer_encoder_from_tencentpretrain_to_huggingface(input_model, output_model, args.layers_num)
    
    output_model["norm.weight"] = input_model["encoder.layer_norm.weight"]
    output_model["output.weight"] = input_model["target.lm.output_layer.weight"]

    torch.save(output_model, args.output_model_path)

if __name__ == "__main__":
    main()

