import argparse
import collections
import torch


def convert_vit_transformer_encoder_from_tencentpretrain_to_huggingface(input_model, output_model, layers_num):
    for i in range(layers_num):
        output_model["encoder.layer." + str(i) + ".attention.self.query.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.query.bias"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.bias"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.bias"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.bias"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]
        output_model["encoder.layer." + str(i) + ".layernorm_before.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["encoder.layer." + str(i) + ".layernorm_before.bias"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.bias"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["encoder.layer." + str(i) + ".output.dense.weight"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["encoder.layer." + str(i) + ".output.dense.bias"] = \
            input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
        output_model["encoder.layer." + str(i) + ".layernorm_after.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["encoder.layer." + str(i) + ".layernorm_after.bias"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path)

    output_model = collections.OrderedDict()

    output_model["embeddings.cls_token"] = input_model["embedding.patch.cls_emb"]
    output_model["embeddings.patch_embeddings.projection.weight"] = input_model["embedding.patch.projection.weight"]
    output_model["embeddings.patch_embeddings.projection.bias"] = input_model["embedding.patch.projection.bias"]
    output_model["embeddings.position_embeddings"] = input_model["embedding.pos.embedding.weight"].unsqueeze(0)

    convert_vit_transformer_encoder_from_tencentpretrain_to_huggingface(input_model, output_model, args.layers_num)

    output_model["layernorm.weight"] = input_model["encoder.layer_norm.gamma"]
    output_model["layernorm.bias"] = input_model["encoder.layer_norm.beta"]
    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
