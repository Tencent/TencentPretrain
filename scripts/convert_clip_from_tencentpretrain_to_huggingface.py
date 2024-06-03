import argparse
import collections
import torch


def convert_clip_transformer(input_model, output_model, layers_num):

    for i in range(12):
        output_model["text_model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]

        output_model["text_model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]

        output_model["text_model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]

        output_model["text_model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.bias"]

        output_model["text_model.encoder.layers." + str(i) + ".layer_norm1.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["text_model.encoder.layers." + str(i) + ".layer_norm1.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.beta"]

        output_model["text_model.encoder.layers." + str(i) + ".mlp.fc1.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".mlp.fc1.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["text_model.encoder.layers." + str(i) + ".mlp.fc2.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["text_model.encoder.layers." + str(i) + ".mlp.fc2.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.bias"]

        output_model["text_model.encoder.layers." + str(i) + ".layer_norm2.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["text_model.encoder.layers." + str(i) + ".layer_norm2.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.beta"]

    for i in range(12):
        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]

        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]

        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]

        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.final_linear.bias"]

        output_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_1.beta"]

        output_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_2.bias"]

        output_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.weight"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.bias"] = \
            input_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_2.beta"]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    output_model["text_model.embeddings.token_embedding.weight"] = input_model["embedding.dual.embedding_0.word.embedding.weight"]
    output_model["text_model.embeddings.position_embedding.weight"] = input_model["embedding.dual.embedding_0.pos.embedding.weight"]
    output_model["vision_model.embeddings.class_embedding"] = input_model["embedding.dual.embedding_1.patch.cls_emb"].squeeze().squeeze()
    output_model["vision_model.embeddings.patch_embedding.weight"] = input_model["embedding.dual.embedding_1.patch.projection.weight"]
    output_model["vision_model.embeddings.position_embedding.weight"] = input_model["embedding.dual.embedding_1.pos.embedding.weight"]

    output_model["vision_model.pre_layrnorm.weight"] = input_model["embedding.dual.stream_1_layer_norm.gamma"]
    output_model["vision_model.pre_layrnorm.bias"] = input_model["embedding.dual.stream_1_layer_norm.beta"]

    convert_clip_transformer(input_model, output_model, args.layers_num)

    output_model["text_model.final_layer_norm.weight"] = input_model["encoder.encoder_0.layer_norm.gamma"]
    output_model["text_model.final_layer_norm.bias"] = input_model["encoder.encoder_0.layer_norm.beta"]
    output_model["vision_model.post_layernorm.weight"] = input_model["encoder.encoder_1.layer_norm.gamma"]
    output_model["vision_model.post_layernorm.bias"] = input_model["encoder.encoder_1.layer_norm.beta"]
    output_model["logit_scale"] = input_model["target.clr.logit_scale"]
    output_model["text_projection.weight"] = input_model["target.clr.encoder_0_projection"].T
    output_model["visual_projection.weight"] = input_model["target.clr.encoder_1_projection"].T

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
