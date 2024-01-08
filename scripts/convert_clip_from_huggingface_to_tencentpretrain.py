import argparse
import collections
import torch


def convert_clip_transformer(input_model, output_model, layers_num):

    for i in range(layers_num):
        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.gamma"] = \
            input_model["text_model.encoder.layers." + str(i) + ".layer_norm1.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.beta"] = \
            input_model["text_model.encoder.layers." + str(i) + ".layer_norm1.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".mlp.fc1.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".mlp.fc1.bias"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
            input_model["text_model.encoder.layers." + str(i) + ".mlp.fc2.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.bias"] = \
            input_model["text_model.encoder.layers." + str(i) + ".mlp.fc2.bias"]

        output_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.gamma"] = \
            input_model["text_model.encoder.layers." + str(i) + ".layer_norm2.weight"]
        output_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.beta"] = \
            input_model["text_model.encoder.layers." + str(i) + ".layer_norm2.bias"]


        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".self_attn.final_linear.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_1.gamma"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_1.beta"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_1.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.bias"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".feed_forward.linear_2.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.bias"]

        output_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_2.gamma"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.weight"]
        output_model["encoder.encoder_1.transformer." + str(i) + ".layer_norm_2.beta"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.bias"]

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

    output_model["embedding.dual.embedding_0.word.embedding.weight"] = input_model["text_model.embeddings.token_embedding.weight"]
    output_model["embedding.dual.embedding_0.pos.embedding.weight"] = input_model["text_model.embeddings.position_embedding.weight"]
    output_model["embedding.dual.embedding_1.patch.cls_emb"] = input_model["vision_model.embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)
    output_model["embedding.dual.embedding_1.patch.projection.weight"] = input_model["vision_model.embeddings.patch_embedding.weight"]
    output_model["embedding.dual.embedding_1.pos.embedding.weight"] = input_model["vision_model.embeddings.position_embedding.weight"]

    output_model["embedding.dual.stream_1_layer_norm.gamma"] = input_model["vision_model.pre_layrnorm.weight"]
    output_model["embedding.dual.stream_1_layer_norm.beta"] = input_model["vision_model.pre_layrnorm.bias"]

    convert_clip_transformer(input_model, output_model, args.layers_num)

    output_model["encoder.encoder_0.layer_norm.gamma"] = input_model["text_model.final_layer_norm.weight"]
    output_model["encoder.encoder_0.layer_norm.beta"] = input_model["text_model.final_layer_norm.bias"]
    output_model["encoder.encoder_1.layer_norm.gamma"] = input_model["vision_model.post_layernorm.weight"]
    output_model["encoder.encoder_1.layer_norm.beta"] = input_model["vision_model.post_layernorm.bias"]
    output_model["target.clr.logit_scale"] = input_model["logit_scale"]
    output_model["target.clr.encoder_0_projection"] = input_model["text_projection.weight"].T
    output_model["target.clr.encoder_1_projection"] = input_model["visual_projection.weight"].T

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
