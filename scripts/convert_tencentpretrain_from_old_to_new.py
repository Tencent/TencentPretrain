import sys
import os
import argparse
import collections
import torch

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.utils.config import load_hyperparam


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--config_path", type=str,
                        help=".")

    args = parser.parse_args()
    args = load_hyperparam(args)

    input_model = torch.load(args.input_model_path)

    if "word" in args.embedding:
        input_model["embedding.word.embedding.weight"] = input_model["embedding.word_embedding.weight"]
        input_model.pop("embedding.word_embedding.weight")
    if "pos" in args.embedding:
        input_model["embedding.pos.embedding.weight"] = input_model["embedding.position_embedding.weight"]
        input_model.pop("embedding.position_embedding.weight")
    if "seg" in args.embedding:
        input_model["embedding.seg.embedding.weight"] = input_model["embedding.segment_embedding.weight"]
        input_model.pop("embedding.segment_embedding.weight")
    if "sinusoidalpos" in args.embedding:
        input_model["embedding.sinusoidalpos.pe"] = input_model["embedding.pe"]
        input_model.pop("embedding.pe")

    if hasattr(args, "decoder") and args.decoder is not None:
        for n in list(input_model.keys()): # target.decoder -> decoder
            if n.split('.')[1] == "decoder":
                input_model[".".join(n.split('.')[1:])] = input_model[n]
                input_model.pop(n)
            if n.split('.')[1] == "embedding":
                input_model[".".join(["tgt_embedding"] + n.split('.')[2:])] = input_model[n]
                input_model.pop(n)

        if "word" in args.embedding:
            input_model["tgt_embedding.word.embedding.weight"] = input_model["tgt_embedding.word_embedding.weight"]
            input_model.pop("tgt_embedding.word_embedding.weight")
        if "pos" in args.embedding:
            input_model["tgt_embedding.pos.embedding.weight"] = input_model["tgt_embedding.position_embedding.weight"]
            input_model.pop("tgt_embedding.position_embedding.weight")
        if "seg" in args.embedding:
            input_model["tgt_embedding.seg.embedding.weight"] = input_model["tgt_embedding.segment_embedding.weight"]
            input_model.pop("tgt_embedding.segment_embedding.weight")
        if "sinusoidalpos" in args.embedding:
            input_model["tgt_embedding.sinusoidalpos.pe"] = input_model["tgt_embedding.pe"]
            input_model.pop("tgt_embedding.pe")

    if "mlm" in args.target:
        try:
            input_model["target.mlm.linear_1.weight"] = input_model["target.mlm_linear_1.weight"]
            input_model.pop("target.mlm_linear_1.weight")
            input_model["target.mlm.linear_1.bias"] = input_model["target.mlm_linear_1.bias"]
            input_model.pop("target.mlm_linear_1.bias")
            input_model["target.mlm.layer_norm.gamma"] = input_model["target.layer_norm.gamma"]
            input_model.pop("target.layer_norm.gamma")
            input_model["target.mlm.layer_norm.beta"] = input_model["target.layer_norm.beta"]
            input_model.pop("target.layer_norm.beta")
            input_model["target.mlm.linear_2.weight"] = input_model["target.mlm_linear_2.weight"]
            input_model.pop("target.mlm_linear_2.weight")
            input_model["target.mlm.linear_2.bias"] = input_model["target.mlm_linear_2.bias"]
            input_model.pop("target.mlm_linear_2.bias")
        except:
            pass

    if "sp" in args.target:
        try:
            input_model["target.sp.linear_1.weight"] = input_model["target.sp_linear_1.weight"]
            input_model.pop("target.sp_linear_1.weight")
            input_model["target.sp.linear_1.bias"] = input_model["target.sp_linear_1.bias"]
            input_model.pop("target.sp_linear_1.bias")
            input_model["target.sp.linear_2.weight"] = input_model["target.sp_linear_2.weight"]
            input_model.pop("target.sp_linear_2.weight")
            input_model["target.sp.linear_2.bias"] = input_model["target.sp_linear_2.bias"]
            input_model.pop("target.sp_linear_2.bias")
        except:
            pass
        try:
            input_model["target.sp.linear_1.weight"] = input_model["target.nsp_linear_1.weight"]
            input_model.pop("target.nsp_linear_1.weight")
            input_model["target.sp.linear_1.bias"] = input_model["target.nsp_linear_1.bias"]
            input_model.pop("target.nsp_linear_1.bias")
            input_model["target.sp.linear_2.weight"] = input_model["target.nsp_linear_2.weight"]
            input_model.pop("target.nsp_linear_2.weight")
            input_model["target.sp.linear_2.bias"] = input_model["target.nsp_linear_2.bias"]
            input_model.pop("target.nsp_linear_2.bias")
        except:
            pass
        try:
            input_model["target.sp.linear_1.weight"] = input_model["target.sop_linear_1.weight"]
            input_model.pop("target.sop_linear_1.weight")
            input_model["target.sp.linear_1.bias"] = input_model["target.sop_linear_1.bias"]
            input_model.pop("target.sop_linear_1.bias")
            input_model["target.sp.linear_2.weight"] = input_model["target.sop_linear_2.weight"]
            input_model.pop("target.sop_linear_2.weight")
            input_model["target.sp.linear_2.bias"] = input_model["target.sop_linear_2.bias"]
            input_model.pop("target.sop_linear_2.bias")
        except:
            pass
    if "lm" in args.target:
        try:
            input_model["target.lm.output_layer.weight"] = input_model["target.output_layer.weight"]
            input_model.pop("target.output_layer.weight")
            if args.has_lmtarget_bias:
                input_model["target.lm.output_layer.bias"] = input_model["target.output_layer.bias"]
                input_model.pop("target.output_layer.bias")
        except:
            pass

    torch.save(input_model, args.output_model_path)

if __name__ == "__main__":
    main()
