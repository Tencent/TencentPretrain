import argparse
import collections
import torch


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")


    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    for k in input_model.keys():
        if k == "embedding.word.embedding.weight":
            output_model["embedding.vision_language.text_embedding.word.embedding.weight"] = input_model[k]
        else:
            output_model[k] = input_model[k]
    
    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
