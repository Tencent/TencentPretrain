#!/usr/bin/env python3

"""
Helper script to generate speech2text dataset
"""

import argparse
import os
import glob
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--add_column", action='store_true')
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--ext", default="flac")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rand = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.realpath(args.output_path)), exist_ok=True)

    dir_path = os.path.realpath(args.input_path)
    search_path = os.path.join(dir_path, "**/*." + args.ext)

    transcriptions = {}

    with open(args.output_path, "w") as out_file:
        if args.add_column:
            print("text" + "\t" + "wav_path", file=out_file)
        else:
            print("wav_path", file=out_file)
        for fname in glob.iglob(search_path, recursive=True):
            if rand.random() > args.sample_ratio:
                continue
            file_path = os.path.realpath(fname)
            if args.add_column:
                dir = os.path.dirname(file_path)
                if dir not in transcriptions:
                    parts = dir.split(os.path.sep)
                    trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
                    path = os.path.join(args.input_path, dir, trans_path)
                    assert os.path.exists(path)
                    texts = {}
                    with open(path, "r") as trans_f:
                        for tline in trans_f:
                            items = tline.strip().split()
                            texts[items[0]] = " ".join(items[1:]).lower()
                    transcriptions[dir] = texts
                part = os.path.basename(file_path).split(".")[0]
                assert part in transcriptions[dir]
                print(
                        transcriptions[dir][part] + "\t" + file_path,
                        file=out_file
                )
            else:
                print(file_path, file=out_file)


if __name__ == "__main__":
    main()
