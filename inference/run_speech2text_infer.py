"""
This script provides an example to wrap TencentPretrain for speech-to-text inference.
"""
import sys
import os
import tqdm
import argparse
import math
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.vocab import Vocab
from tencentpretrain.model_loader import load_model
from tencentpretrain.model_builder import build_model
from tencentpretrain.opts import infer_opts, tokenizer_opts
from finetune.run_text2text import Text2text
from inference.run_classifier_infer import batch_loader


def read_dataset(args, path):
    dataset = []
    padding_vector = torch.FloatTensor(args.audio_feature_size * [0.0] if args.audio_feature_size > 1 else 0.0).unsqueeze(0)
    def utterance_cmvn( x, normalize_means=True, normalize_vars=True):
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        if normalize_means:
            x = torch.sub(x, mean)
        if normalize_vars:
            var = square_sums / x.size(0) - mean ** 2
            std = torch.sqrt(torch.maximum(var, torch.full(var.size() , 1e-10)))
            x = torch.div(x, std)
        return x

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            text, wav_path = line.rstrip("\r\n").split("\t")
            tgt_text = [args.tokenizer.convert_tokens_to_ids(['<s>'])] + \
                args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text)) + \
                [args.tokenizer.convert_tokens_to_ids(['</s>'])]

            if len(tgt_text) > args.seq_length:
                tgt_text = tgt_text[: args.seq_length]
                tgt_seg = tgt_seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids(['<pad>'])
            pad_num = args.seq_length - len(tgt_text)
            tgt_text = tgt_text + [PAD_ID] * pad_num

            waveform, sample_rate = torchaudio.load(wav_path)
            waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
            feature = ta_kaldi.fbank(waveform, num_mel_bins=args.audio_feature_size, sample_frequency=sample_rate)
            if "ceptral_normalize" in args.audio_preprocess:
                feature = utterance_cmvn(feature)
            difference = args.max_audio_frames - feature.size(0)

            if difference < 0:
                continue
            else:
                src_audio = torch.cat([feature] + [padding_vector] * difference)
                seg_audio = [1] * int(feature.size(0) / args.conv_layers_num / 2) + [0] * (int(args.max_audio_frames /args.conv_layers_num / 2) - int(feature.size(0) / args.conv_layers_num / 2))

            tgt_in = tgt_text[:-1]
            tgt_out = tgt_text[1:]
            tgt_seg = [1] * (len(tgt_text[1:]) - pad_num) + [0] * pad_num

            dataset.append((src_audio, tgt_in, tgt_out, seg_audio, tgt_seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam width.")
    parser.add_argument("--max_step", type=int, default=100,
                        help="inference step.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build s2t model.
    model = Text2text(args)
    model = load_model(model, args.load_model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src_audio = torch.stack([sample[0] for sample in dataset], dim=0)
    seg_audio = torch.LongTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size
    beam_width=args.beam_width
    instances_num = src_audio.size()[0]
    max_step = args.max_step

    print("The number of prediction instances: ", instances_num)

    model.eval()
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        for i, (src_batch, seg_batch) in tqdm.tqdm(enumerate(batch_loader(batch_size, src_audio, seg_audio))):
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)

            seq_length = seg_batch.sum(dim=-1).max()
            src_batch = src_batch[:,:seq_length * 4,:]
            seg_batch = seg_batch[:,:seq_length]

            tgt_in_batch = torch.zeros(src_batch.size()[0], 1, dtype = torch.long, device = args.device)
            current_batch_size=tgt_in_batch.size()[0]
            for j in range(current_batch_size):
                tgt_in_batch[j][0] = torch.LongTensor(args.tokenizer.convert_tokens_to_ids(['</s>']))

            with torch.no_grad():
                memory_bank, emb = model(src_batch, None, seg_batch, None, only_use_encoder=True, use_audio_input=True)

            step = 0
            scores = torch.zeros(current_batch_size, beam_width, max_step)
            tokens = torch.zeros(current_batch_size, beam_width, max_step+1, dtype = torch.long)
            tokens[:,:,0] = torch.LongTensor(args.tokenizer.convert_tokens_to_ids(['</s>'])) #2
            emb = emb.repeat(1, beam_width, 1).reshape(current_batch_size * beam_width, -1, int(args.conv_channels[-1] / 2)) #same batch nearby
            memory_bank = memory_bank.repeat(1, beam_width, 1).reshape(current_batch_size * beam_width, -1, args.emb_size) 
            tgt_in_batch = tgt_in_batch.repeat(beam_width, 1)
            while step < max_step and step < seq_length:
                with torch.no_grad():
                    outputs = model(emb, (tgt_in_batch, None, None), None, None, memory_bank=memory_bank, use_audio_input=True)

                vocab_size = outputs.shape[-1]
                log_prob = F.log_softmax(outputs[:, -1, :], dim=-1) #(B*beam_size) * 1 * vocab_size
                log_prob = log_prob.squeeze() #(B*beam_size) * vocab_size
                log_prob[:,1] = -math.inf # do not select pad
                if step == 0:
                    log_prob[:,2] = -math.inf # </s>

                log_prob_beam = log_prob.reshape(current_batch_size, beam_width, -1) # B * beam * vocab_size

                if step == 0:
                    log_prob_beam = log_prob_beam[:, ::beam_width, :].contiguous().to(scores.device)
                else:
                    log_prob_beam = log_prob_beam.to(scores.device) + scores[:,:, step-1].unsqueeze(-1)
                
                top_prediction_prob, top_prediction_indices = torch.topk(log_prob_beam.view(current_batch_size, -1), k=beam_width)
                beams_buf = torch.div(top_prediction_indices, vocab_size).trunc().long()
                beams_buf = beams_buf + torch.arange(current_batch_size).repeat(beam_width).reshape(beam_width,-1).transpose(0,1) * beam_width
                top_prediction_indices = top_prediction_indices.fmod(vocab_size)
                
                scores[:, :, step] = top_prediction_prob
                tokens[:, :, step+1] = top_prediction_indices

                if step > 0 and current_batch_size == 1:
                    tokens[:, :, :step+1] = torch.index_select(tokens, dim=1, index=beams_buf.squeeze())[:, :, :step+1]
                elif step > 0:
                    tokens[:, :, step+1] = torch.index_select(tokens.reshape(-1,tokens.shape[2]), dim=0, index=beams_buf.reshape(-1)).reshape(current_batch_size, -1, tokens.shape[2])[:, :, step+1]
                tgt_in_batch = tokens[:, :, :step+2].view(current_batch_size * beam_width, -1)
                tgt_in_batch = tgt_in_batch.long().to(emb.device)

                step = step + 1
            for i in range(current_batch_size):
                for j in range(1):
                    res = "".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in tokens[i,j,:]])
                    res = res.split("</s>")[1].split("<s>")[0]
                    res = res.replace('▁',' ')
                    print(res)
                    f.write(res)
                    f.write("\n")


if __name__ == "__main__":
    main()
