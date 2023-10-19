"""
This script provides an example to wrap TencentPretrain for speech-to-text fine-tuning.
"""
import sys
import os
import random
import argparse
import editdistance
import math
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.model_saver import save_model
from tencentpretrain.decoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils import utterance_cmvn
from finetune.run_classifier import *


class Speech2text(torch.nn.Module):
    def __init__(self, args):
        super(Speech2text, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.tgt_embedding = Embedding(args)
        for embedding_name in args.tgt_embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.tgt_embedding.update(tmp_emb, embedding_name)
        self.decoder = str2decoder[args.decoder](args)
        self.target = Target()
        for target_name in args.target:
            tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
            self.target.update(tmp_target, target_name)
        if args.tie_weights:
            self.target.lm.output_layer.weight = self.tgt_embedding.word.embedding.weight

    def encode(self, src, seg):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        return memory_bank, emb

    def decode(self, emb, memory_bank, tgt, tgt_seg):
        tgt_in, tgt_out, _ = tgt
        decoder_emb = self.tgt_embedding(tgt_in, tgt_seg)
        hidden = self.decoder(memory_bank, decoder_emb, [emb.abs()[:,:,0]])
        output = self.target.lm.output_layer(hidden)
        return output

    def forward(self, src, tgt, seg, tgt_seg, memory_bank=None, only_use_encoder=False):
        if only_use_encoder:
            return self.encode(src, seg)
        if memory_bank is not None:
            emb = src
            return self.decode(emb, memory_bank, tgt, tgt_seg)
        tgt_in, tgt_out, _ = tgt
        memory_bank, emb = self.encode(src, seg)

        if tgt_out is None:
            output = self.decode(emb, memory_bank, tgt, None)
            return None, output
        else:
            decoder_emb = self.tgt_embedding(tgt_in, tgt_seg)
            hidden = self.decoder(memory_bank, decoder_emb, (seg,))
            loss = self.target(hidden, tgt_out, tgt_seg)[0]
            return loss, None


def read_dataset(args, path):
    dataset, columns = [], {}
    padding_vector = torch.FloatTensor(args.audio_feature_size * [0.0] if args.audio_feature_size > 1 else 0.0).unsqueeze(0)

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            text, wav_path = line[columns["text"]], line[columns["wav_path"]]
            tgt = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN]) + \
                args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text)) + \
                args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])

            if len(tgt) > args.seq_length:
                tgt = tgt[: args.seq_length]

            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])
            pad_num = args.seq_length - len(tgt)
            tgt = tgt + PAD_ID * pad_num

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

            tgt_in = tgt[:-1]
            tgt_out = tgt[1:]
            tgt_seg = [1] * (len(tgt[1:]) - pad_num) + [0] * pad_num

            dataset.append((src_audio, tgt_in, tgt_out, seg_audio, tgt_seg))

    return dataset


def batch_loader(batch_size, src, tgt_in, tgt_out, seg, tgt_seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_in_batch = tgt_in[i * batch_size : (i + 1) * batch_size, :]
        tgt_out_batch = tgt_out[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        tgt_seg_batch = tgt_seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_in_batch = tgt_in[instances_num // batch_size * batch_size :, :]
        tgt_out_batch = tgt_out[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        tgt_seg_batch = tgt_seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch


def train_model(args, model, optimizer, scheduler, src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_in_batch = tgt_in_batch.to(args.device)
    tgt_out_batch = tgt_out_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    tgt_seg_batch = tgt_seg_batch.to(args.device)

    loss, _ = model(src_batch, (tgt_in_batch, tgt_out_batch, src_batch), seg_batch, tgt_seg_batch)

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):

    src = torch.stack([example[0] for example in dataset], dim=0)
    tgt_in = torch.LongTensor([example[1] for example in dataset])
    tgt_out = torch.LongTensor([example[2] for example in dataset])
    seg = torch.LongTensor([example[3] for example in dataset])
    tgt_seg = torch.LongTensor([example[4] for example in dataset])

    generated_sentences = []
    args.model.eval()
    PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])
    SEP_ID = args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])

    for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch) in enumerate(batch_loader(args.batch_size, src, tgt_in, tgt_out, seg, tgt_seg)):

        src_batch = src_batch.to(args.device)
        tgt_in_batch = torch.zeros(tgt_in_batch.size()[0], 1, dtype=torch.long, device=args.device)
        tgt_seg_batch  = torch.ones(tgt_in_batch.size()[0], 1, dtype=torch.long, device=args.device)
        for j in range(tgt_in_batch.size()[0]):
            tgt_in_batch[j][0] = args.tokenizer.vocab.get(CLS_TOKEN)

        seg_batch = seg_batch.to(args.device)

        with torch.no_grad():
            memory_bank, emb = args.model(src_batch, None, seg_batch, None, only_use_encoder=True)

        for step in range(args.tgt_seq_length):
            tgt_out_batch = tgt_in_batch
            with torch.no_grad():
                outputs = args.model(emb, (tgt_in_batch, None, None), None, None, memory_bank=memory_bank)

            next_token_logits = outputs[:, -1]
            log_prob = F.log_softmax(next_token_logits, dim=-1)
            log_prob[:,PAD_ID] = -math.inf # do not select pad
            if step == 0:
                log_prob[:,SEP_ID] = -math.inf # </s>

            next_tokens = torch.argmax(log_prob, dim=1).unsqueeze(1)
            tgt_in_batch = torch.cat([tgt_in_batch, next_tokens], dim=1)
            tgt_seg_batch  = torch.ones(tgt_in_batch.size()[0], tgt_in_batch.size()[1], dtype=torch.long, device=args.device)
        for j in range(len(outputs)):
            sentence = "".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in tgt_in_batch[j]])
            generated_sentences.append(sentence)

    w_errs = 0
    w_total = 0

    for i, example in enumerate(dataset):
        tgt = example[2]
        tgt_token = "".join([args.tokenizer.inv_vocab[token_id] for token_id in tgt[:-2]])
        generated_sentences[i] = generated_sentences[i].split(CLS_TOKEN)[1].split(SEP_TOKEN)[0]

        pred = generated_sentences[i].split("▁")
        gold = tgt_token.split(SEP_TOKEN)[0].split("▁")
        w_errs += editdistance.eval(pred, gold)
        w_total += len(gold)

    args.logger.info("WER. (Word_Errors/Total): {:.4f} ({}/{}) ".format(w_errs / w_total, w_errs, w_total))
    return w_errs / w_total


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--tgt_seq_length", type=int, default=50,
                        help="Output sequence length.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Speech2text(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 100.0

    args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.stack([example[0] for example in trainset], dim=0)
        tgt_in = torch.LongTensor([example[1] for example in trainset])
        tgt_out = torch.LongTensor([example[2] for example in trainset])
        seg = torch.LongTensor([example[3] for example in trainset])
        tgt_seg = torch.LongTensor([example[4] for example in trainset])

        model.train()
        for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch) in enumerate(batch_loader(batch_size, src, tgt_in, tgt_out, seg, tgt_seg)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_in_batch, tgt_out_batch, seg_batch, tgt_seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        save_model(model, args.output_model_path+"-ep"+str(epoch))
        if result < best_result:
            best_result = result
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
