import torch
import torch.nn as nn

from tencentpretrain.utils.constants import *
from tencentpretrain.decoders import *
from tencentpretrain.embeddings import *


class DecodeTarget(nn.Module):
    """
    Language Model Target
    """

    def __init__(self, args, vocab_size):
        super(DecodeTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        args.decoder_layers_num = 2
        self.tgt_embedding = Embedding(args)
        for embedding_name in args.tgt_embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.tgt_embedding.update(tmp_emb, embedding_name)
        self.decoder = str2decoder["transformer"](args)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=args.has_lmtarget_bias)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def lm(self, memory_bank, tgt_lm):
        # Language modeling (LM) with full softmax prediction.

        tgt_lm = tgt_lm.contiguous().view(-1)
        memory_bank = memory_bank.contiguous().view(-1, self.hidden_size)
        memory_bank = memory_bank[tgt_lm > 0, :]
        tgt_lm = tgt_lm[tgt_lm > 0]
        output = self.output_layer(memory_bank)
        output = self.softmax(output)
        denominator = torch.tensor(output.size(0) + 1e-6)
        if output.size(0) == 0:
            correct = torch.tensor(0.0)
        else:
            correct = torch.sum((output.argmax(dim=-1).eq(tgt_lm)).float())

        loss = self.criterion(output, tgt_lm)


        return loss, correct, denominator

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """
        tgt_in, tgt_out, tgt_seg = tgt
        decoder_emb = self.tgt_embedding(tgt_in, tgt_seg)
        hidden = self.decoder(memory_bank, decoder_emb, (seg,))
        # Language modeling (LM) with full softmax prediction.
        loss, correct, denominator = self.lm(hidden, tgt_out)

        return loss, correct, denominator
