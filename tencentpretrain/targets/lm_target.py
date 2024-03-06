import torch
import torch.nn as nn

from tencentpretrain import mpu
from tencentpretrain.utils.constants import *


class LmTarget(nn.Module):
    """
    Language Model Target
    """

    def __init__(self, args, vocab_size):
        super(LmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        if hasattr(args, "tensor_model_parallel_size"):
            self.tensor_model_parallel_size = args.tensor_model_parallel_size
        else:
            self.tensor_model_parallel_size = 1
        if hasattr(args, "pipeline_model_parallel_size"):
            self.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        else:
            self.pipeline_model_parallel_size = 1
        if "label_smoothing" in args:
            self.label_smoothing = args.label_smoothing
        else:
            self.label_smoothing = None
        if "ignore_index" in args and args.ignore_index:
            self.ignore_index = args.tokenizer.vocab.get(PAD_TOKEN)
        else:
            self.ignore_index = None
        self.prefix_lm_loss = args.prefix_lm_loss
        if self.tensor_model_parallel_size > 1:
            self.output_layer = mpu.ColumnParallelLinear(self.hidden_size, self.vocab_size, gather_output=False,
                                                         skip_bias_add=True, bias=False)
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=args.has_lmtarget_bias)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def lm(self, memory_bank, tgt_lm, seg):
        # Language modeling (LM) with full softmax prediction.
        seg = seg.contiguous().view(-1)

        memory_bank = memory_bank.contiguous().view(-1, self.hidden_size)

        loss_mask = 1 if self.prefix_lm_loss else 0
        # For example seg=[1,1,1,2,2,0], when loss_prefix = 0 , parts of seg > 0 tokens are computed loss
        # when loss_prefix=1 , only parts of seg = 2 tokens are computed loss
        memory_bank = memory_bank[seg > loss_mask, :]
        if tgt_lm is not None:
            tgt_lm = tgt_lm.contiguous().view(-1)
            tgt_lm = tgt_lm[seg > loss_mask]

        output = self.output_layer(memory_bank)
        if self.pipeline_model_parallel_size > 1:

            return output, loss_mask
        elif self.tensor_model_parallel_size > 1:
            losses = mpu.vocab_parallel_cross_entropy(output, tgt_lm)
            loss = torch.sum(losses.view(-1)) / len(losses)
        else:
            output = self.softmax(output)
            denominator = torch.tensor(output.size(0) + 1e-6)

            if self.label_smoothing is None:
                loss = self.criterion(output, tgt_lm)
            else:
                if tgt_lm.dim() == output.dim() - 1:
                    tgt_lm = tgt_lm.unsqueeze(-1)
                nll_loss = -output.gather(dim=-1, index=tgt_lm)
                smooth_loss = -output.sum(dim=-1, keepdim=True)
                if self.ignore_index is not None:
                    pad_mask = tgt_lm.eq(self.ignore_index)
                    nll_loss.masked_fill_(pad_mask, 0.0)
                    smooth_loss.masked_fill_(pad_mask, 0.0)
                else:
                    nll_loss = nll_loss.squeeze(-1)
                    smooth_loss = smooth_loss.squeeze(-1)
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()
                eps_i = self.label_smoothing / (output.size(-1) - 1)
                loss = (1.0 - self.label_smoothing - eps_i) * nll_loss + eps_i * smooth_loss

        return loss

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
        """
        # Language modeling (LM) with full softmax prediction.
        loss = self.lm(memory_bank, tgt, seg)

        return loss
