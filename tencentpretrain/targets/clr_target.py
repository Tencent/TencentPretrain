import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tencentpretrain.utils.misc import pooling


class ClrTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(ClrTarget, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = args.batch_size

        self.criterion_0 = nn.CrossEntropyLoss()
        self.criterion_1 = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.pooling_type = [args.stream_0["pooling"], args.stream_1["pooling"]]

        if args.projection:
            self.projection = True
            self.encoder_0_projection = nn.Parameter(torch.randn(args.stream_0["hidden_size"], args.feature_size))
            self.encoder_1_projection = nn.Parameter(torch.randn(args.stream_1["hidden_size"], args.feature_size))
        else:
            self.projection = False


    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        embedding_0, embedding_1 = memory_bank
        features_0 = pooling(embedding_0, seg[0], self.pooling_type[0])
        features_1 = pooling(embedding_1, seg[1], self.pooling_type[1])
        if self.projection:
            features_0 = torch.matmul(features_0, self.encoder_0_projection)
            features_1 = torch.matmul(features_1, self.encoder_1_projection)

        features_0 = features_0 / features_0.norm(dim=-1, keepdim=True)
        features_1 = features_1 / features_1.norm(dim=-1, keepdim=True)

        # https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L169
        # Gather all embeddings if using distributed training
        if dist.is_initialized():
            # Dummy vectors for allgather
            features_0_list = [torch.zeros_like(features_0) for _ in range(dist.get_world_size())]
            features_1_list = [torch.zeros_like(features_1) for _ in range(dist.get_world_size())]

            # Allgather
            dist.all_gather(tensor_list=features_0_list, tensor=features_0.contiguous())
            dist.all_gather(tensor_list=features_1_list, tensor=features_1.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            features_0_list[dist.get_rank()] = features_0
            features_1_list[dist.get_rank()] = features_1

            # Get full batch embeddings: (bs x N, hidden)
            features_0 = torch.cat(features_0_list, 0)
            features_1 = torch.cat(features_1_list, 0)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_0 = logit_scale * torch.matmul(features_0, features_1.transpose(-2, -1))
        logits_1 = logit_scale * torch.matmul(features_1 , features_0.transpose(-2, -1))


        tgt = torch.arange(features_0.size()[0], device = logits_0.device, dtype=torch.long)
        loss = (self.criterion_0(logits_0, tgt) + self.criterion_1(logits_1, tgt)) / 2
        if dist.is_initialized():
            correct = self.softmax(logits_0).argmax(dim=-1).eq(tgt).sum() / dist.get_world_size()
        else:
            correct = self.softmax(logits_0).argmax(dim=-1).eq(tgt).sum()
        return loss, correct
