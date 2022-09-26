import torch
import torch.nn as nn

from tencentpretrain.embeddings.word_embedding import WordEmbedding
from tencentpretrain.embeddings.patch_embedding import PatchEmbedding


class WordPatchEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordPatchEmbedding, self).__init__()
        self.language_embedding = WordEmbedding(args, vocab_size)
        self.vision_embedding = PatchEmbedding(args, None)


    def forward(self, src, _):
        l_emb = self.language_embedding(src[0], None)
        v_emb = self.vision_embedding(src[1], None)
        emb = torch.cat([l_emb, v_emb], dim=1)

        return emb
