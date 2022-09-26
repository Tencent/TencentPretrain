from tencentpretrain.embeddings.embedding import Embedding
from tencentpretrain.embeddings.dual_embedding import DualEmbedding
from tencentpretrain.embeddings.word_embedding import WordEmbedding
from tencentpretrain.embeddings.pos_embedding import PosEmbedding
from tencentpretrain.embeddings.seg_embedding import SegEmbedding
from tencentpretrain.embeddings.sinusoidalpos_embedding import SinusoidalposEmbedding
from tencentpretrain.embeddings.patch_embedding import PatchEmbedding
from tencentpretrain.embeddings.word_patch_embedding import WordPatchEmbedding
from tencentpretrain.embeddings.speech_embedding import SpeechEmbedding


str2embedding = {"word": WordEmbedding, "pos": PosEmbedding, "seg": SegEmbedding,
                 "sinusoidalpos": SinusoidalposEmbedding, "dual": DualEmbedding,
                 "patch": PatchEmbedding, "word_patch": WordPatchEmbedding, "speech": SpeechEmbedding}

__all__ = ["Embedding", "WordEmbedding", "PosEmbedding", "SegEmbedding", "SinusoidalposEmbedding",
           "DualEmbedding", "PatchEmbedding", "WordPatchEmbedding", "SpeechEmbedding", "str2embedding"]
