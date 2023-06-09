from tencentpretrain.layers.layer_norm import *
from tencentpretrain.layers.multi_headed_attn import *
from tencentpretrain.layers.position_ffn import *

str2layernorm = {"t5": T5LayerNorm, "rms": RMSNorm, "normal_torch": TorchLayerNorm, "normal": LayerNorm}

str2attention = {"multi_head": MultiHeadedAttention, "flash_attention": FlashAttention}

str2feedforward = {"gated": GatedFeedForward, "dense": PositionwiseFeedForward}

__all__ = ["T5LayerNorm", "RMSNorm", "TorchLayerNorm", "LayerNorm", "MultiHeadedAttention",
           "FlashAttention", "GatedFeedForward", "PositionwiseFeedForward", "str2layernorm",
           "str2attention", "str2feedforward"]

