from tencentpretrain.layers.layer_norm import *
from tencentpretrain.layers.multi_headed_attn import *
from tencentpretrain.layers.position_ffn import *
import torch.nn as nn


str2layernorm = {"t5": T5LayerNorm, "rms": RMSNorm, "normal": LayerNorm}

str2feedforward = {"gated": GatedFeedForward, "dense": PositionwiseFeedForward}

str2parallelfeedforward = {"gated": ParallelGatedFeedForward, "dense": ParallelPositionwiseFeedForward}

__all__ = ["T5LayerNorm", "RMSNorm", "LayerNorm", "GatedFeedForward", "PositionwiseFeedForward", "ParallelGatedFeedForward",
           "ParallelPositionwiseFeedForward", "str2layernorm", "str2feedforward", "str2parallelfeedforward"]