import torch
import torch.nn as nn
import math


class SpeechEmbedding(nn.Module):
    """
    """
    def __init__(self, args, _):
        super(SpeechEmbedding, self).__init__()
        self.conv = Conv1dModule(args)
        self.sinusoidalpos = False
        self.emb_size = args.emb_size
        if "sinusoidalpos" in args.embedding:
            self.sinusoidalpos = True

    def forward(self, src, _):
        """Embed inputs.
        Args:
            src (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        speech_emb = self.conv(src)
        if self.sinusoidalpos:
            return speech_emb * math.sqrt(self.emb_size)
        else:
            return speech_emb


class Transpose_module(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.transpose(-2, -1)


class Conv1dModule(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, args):
        super(Conv1dModule, self).__init__()
        self.embedding_dim = args.emb_size
        self.norm_mode = None
        self.feature_grad_mult = 1.0
        self.conv_bias = True
        self.dropout_input = 0.0
        self.use_glu = True if args.data_processor == "s2t" else False
        self.padding = True

        self.conv_channels = args.conv_channels
        self.audio_feature_size = args.audio_feature_size
        self.kernel_sizes = args.conv_kernel_sizes
        self.strides = [2 for _ in range(len(self.kernel_sizes))]

        self.conv_layers = nn.ModuleList()
        
        def conv_layer_block(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            norm_mode=None,
            conv_bias=False,
        ):
            def make_conv(in_channels, out_channels, kernel_size, stride, padding, conv_bias):
                conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if norm_mode == "layer":
                return nn.Sequential(
                    make_conv(in_channels, out_channels, kernel_size, stride, padding, conv_bias),
                    Transpose_module(),
                    nn.LayerNorm(out_channels, eps=1e-5, elementwise_affine=True),
                    Transpose_module(),
                    nn.GELU(),
                )
            elif norm_mode == "group":
                return nn.Sequential(
                    make_conv(in_channels, out_channels, kernel_size, stride, padding, conv_bias),
                    nn.GroupNorm(out_channels, out_channels, eps=1e-5, affine=True),
                    nn.GELU(),
                )
            elif self.use_glu:
                return nn.Sequential(
                    make_conv(in_channels, out_channels, kernel_size, stride, padding, conv_bias),
                )
            else:
                return nn.Sequential(
                    make_conv(in_channels, out_channels, kernel_size, stride, padding, conv_bias),
                    nn.GELU(),
                )
        assert len(self.strides) == len(self.kernel_sizes), "strides and kernel_sizes are not matched"
        assert len(self.strides) == len(self.conv_channels), "strides and conv_channels are not matched"
        in_channel = self.conv_channels[0] // 2
        for i, (k, s, c) in enumerate(zip(self.kernel_sizes, self.strides, self.conv_channels)):
            if self.audio_feature_size == 1:
                in_channel = c
            if self.norm_mode == "group" and i != 0:
                self.norm_mode = None
            if self.padding:
                padding = k // 2
            else:
                padding = 0
            self.conv_layers.append(
                conv_layer_block(
                    self.audio_feature_size if i == 0 else in_channel,
                    c,
                    k,
                    s,
                    padding,
                    norm_mode=self.norm_mode,
                    conv_bias=self.conv_bias,
                )
            )


    def forward(self, input_features, mask_indices=None, mask_channel_indices=None):
        if len(input_features.size()) == 2:
            hidden_states = input_features.unsqueeze(1) # wav B x T -> B x (C x D) x T
        else:
            hidden_states = input_features.transpose(1, 2).contiguous()  #acoustic feature B x T x (C x D) -> B x (C x D) x T

        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
            if self.use_glu:
                hidden_states = nn.functional.glu(hidden_states, dim=1)

        hidden_states = hidden_states.transpose(1, 2).contiguous()  # -> B x T x (C x D)

        return hidden_states
