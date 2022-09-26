import torch.nn as nn


class SpeechEmbedding(nn.Module):
    """
    """

    def __init__(self, args, _):
        super(SpeechEmbedding, self).__init__()
        self.conv = Conv1dSubsampler(args)


    def forward(self, src, _):
        """Embed inputs.
        Args:
            src (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        speech_emb = self.conv(src)

        return speech_emb


class Conv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, args):
        super(Conv1dSubsampler, self).__init__()
        self.layers_num = args.conv_layers_num
        self.in_channels = args.audio_feature_size
        self.mid_channels = args.conv_channels
        self.out_channels = args.emb_size
        self.kernel_sizes = args.conv_kernel_sizes

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                self.in_channels if i == 0 else self.mid_channels // 2,
                self.mid_channels if i < self.layers_num - 1 else self.out_channels * 2,
                kernel_size=k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(self.kernel_sizes)
        )

    def forward(self, input_features):
        hidden_states = input_features.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.glu(hidden_states, dim=1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # -> T x B x (C x D)
        return hidden_states
