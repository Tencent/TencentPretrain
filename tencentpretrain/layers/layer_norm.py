import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    https://arxiv.org/abs/1607.06450
    """
    def __init__(self, hidden_size, eps=1e-6, eps_inside=False):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.eps_inside = eps_inside
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        if self.eps_inside:
            std = torch.sqrt(x.var(-1, keepdim=True) + self.eps)
        else:
            std = x.std(-1, keepdim=True) + self.eps
        hidden_states = self.gamma * (x-mean) / std

        return hidden_states + self.beta


class T5LayerNorm(nn.Module):
    """
    Construct a layernorm module in the T5 style No bias and no subtraction of mean.
    """
    def __init__(self, hidden_size, eps=1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.type_as(self.weight)


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
