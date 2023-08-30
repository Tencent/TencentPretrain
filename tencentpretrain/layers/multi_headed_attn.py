import math
import torch
import torch.nn as nn
from tencentpretrain.utils.rope import apply_rotary_emb
from tencentpretrain.utils.lora import LoraLinear

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, n_local_kv_heads, dropout, has_bias=True, with_scale=True,
                 lora_params=None):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size
        self.n_local_kv_heads = n_local_kv_heads

        self.kv_embed_dim = self.inner_hidden_size // heads_num * self.n_local_kv_heads
        self.num_head_groups = heads_num // self.n_local_kv_heads
        assert heads_num >= self.n_local_kv_heads, "heads_num should be greater than or equal to n_local_kv_heads"
        assert heads_num % self.n_local_kv_heads == 0, "heads_num should be divisible by n_local_kv_heads"
        self.n_rep = self.heads_num // self.n_local_kv_heads

        if lora_params is not None:

            self.linear_layers = nn.ModuleList(
                [LoraLinear(hidden_size, self.inner_hidden_size, r=lora_params['lora_r'],
                             lora_alpha=lora_params['lora_alpha'],
                             lora_dropout=lora_params['lora_dropout'], bias=has_bias),
                 nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias),
                 LoraLinear(hidden_size, self.inner_hidden_size, r=lora_params['lora_r'],
                             lora_alpha=lora_params['lora_alpha'],
                             lora_dropout=lora_params['lora_dropout'], bias=has_bias)]
            )
        else:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) if i==0 else nn.Linear(hidden_size, self.kv_embed_dim, bias=has_bias) for i in range(3)]
            )
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None,
                freqs_cis=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size
        n_local_kv_heads = self.n_local_kv_heads

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_hidden_size)

        query, key, value = [linear_layer(x) for linear_layer, x in zip(self.linear_layers, [query, key, value])]

        query = query.view(batch_size, seq_length, heads_num, per_head_size)
        key = key.view(batch_size, seq_length, n_local_kv_heads, per_head_size)
        value = value.view(batch_size, seq_length, n_local_kv_heads, per_head_size)

        if freqs_cis is not None:
            query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

        query = query.transpose(1, 2)
        key = repeat_kv(key, self.n_rep).transpose(1, 2)
        value = repeat_kv(value, self.n_rep).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3))

        if position_bias is not None:
            scores = scores + position_bias

        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))

        scores = scores + mask.type_as(scores)
        prev_attn_out = None

        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output, prev_attn_out

