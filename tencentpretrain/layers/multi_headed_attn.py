import math
import torch
import torch.nn as nn
from tencentpretrain import mpu
from tencentpretrain.utils.rope import apply_rotary_emb
from tencentpretrain.utils.lora import LoraLinear


def repeat_kv(x: torch.Tensor, repeat_num: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, seq_length, kv_heads_num, head_dim = x.shape
    if repeat_num == 1:
        return x

    else:
        return (
            x[:, :, :, None, :]
            .expand(bs, seq_length, kv_heads_num, repeat_num, head_dim)
            .reshape(bs, seq_length, kv_heads_num * repeat_num, head_dim)
        )


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, local_kv_heads_num, dropout, has_bias=True, with_scale=True,
                 lora_params=None, layer_number=None):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size
        self.local_kv_heads_num = local_kv_heads_num

        self.kv_embed_dim = self.inner_hidden_size // heads_num * self.local_kv_heads_num
        self.num_head_groups = heads_num // self.local_kv_heads_num
        assert heads_num >= self.local_kv_heads_num, "heads_num should be greater than or equal to n_local_kv_heads"
        assert heads_num % self.local_kv_heads_num == 0, "heads_num should be divisible by n_local_kv_heads"
        self.repeat_num = self.heads_num // self.local_kv_heads_num

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
        # layer-wise attention scaling
        if layer_number is not None:
            self.layer_number = max(1, layer_number)
            self.norm_factor = math.sqrt(self.per_head_size) * self.layer_number
        else:
            self.layer_number = None

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None,
                freqs_cis=None, alibi=None):
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

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, self.inner_hidden_size)

        query, key, value = [linear_layer(x) for linear_layer, x in zip(self.linear_layers, [query, key, value])]

        query = query.view(batch_size, -1, heads_num, per_head_size)
        key = key.view(batch_size, -1, self.local_kv_heads_num, per_head_size)
        value = value.view(batch_size, -1, self.local_kv_heads_num, per_head_size)

        query = query.transpose(1, 2)
        key = repeat_kv(key, self.repeat_num).transpose(1, 2)
        value = repeat_kv(value, self.repeat_num).transpose(1, 2)


        if freqs_cis is not None:
            query, key = apply_rotary_emb(query.transpose(1,2), key.transpose(1,2), freqs_cis=freqs_cis)

        scores = torch.matmul(query, key.transpose(-2, -1))

        if position_bias is not None:
            scores = scores + position_bias

        if self.with_scale:
            if self.layer_number is not None:
                scores = scores * (1.0 / self.norm_factor)
            else:
                scores = scores / math.sqrt(float(per_head_size))
        if alibi is not None:
            scores = scores.reshape((-1, scores.shape[-2], scores.shape[-1]))
            scores += (1.0 / self.layer_number) * alibi
            scores = scores.view(-1, heads_num, scores.shape[-2], scores.shape[-1])

        scores = scores + mask.type_as(scores)

        # scaled softmax
        if self.layer_number is not None:
            scores = (scores * self.layer_number) + mask
            scores = torch.max(scores, torch.tensor(-10000))

        prev_attn_out = None

        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores

        # probs = nn.Softmax(dim=-1)(scores)
        probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output, prev_attn_out


class ParallelMultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, local_kv_heads_num, dropout, has_bias=True, with_scale=True,
                 lora_params=None, layer_number=None):
        super(ParallelMultiHeadedAttention, self).__init__()
        self.heads_num = heads_num
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size
        self.local_kv_heads_num = local_kv_heads_num

        self.kv_embed_dim = self.inner_hidden_size // heads_num * self.local_kv_heads_num
        self.num_head_groups = heads_num // self.local_kv_heads_num
        assert heads_num >= self.local_kv_heads_num, "heads_num should be greater than or equal to n_local_kv_heads"
        assert heads_num % self.local_kv_heads_num == 0, "heads_num should be divisible by n_local_kv_heads"
        self.repeat_num = self.heads_num // self.local_kv_heads_num
        self.linear_layers = nn.ModuleList(
            [
                mpu.ColumnParallelLinear(hidden_size, self.inner_hidden_size, skip_bias_add=False if has_bias else True, gather_output=False) if i==0 else 
                mpu.ColumnParallelLinear(hidden_size, self.kv_embed_dim, skip_bias_add=False if has_bias else True, gather_output=False) for i in range(3)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_linear = mpu.RowParallelLinear(self.inner_hidden_size, hidden_size, bias=has_bias, input_is_parallel=True, skip_bias_add=False if has_bias else True)
        # layer-wise attention scaling
        if layer_number is not None:
            self.layer_number = max(1, layer_number)
            self.norm_factor = math.sqrt(self.per_head_size) * self.layer_number
        else:
            self.layer_number = None

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None,
                freqs_cis=None, alibi=None):
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

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, -1)

        query, key, value = [linear_layer(x) for linear_layer, x in zip(self.linear_layers, [query, key, value])]

        query = query.view(batch_size, seq_length, -1, per_head_size)
        key = key.view(batch_size, seq_length, -1, per_head_size)
        value = value.view(batch_size, seq_length, -1, per_head_size)

        query = query.transpose(1, 2)
        key = repeat_kv(key, self.repeat_num).transpose(1, 2)
        value = repeat_kv(value, self.repeat_num).transpose(1, 2)
        
        if freqs_cis is not None:
            query, key = apply_rotary_emb(query.transpose(1,2), key.transpose(1,2), freqs_cis=freqs_cis)

        scores = torch.matmul(query, key.transpose(-2, -1))

        if position_bias is not None:
            scores = scores + position_bias

        if self.with_scale:
            if self.layer_number is not None:
                scores = scores * (1.0 / self.norm_factor)
            else:
                scores = scores / math.sqrt(float(per_head_size))
        if alibi is not None:
            scores = scores.reshape((-1, scores.shape[-2], scores.shape[-1]))
            scores += (1.0 / self.layer_number) * alibi
            scores = scores.view(-1, heads_num, scores.shape[-2], scores.shape[-1])

        scores = scores + mask.type_as(scores)

        # scaled softmax
        if self.layer_number is not None:
            scores = (scores * self.layer_number) + mask
            scores = torch.max(scores, torch.tensor(-10000))

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
