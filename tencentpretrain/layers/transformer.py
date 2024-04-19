import torch
import torch.nn as nn
from tencentpretrain.layers.multi_headed_attn import MultiHeadedAttention, ParallelMultiHeadedAttention
from tencentpretrain.layers import *


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args, layer_number=None):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.rotary_position_embedding = args.rotary_position_embedding
        self.has_residual_attention = args.has_residual_attention
        if self.relative_position_embedding:
            self.relative_pos_emb = args.relative_pos_emb
        if self.rotary_position_embedding:
            self.freqs_cis = args.freqs_cis

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        if hasattr(args, "local_kv_heads_num"):
            local_kv_heads_num = args.local_kv_heads_num
        else:
            local_kv_heads_num = args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        lora_params = None
        if hasattr(args, "lora_params"):
            lora_params = args.lora_params

        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params, layer_number=layer_number
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = str2feedforward[args.feed_forward](
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        self.layer_norm_1 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_2 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)

    def forward(self, *inputs):

        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if len(inputs) == 2:
            hidden, mask = inputs
            prev_attn = None
        else:
            hidden, mask, prev_attn = inputs

        _, seq_length, _ = hidden.size()

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        if self.rotary_position_embedding:
            freqs_cis = self.freqs_cis[:seq_length].to(hidden.device)
        else:
            freqs_cis = None

        if self.layernorm_positioning == "post":
            inter, prev_attn_out = self.self_attn(hidden, hidden, hidden, mask, position_bias, self.has_residual_attention,
                                                  prev_attn, freqs_cis)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter, prev_attn_out = self.self_attn(inter, inter, inter, mask, position_bias, self.has_residual_attention,
                                                  prev_attn, freqs_cis)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden

        if self.has_residual_attention:
            return output, mask, prev_attn_out
        else:
            return output, mask


class ParallelTransformerLayer(nn.Module):

    def __init__(self, args, layer_number=None):
        super(ParallelTransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.rotary_position_embedding = args.rotary_position_embedding
        self.has_residual_attention = args.has_residual_attention
        if self.relative_position_embedding:
            self.relative_pos_emb = args.relative_pos_emb
        if self.rotary_position_embedding:
            self.freqs_cis = args.freqs_cis

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        if hasattr(args, "local_kv_heads_num"):
            local_kv_heads_num = args.local_kv_heads_num
        else:
            local_kv_heads_num = args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        lora_params = None
        if hasattr(args, "lora_params"):
            lora_params = args.lora_params

        self.self_attn = ParallelMultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params, layer_number=layer_number
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = str2parallelfeedforward[args.feed_forward](
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )

        self.dropout_2 = nn.Dropout(args.dropout)

        self.layer_norm_1 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_2 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)

    def forward(self, *inputs):

        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if len(inputs) == 2:
            hidden, mask = inputs
            prev_attn = None
        else:
            hidden, mask, prev_attn = inputs

        _, seq_length, _ = hidden.size()

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        if self.rotary_position_embedding:
            freqs_cis = self.freqs_cis[:seq_length].to(hidden.device)
        else:
            freqs_cis = None

        if self.layernorm_positioning == "post":
            inter, prev_attn_out = self.self_attn(hidden, hidden, hidden, mask, position_bias, self.has_residual_attention,
                                                  prev_attn, freqs_cis)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter, prev_attn_out = self.self_attn(inter, inter, inter, mask, position_bias, self.has_residual_attention,
                                                  prev_attn, freqs_cis)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden

        if self.has_residual_attention:
            return output, mask, prev_attn_out
        else:
            return output, mask


class ParallelTransformerLayerPipe(nn.Module):

    def __init__(self, args, model, layer_idx):
        super(ParallelTransformerLayerPipe, self).__init__()
        self.layer_idx = layer_idx
        self.layer_num = args.layers_num
        self.layernorm_positioning = args.layernorm_positioning
        self.has_residual_attention = args.has_residual_attention
        if self.layernorm_positioning == "pre":
           self.layer_norm = model.encoder.layer_norm 
        self.layer = model.encoder.transformer[layer_idx]

    def generate_mask(self, seq_length, batch_size, device):
        mask = torch.ones(seq_length, seq_length, device=device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = mask.repeat(batch_size, 1, 1, 1)
        return mask

    def forward(self, *inputs):

        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if len(inputs) == 2:
            hidden, seg = inputs
            prev_attn = None
        else:
            hidden, seg, prev_attn = inputs
        batch_size, seq_length, _ = hidden.size()
        mask = self.generate_mask(seq_length, batch_size, hidden.device)
        layer_inputs = hidden, mask, prev_attn
        outputs = self.layer(*layer_inputs)

        if self.has_residual_attention:
            hidden, mask, prev_attn_out = outputs
        else:
            hidden, mask = outputs

        if self.layer_idx == self.layer_num-1 and self.layernorm_positioning == "pre":
            hidden = self.layer_norm(hidden)

        if self.has_residual_attention:
            return hidden, seg, prev_attn
        else:
            return hidden, seg


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        if hasattr(args, "local_kv_heads_num"):
            local_kv_heads_num = args.local_kv_heads_num
        else:
            local_kv_heads_num = args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        lora_params = None
        if hasattr(args, "lora_params"):
            lora_params = args.lora_params

        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = str2feedforward[args.feed_forward](
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )
        self.dropout_3 = nn.Dropout(args.dropout)

        # Layer Normalization
        self.layer_norm_1 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_2 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_3 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)

    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias=None, context_position_bias=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            query, _ = self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query_norm = self.layer_norm_1(query + hidden)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query, _ = self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output
