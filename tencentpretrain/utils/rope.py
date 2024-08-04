import torch
import math
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=inv_freq.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq) 
    return freqs

def get_ntk_alpha(true_seq_len, max_seq_length):
    context_value = math.log(true_seq_len / max_seq_length, 2) + 1
    ntk_alpha = 2 ** math.ceil(context_value) - 1
    ntk_alpha = max(ntk_alpha, 1)
    return ntk_alpha

def update_freqs_cis(freqs: torch.Tensor,
                     seq_length: int,
                     max_seq_length: int,
                     dim: int,
                     seq_len_cached: int=0,
                     ntk_alpha_cached: float=1.0,
                     theta: float = 10000.0
):
    ntk_alpha = get_ntk_alpha(seq_length, max_seq_length) if seq_length > max_seq_length else 1.0
    if seq_length > seq_len_cached or ntk_alpha != ntk_alpha_cached:
        theta = theta * ntk_alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        end = seq_length * 2
        t = torch.arange(end, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        seq_len_cached = end
        ntk_alpha_cached = ntk_alpha
    return freqs, seq_len_cached, ntk_alpha_cached

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids=None,
    use_rotate_half: bool=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_rotate_half:
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        dtype = xq.dtype
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs_cis, freqs_cis), dim=-1)
        cos = emb.cos().to(dtype).to(xq.device)
        sin = emb.sin().to(dtype).to(xq.device)
        if position_ids is None:
            _, _, seq_len, _ = xq.shape
            cos, sin = cos[:seq_len], sin[:seq_len]
        else:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

        xq_out = (xq * cos) + (rotate_half(xq) * sin)
        xk_out = (xk * cos) + (rotate_half(xk) * sin)
        return xq_out, xk_out
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)  # complex64
        
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq).transpose(1,2), xk_out.type_as(xk).transpose(1,2)
