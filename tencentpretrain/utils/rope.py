import torch
import math
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def get_ntk_alpha(true_seq_len, max_seq_length):
    context_value = math.log(true_seq_len / max_seq_length, 2) + 1
    ntk_alpha = 2 ** math.ceil(context_value) - 1
    ntk_alpha = max(ntk_alpha, 1)
    return ntk_alpha

def update_freqs_cis(dim: int, end: int, theta: float = 10000.0, ntk_alpha: float = 1.0):
    theta = theta * ntk_alpha ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    seq = torch.arange(end, device=freqs.device)
    freqs = torch.outer(seq.type_as(freqs), freqs)
    emb = torch.cat((freqs, freqs), dim=-1)

    from einops import rearrange
    emb = rearrange(emb, "n d -> 1 n 1 d")

    cos, sin = emb.cos(), emb.sin()
    return [cos[:, :end], sin[:, :end]]

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).transpose(1,2), xk_out.type_as(xk).transpose(1,2)

def _rotate_half(x: torch.Tensor):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t: torch.Tensor, freqs: list[torch.Tensor, torch.Tensor]):
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.float()
    t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
    cos = cos.to(t_rot.device)
    sin = sin.to(t_rot.device)
    t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)
    return torch.cat((t_rot, t_pass), dim=-1).type_as(t).transpose(1, 2)