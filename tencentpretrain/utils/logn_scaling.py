import torch

def apply_logn_scaling(key_size: int,
                       query_size: int,
                       logn_tensor: torch.Tensor,
                       xq: torch.Tensor
) -> torch.tensor:
    seq_start = key_size - query_size
    seq_end = key_size
    logn_tensor = logn_tensor[:, :, seq_start:seq_end, :].type_as(xq)
    return xq * logn_tensor.expand_as(xq)
