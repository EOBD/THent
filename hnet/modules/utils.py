from dataclasses import asdict

import torch


def get_seq_idx(cu_seqlens, device=None):
    if cu_seqlens is None or cu_seqlens.numel() <= 1 or cu_seqlens[-1] == 0:
        return torch.zeros((1, 0), dtype=torch.int32, device=device)
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    return {
        k: (v[stage_idx] if stage_idx < len(v) else v[-1]) if isinstance(v, list) and len(v) > 0 else v
        for k, v in asdict(cfg).items()
    }


def apply_optimization_params(
    param: torch.Tensor,
    **kwargs,
) -> None:
    """
    Annotates a parameter with optimization parameters.

    Specifically, updates the parameter's `_optim` attribute with the given kwargs.
    """

    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs
