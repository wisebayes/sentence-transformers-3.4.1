from __future__ import annotations
import torch
from torch import Tensor


def l1_distance(
    x: Tensor,
    y: Tensor,
    attention_mask: Tensor | None = None,
    reduce_tokens: bool = False) -> Tensor:

    Q, S, D = x.shape
    N, D2 = y.shape
    assert D == D2, "Dim mismatch"

    abs_diff = torch.abs(x.unsqueeze(2) - y.unsqueeze(0).unsqueeze(0))
    token_dist = abs_diff.sum(dim=-1)

    if attention_mask is not None:
    
        mask = attention_mask.unsqueeze(-1)
        token_dist = token_dist * mask

        sums = token_dist.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)  # avoid div0
        if reduce_tokens:
            return sums / counts          # [Q, N]
        else:
            return token_dist             # [Q, S, N]
    else:
        return token_dist.mean(dim=1) if reduce_tokens else token_dist


def hamming_zero_thresh(
    x: Tensor,
    y: Tensor,
    attention_mask: Tensor | None = None
) -> Tensor:

    xb = (x > 0).to(torch.uint8)
    yb = (y > 0).to(torch.uint8)

    xor_dist = (xb.unsqueeze(2) ^ yb.unsqueeze(0).unsqueeze(0)).float()
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()
        xor_dist = xor_dist * mask
        sums = xor_dist.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return sums / counts
    else:
        return xor_dist.mean(dim=1)


def hamming_thresh_quantile(
    x: Tensor,
    y: Tensor,
    attention_mask: Tensor | None = None,
    quantile: float = 0.5
) -> Tensor:

    flat = x.flatten()
    thresh = torch.quantile(flat, quantile)

    xb = (x > thresh).to(torch.uint8)
    yb = (y > thresh).to(torch.uint8)
    xor_dist = (xb.unsqueeze(2) ^ yb.unsqueeze(0).unsqueeze(0)).float()
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()
        xor_dist = xor_dist * mask
        sums = xor_dist.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return sums / counts
    else:
        return xor_dist.mean(dim=1)

def l1_pairwise(
    x: Tensor,    # [M, S, E]
    y: Tensor,    # [M,     E]
    attention_mask: Tensor | None = None,
) -> Tensor:     # [M]
    # exactly what l1_distance does, but only x[i] vs y[i]
    td = torch.abs(x - y.unsqueeze(1)).sum(-1)  # [M, S]
    
    if attention_mask is not None:
        td = td * attention_mask
        counts = attention_mask.sum(dim=1).clamp(min=1)
        return td.sum(dim=1) / counts
    else:
        return td.mean(dim=1)

def hamming_pairwise(
    x: Tensor, y: Tensor, attention_mask: Tensor | None = None, quantile: float|None = None
) -> Tensor:
    # threshold x and y (zero or median)
    if quantile is None:
        xb = (x > 0).to(torch.uint8)
        yb = (y > 0).to(torch.uint8)
    else:
        thresh = x.flatten().quantile(quantile)
        xb = (x > thresh).to(torch.uint8)
        yb = (y > thresh).to(torch.uint8)

    # per‑token xor: [M, S]
    td = (xb.unsqueeze(-1) ^ yb.unsqueeze(1)).float().sum(-1)

    if attention_mask is not None:
        td = td * attention_mask
        counts = attention_mask.sum(dim=1).clamp(min=1)
        return td.sum(dim=1) / counts
    else:
        return td.mean(dim=1)