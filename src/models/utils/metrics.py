import torch
import numpy as np


def precision_at_k_masked(
    probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, k_mode: str = "L"
) -> float:
    """
    probs:   (B,1,L,L) or (B,L,L)
    targets: (B,L,L) in {0,1}
    mask:    (B,L,L) in {0,1}  (e.g., long_mask: both residues real AND |i-j|>=min_sep)
    """
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    B, L, _ = probs.shape
    precs = []
    for b in range(B):
        # Estimate usable length: number of rows that have any valid pair
        Lb = int(torch.count_nonzero(mask[b].sum(dim=1) > 0).item())
        if Lb == 0:
            continue
        K = Lb if k_mode == "L" else max(1, Lb // 2 if k_mode == "L/2" else Lb // 5)

        m = mask[b].bool().view(-1)
        y = targets[b].float().view(-1)[m]
        p = probs[b].view(-1)[m]
        if p.numel() == 0:
            continue
        K = min(K, p.numel())
        topk = torch.topk(p, K).indices
        precs.append(y[topk].mean().item())
    return float(np.mean(precs)) if precs else 0.0


def precision_at_k_masked_multi(probs, targets, mask, ks=("L", "L/2", "L/5")):
    return {k: precision_at_k_masked(probs, targets, mask, k) for k in ks}
