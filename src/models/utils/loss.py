import torch
import torch.nn.functional as F
from typing import Optional

def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))

def masked_bce_balanced(
    logits: torch.Tensor,          # (B,1,L,L) or (B,L,L)
    targets: torch.Tensor,         # (B,L,L) in {0,1}
    mask: torch.Tensor,            # (B,L,L) in {0,1}
    *,
    pos_weight_scale: float = 1.0,
    label_smoothing: float = 0.0,
    enforce_symmetry: bool = True,
) -> torch.Tensor:
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    y = targets.float()
    m = mask.float()

    # (optional) enforce pair symmetry on predictions before loss
    if enforce_symmetry:
        logits = symmetrize(logits)

    # label smoothing (pushes targets away from 0/1 a bit)
    if label_smoothing > 0.0:
        eps = label_smoothing
        y = y * (1.0 - eps) + 0.5 * eps

    # class balance within mask
    pos = (targets == 1).float() * m
    neg = (targets == 0).float() * m
    n_pos = pos.sum().clamp_min(1.0)
    n_neg = neg.sum().clamp_min(1.0)
    pos_weight = (n_neg / n_pos) * pos_weight_scale
    pos_weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)

    # base BCE-with-logits per element
    loss = F.binary_cross_entropy_with_logits(
        logits, y, weight=m, pos_weight=pos_weight, reduction="none"
    )

    denom = m.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_focal_tversky(
    logits: torch.Tensor,          # (B,1,L,L) or (B,L,L)
    targets: torch.Tensor,         # (B,L,L) in {0,1}
    mask: torch.Tensor,            # (B,L,L) in {0,1}
    *,
    alpha: float = 0.7,            # Weight for false positives
    beta: float = 0.3,             # Weight for false negatives (beta < alpha = recall-biased)
    gamma: float = 1.0,            # Focal modulation strength (1.0 = standard focal)
    enforce_symmetry: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Masked Focal Tversky Loss for imbalanced segmentation tasks.
    
    The Tversky Index is a generalization of Dice/F1 that allows asymmetric 
    weighting of false positives and false negatives:
    
        TI = TP / (TP + α*FP + β*FN)
    
    When α=β=0.5, it reduces to Dice/F1.
    When α < β, it penalizes false negatives more (recall-biased).
    When α > β, it penalizes false positives more (precision-biased).
    
    Focal modulation: (1 - TI)^gamma makes it focus on hard examples.
    
    Args:
        logits: Raw model predictions (before sigmoid)
        targets: Ground truth binary labels {0, 1}
        mask: Valid region mask {0, 1}
        alpha: Weight for false positives (default 0.7)
        beta: Weight for false negatives (default 0.3, recall-biased)
        gamma: Focal modulation strength (default 1.0)
        enforce_symmetry: Whether to symmetrize logits before computing loss
        eps: Numerical stability constant
        
    Returns:
        Scalar loss value
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    
    # Optional symmetry enforcement
    if enforce_symmetry:
        logits = symmetrize(logits)
    
    # Get probabilities
    probs = torch.sigmoid(logits)
    
    # Flatten and apply mask
    m = mask.float()
    p_flat = probs[m > 0]      # Predicted probabilities for valid pairs
    t_flat = targets[m > 0].float()  # Ground truth for valid pairs
    
    # Compute TP, FP, FN
    tp = (p_flat * t_flat).sum()
    fp = (p_flat * (1 - t_flat)).sum()
    fn = ((1 - p_flat) * t_flat).sum()
    
    # Tversky Index
    tversky_index = tp / (tp + alpha * fp + beta * fn + eps)
    
    # Focal modulation: penalize easy examples less
    focal_tversky = (1 - tversky_index).pow(gamma)
    
    return focal_tversky
