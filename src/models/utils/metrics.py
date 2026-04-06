import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import average_precision_score, matthews_corrcoef


# Range definitions for contact prediction
RANGE_SHORT = (6, 12)    # Short-range: 6 <= |i-j| < 12
RANGE_MEDIUM = (12, 24)  # Medium-range: 12 <= |i-j| < 24
RANGE_LONG = (24, None)  # Long-range: |i-j| >= 24


def _create_range_mask(L: int, min_sep: int, max_sep: Optional[int], device: torch.device) -> torch.Tensor:
    """
    Create a mask for pairs within a specific sequence separation range.
    
    Args:
        L: Sequence length
        min_sep: Minimum separation (inclusive)
        max_sep: Maximum separation (exclusive), None for no upper limit
        device: Torch device
        
    Returns:
        (L, L) boolean mask
    """
    i_idx = torch.arange(L, device=device).unsqueeze(1)
    j_idx = torch.arange(L, device=device).unsqueeze(0)
    sep = torch.abs(i_idx - j_idx)
    
    if max_sep is None:
        return sep >= min_sep
    else:
        return (sep >= min_sep) & (sep < max_sep)


def precision_at_k_masked(
    probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, k_mode: str = "L"
) -> float:
    """
    Precision at K: fraction of the top-K highest-confidence predictions that
    are true contacts.  K is derived from each sample's usable sequence length
    (number of rows with ≥1 valid pair).  k_mode controls K:

        "L"   → K = Lb  (one prediction per residue)
        "L/2" → K = Lb // 2
        "L/5" → K = Lb // 5

    This is the standard contact-prediction metric (Schaarschmidt et al., 2018).

    Args:
        probs:   (B,1,L,L) or (B,L,L)  prediction probabilities
        targets: (B,L,L)               ground truth contacts {0,1}
        mask:    (B,L,L)               valid pair mask (long_mask * pair_mask)
        k_mode:  one of "L", "L/2", "L/5"

    Returns:
        float: mean P@K across the batch
    """
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    B, L, _ = probs.shape
    precs = []
    for b in range(B):
        # Estimate usable length: number of rows that have any valid pair
        Lb = int(mask[b].sum(dim=1).gt(0).sum().item())
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
        precs.append(y[topk].mean())
    if not precs:
        return 0.0
    return float(torch.stack(precs).mean().item())


def precision_at_k_masked_multi(probs, targets, mask, ks=("L", "L/2", "L/5")):
    return {k: precision_at_k_masked(probs, targets, mask, k) for k in ks}


def precision_at_k_by_range(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    k_mode: str = "L",
) -> Dict[str, float]:
    """
    Compute P@K separately for short, medium, and long-range contacts.
    
    Args:
        probs: (B, 1, L, L) or (B, L, L) prediction probabilities
        targets: (B, L, L) ground truth contacts
        mask: (B, L, L) valid pair mask (already includes min_sep filtering)
        k_mode: "L", "L/2", or "L/5"
        
    Returns:
        Dict with keys "short", "medium", "long" and P@K values
    """
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    
    B, L, _ = probs.shape
    device = probs.device
    
    results = {}
    ranges = {
        "short": RANGE_SHORT,
        "medium": RANGE_MEDIUM,
        "long": RANGE_LONG,
    }
    
    for range_name, (min_sep, max_sep) in ranges.items():
        range_mask = _create_range_mask(L, min_sep, max_sep, device)
        combined_mask = mask * range_mask.float()
        
        p_at_k = precision_at_k_masked(probs, targets, combined_mask, k_mode)
        results[range_name] = p_at_k
    
    return results


def auc_pr_masked(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    range_type: str = "all",
) -> float:
    """
    Compute Area Under Precision-Recall curve (Average Precision).
    
    Args:
        probs: (B, 1, L, L) or (B, L, L) prediction probabilities
        targets: (B, L, L) ground truth contacts
        mask: (B, L, L) valid pair mask
        range_type: "all", "short", "medium", or "long"
        
    Returns:
        Average Precision score
    """
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    
    B, L, _ = probs.shape
    device = probs.device
    
    # Apply range mask if specified
    if range_type != "all":
        ranges = {
            "short": RANGE_SHORT,
            "medium": RANGE_MEDIUM,
            "long": RANGE_LONG,
        }
        min_sep, max_sep = ranges[range_type]
        range_mask = _create_range_mask(L, min_sep, max_sep, device)
        mask = mask * range_mask.float()
    
    # Flatten and collect valid pairs
    all_probs = []
    all_targets = []
    
    for b in range(B):
        m = mask[b].bool().view(-1)
        p = probs[b].view(-1)[m].cpu().to(torch.float32).numpy()
        t = targets[b].view(-1)[m].cpu().to(torch.float32).numpy()
        
        if len(p) > 0:
            all_probs.append(p)
            all_targets.append(t)
    
    if not all_probs:
        return 0.0
    
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Filter out NaN values (can occur with short sequences / empty ranges)
    finite = np.isfinite(all_probs) & np.isfinite(all_targets)
    all_probs = all_probs[finite]
    all_targets = all_targets[finite]
    
    if len(all_probs) == 0:
        return 0.0
    
    # Need at least one positive and one negative
    if all_targets.sum() == 0 or all_targets.sum() == len(all_targets):
        return 0.0
    
    return float(average_precision_score(all_targets, all_probs))


def mcc_at_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Matthews Correlation Coefficient at a fixed threshold.
    
    Args:
        probs: (B, 1, L, L) or (B, L, L) prediction probabilities
        targets: (B, L, L) ground truth contacts
        mask: (B, L, L) valid pair mask
        threshold: Classification threshold (should be determined on validation set)
        
    Returns:
        MCC score in range [-1, 1]
    """
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    
    B, L, _ = probs.shape
    
    # Flatten and collect valid pairs
    all_preds = []
    all_targets = []
    
    for b in range(B):
        m = mask[b].bool().view(-1)
        p = (probs[b].view(-1)[m] >= threshold).cpu().to(torch.float32).numpy().astype(int)
        t = targets[b].view(-1)[m].cpu().to(torch.float32).numpy().astype(int)
        
        if len(p) > 0:
            all_preds.append(p)
            all_targets.append(t)
    
    if not all_preds:
        return 0.0
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Need variation in both predictions and targets
    if len(np.unique(all_preds)) < 2 or len(np.unique(all_targets)) < 2:
        return 0.0
    
    return float(matthews_corrcoef(all_targets, all_preds))


def find_optimal_threshold_for_mcc(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """
    Find threshold that maximizes MCC on validation data.
    
    Args:
        probs: (B, 1, L, L) or (B, L, L) prediction probabilities
        targets: (B, L, L) ground truth contacts
        mask: (B, L, L) valid pair mask
        thresholds: Optional tensor of thresholds to try
        
    Returns:
        Tuple of (best_threshold, best_mcc)
    """
    if thresholds is None:
        thresholds = torch.linspace(0.1, 0.9, 17)
    
    best_mcc = -2.0
    best_threshold = 0.5
    
    for tau in thresholds:
        mcc = mcc_at_threshold(probs, targets, mask, float(tau))
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = float(tau)
    
    return best_threshold, best_mcc


def compute_all_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for contact prediction.
    
    Args:
        probs: (B, 1, L, L) or (B, L, L) prediction probabilities
        targets: (B, L, L) ground truth contacts
        mask: (B, L, L) valid pair mask
        threshold: Classification threshold for binary metrics
        
    Returns:
        Dict with all metrics
    """
    results = {}
    
    # P@L metrics
    for k in ["L", "L/2", "L/5"]:
        results[f"P@{k}"] = precision_at_k_masked(probs, targets, mask, k)
    
    # P@L by range (long-range is headline metric)
    range_metrics = precision_at_k_by_range(probs, targets, mask, k_mode="L")
    for range_name, value in range_metrics.items():
        results[f"P@L_{range_name}"] = value
    
    # AUC-PR (all and long-range)
    results["AUC-PR"] = auc_pr_masked(probs, targets, mask, range_type="all")
    results["AUC-PR_long"] = auc_pr_masked(probs, targets, mask, range_type="long")
    
    # MCC at threshold
    results["MCC"] = mcc_at_threshold(probs, targets, mask, threshold)
    
    return results
