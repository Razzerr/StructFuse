import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import average_precision_score, matthews_corrcoef

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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

    # Unique pairs only (i<j): the contact map is symmetric, so counting both
    # (i,j) and (j,i) double-counts each physical contact and makes P@K / AUC / F1
    # non-comparable to the standard CASP protocol. Restrict every range to the
    # strict upper triangle.
    upper = j_idx > i_idx

    if max_sep is None:
        return (sep >= min_sep) & upper
    else:
        return (sep >= min_sep) & (sep < max_sep) & upper


def unique_pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """Restrict a (B,L,L) or (L,L) valid-pair mask to the strict upper triangle
    (j>i) so each symmetric contact is counted once. Idempotent on masks already
    built from `_create_range_mask` (which now carries the triangle)."""
    L = mask.shape[-1]
    tri = torch.triu(torch.ones(L, L, device=mask.device, dtype=mask.dtype), diagonal=1)
    return mask * tri


def precision_at_k_masked(
    probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, k_mode: str = "L",
    seq_len=None,
) -> float:
    """
    Precision at K: fraction of the top-K highest-confidence predictions that
    are true contacts. K is derived from the explicitly supplied nominal crop
    length when available; the active-row count is only a monitoring fallback.
    k_mode controls K:

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
    # Candidate pool = unique pairs (i<j) only — never double-count symmetric contacts.
    umask = unique_pair_mask(mask)
    precs = []
    for b in range(B):
        if seq_len is not None:
            # K from the NOMINAL crop length (crop_end-crop_start), passed explicitly.
            # Missing-coordinate residues must NOT shrink L.
            Lb = int(seq_len[b]) if hasattr(seq_len, "__len__") else int(seq_len)
        else:
            # Fallback: number of rows with any valid pair (monitoring paths only).
            Lb = int(mask[b].sum(dim=1).gt(0).sum().item())
        if Lb == 0:
            continue
        K = Lb if k_mode == "L" else max(1, Lb // 2 if k_mode == "L/2" else Lb // 5)

        m = umask[b].bool().view(-1)
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


def range_metrics_at_threshold(tp, fp, fn, tn, val_thresholds, pred_threshold):
    """Pick the threshold-sweep index NEAREST the val-selected `pred_threshold` and
    return f1/precision/recall/MCC there — NOT argmax-on-test (which would tune the
    threshold on the test set). tp/fp/fn/tn are 1-D vectors aligned to val_thresholds.

    Shared by all three Lit modules so their per-range test metrics can't drift.
    """
    vt = torch.as_tensor(val_thresholds, dtype=torch.float64)
    idx = int(torch.argmin(torch.abs(vt - float(pred_threshold))).item())
    tpi, fpi, fni, tni = float(tp[idx]), float(fp[idx]), float(fn[idx]), float(tn[idx])
    precision = tpi / (tpi + fpi + 1e-8)
    recall = tpi / (tpi + fni + 1e-8)
    f1 = 2 * tpi / (2 * tpi + fpi + fni + 1e-8)
    mcc_den = math.sqrt((tpi + fpi) * (tpi + fni) * (tni + fpi) * (tni + fni) + 1e-8)
    mcc = (tpi * tni - fpi * fni) / mcc_den
    return {"idx": idx, "threshold": float(vt[idx]), "f1": f1,
            "precision": precision, "recall": recall, "mcc": mcc}


def _binary_prf(prob_b, contact_b, mask_b, threshold):
    """precision/recall/f1 + (tp,fp,fn,n_pos,n_neg,n_valid) over a single-sample
    (1,L,L) mask at a fixed threshold, restricted to unique pairs (i<j)."""
    m = unique_pair_mask(mask_b).bool()
    p = prob_b[m]
    t = contact_b[m] > 0.5
    pred = p >= threshold
    tp = float((pred & t).sum())
    fp = float((pred & ~t).sum())
    fn = float((~pred & t).sum())
    n_valid = int(m.sum())
    n_pos = int(t.sum())
    n_neg = n_valid - n_pos
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn, n_pos, n_neg, n_valid


def per_sample_metric_rows(prob, contact, mask, pids, subsets, threshold,
                           seq_lens=None, n_templates=None, best_sims=None,
                           cluster_ids=None):
    """Per-chain test metrics → list of row dicts for per_sample_metrics.tsv.

    Single implementation shared by ContactLitModule + the B1/B4 baselines so the
    headline (macro) and the paired-significance analysis use the SAME per-protein
    numbers. All metrics are on unique pairs (i<j); K comes from the nominal crop
    length (`seq_lens[b]`) — NOT active mask rows — so missing coordinates don't
    shrink L. Floats are returned full-precision (NO rounding here). All long-range
    metrics are NaN when a chain has no valid long-range pair; AUC-PR_long is also
    NaN when its targets have no class variation. Template stats are emitted only
    when provided (None for B1/B4).

    Args:
        prob, contact, mask: (B,L,L) or (B,1,L,L) tensors; prob in [0,1].
        pids: list[str] length B ("<pdb>_<chain>").
        subsets: list[str] length B.
        threshold: val-selected decision threshold for f1/precision/recall.
        seq_lens: optional per-chain nominal lengths (crop_end-crop_start).
        n_templates, best_sims: optional per-chain retrieval stats.
        cluster_ids: optional per-chain sequence-cluster ids. Emitted as a
            `cluster_id` column so the headline aggregation and
            `paired_significance.py` can both work over clusters rather than
            chains; -1 marks unknown.
    """
    prob = prob.detach()
    contact = contact.detach()
    mask = mask.detach()
    if prob.dim() == 4:
        prob = prob.squeeze(1)
    if contact.dim() == 4:
        contact = contact.squeeze(1)
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    B = prob.shape[0]
    L = prob.shape[-1]
    long_min = RANGE_LONG[0]
    rows: List[Dict] = []
    for b in range(B):
        p, t, m = prob[b:b + 1], contact[b:b + 1], mask[b:b + 1]
        if seq_lens is not None:
            sl = int(seq_lens[b]) if hasattr(seq_lens, "__len__") else int(seq_lens)
        else:
            sl = int((m[0].sum(dim=1) > 0).sum().item())
        sla = [sl]
        pl_all = float(precision_at_k_masked(p, t, m, "L", seq_len=sla))
        pl2_all = float(precision_at_k_masked(p, t, m, "L/2", seq_len=sla))
        pl5_all = float(precision_at_k_masked(p, t, m, "L/5", seq_len=sla))
        rng_l = precision_at_k_by_range(p, t, m, "L", seq_len=sla)
        rng_l2 = precision_at_k_by_range(p, t, m, "L/2", seq_len=sla)
        rng_l5 = precision_at_k_by_range(p, t, m, "L/5", seq_len=sla)
        auc_long = auc_pr_masked(p, t, m, range_type="long", undefined_value=float("nan"))

        long_mask = m * _create_range_mask(L, long_min, None, p.device).float()
        prec_l, rec_l, f1_l, _tp, _fp, _fn, npos_l, nneg_l, nval_l = _binary_prf(
            p[0], t[0], long_mask[0], threshold)
        prec_a, rec_a, f1_a, _tpa, _fpa, _fna, npos_a, _nneg_a, _nval_a = _binary_prf(
            p[0], t[0], m[0], threshold)
        if nval_l == 0:
            pl_long = float("nan")
            pl2_long = float("nan")
            pl5_long = float("nan")
            prec_l = float("nan")
            rec_l = float("nan")
            f1_l = float("nan")
            auc_long = float("nan")
        else:
            pl_long = float(rng_l.get("long", 0.0))
            pl2_long = float(rng_l2.get("long", 0.0))
            pl5_long = float(rng_l5.get("long", 0.0))

        pid = pids[b]
        pdb_id, chain_id = (pid.split("_", 1) if "_" in pid else (pid, ""))
        row = {
            "sample_id": pid,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "subset": subsets[b] if b < len(subsets) else "all",
            "cluster_id": (
                int(cluster_ids[b]) if cluster_ids is not None and b < len(cluster_ids)
                else -1
            ),
            "seq_len": sl,
            "P@L": pl_all,
            "P@L/2": pl2_all,
            "P@L/5": pl5_all,
            "P@L_short": float(rng_l.get("short", 0.0)),
            "P@L_medium": float(rng_l.get("medium", 0.0)),
            "P@L_long": pl_long,
            "P@L/2_long": pl2_long,
            "P@L/5_long": pl5_long,
            "AUC-PR_long": float(auc_long),
            "f1_long": f1_l,
            "precision_long": prec_l,
            "recall_long": rec_l,
            "f1": f1_a,
            "precision": prec_a,
            "recall": rec_a,
            "n_valid_long_pairs": nval_l,
            "n_pos_long": npos_l,
            "n_neg_long": nneg_l,
            "n_contacts_true": npos_a,
        }
        if n_templates is not None:
            row["n_templates_retrieved"] = int(n_templates[b])
        if best_sims is not None:
            row["best_tpl_sim"] = float(best_sims[b])
        rows.append(row)
    return rows


# TSV column name -> W&B metric key. TSV uses the slash form (P@L/2); W&B keeps
# the historic no-slash convention (P@L2) so dashboards / fetch_wandb_run parse.
_WANDB_KEY = {
    "P@L/2": "P@L2", "P@L/5": "P@L5",
    "P@L/2_long": "P@L2_long", "P@L/5_long": "P@L5_long",
}

# Canonical per-protein metrics logged (as MACRO) under W&B test/* keys.
_HEADLINE_KEYS = (
    "P@L", "P@L/2", "P@L/5", "P@L_short", "P@L_medium", "P@L_long",
    "P@L/2_long", "P@L/5_long", "f1_long", "precision_long", "recall_long",
    "f1", "precision", "recall",
)


_SUBSET_KEYS = (
    "P@L", "P@L/2", "P@L/5", "P@L_short", "P@L_medium", "P@L_long",
    "P@L/2_long", "P@L/5_long", "AUC-PR_long",
    "f1_long", "precision_long", "recall_long",
)


def _finite(rs, key):
    """Finite values of `key` across rows, paired with their cluster id."""
    out = []
    for r in rs:
        v = r.get(key)
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out.append((int(r.get("cluster_id", -1)), v))
    return out


def chain_macro(rs, key):
    """Unweighted mean over chains. Supplementary — see `cluster_macro`."""
    vals = [v for _, v in _finite(rs, key)]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def cluster_macro(rs, key):
    """HEADLINE aggregation: mean within each sequence cluster, then unweighted
    mean over clusters.

    Chains inside a 30%-identity cluster are near-duplicates, so averaging over
    chains weights each protein family by how many times it was deposited and
    inflates the apparent sample size by about two orders of magnitude. Averaging
    over clusters makes the estimand "performance on a randomly drawn family"
    and the number of independent units the cluster count. NaN-aware at both
    levels: a chain with an undefined metric drops out of its cluster mean, and a
    cluster with no defined chain drops out of the outer mean.

    Returns (value, n_clusters). n_clusters is 0 when no row carries a cluster.
    """
    by_cluster: Dict[int, List[float]] = {}
    for cid, v in _finite(rs, key):
        if cid >= 0:
            by_cluster.setdefault(cid, []).append(v)
    if not by_cluster:
        return float("nan"), 0
    means = [float(np.mean(v)) for v in by_cluster.values()]
    return float(np.mean(means)), len(means)


def log_macro_test_metrics(log_fn, rows, subset_names):
    """Log HEADLINE test metrics (+ per-subset) from per-sample rows. `log_fn` is
    a Lightning module's `self.log`. Canonical key names match ContactLitModule so
    every module (frontier + B1/B4) reports identically.

    Canonical `test/*` keys are CLUSTER-BALANCED (Methods 4.11). This CHANGED the
    meaning of existing keys: `test/P@L_long` and friends were per-chain macro
    before 2026-08-21 and are cluster-balanced after it. The old quantity is kept
    alongside as `test/*_chainmacro`, but under a NEW key — so any run fetched
    across that boundary must be compared via `test/cluster_balanced` and the
    audit manifest's `data_version`, never by key name alone.

    W&B keys use the no-slash convention (P@L2) via `_WANDB_KEY`; TSV columns keep
    the slash form. Each canonical key is written exactly once.

    If no row carries a cluster id, the canonical keys fall back to the chain
    macro and `test/cluster_balanced` is logged as 0. That is a protocol error,
    not a supported mode — it means `data.chain_clusters_file` did not reach the
    eval dataset. The run is not lost: `per_sample_metrics.tsv` is still written
    and can be re-aggregated offline against the cluster TSV.
    """
    _, n_clusters = cluster_macro(rows, "P@L_long")
    balanced = n_clusters > 0
    if not balanced:
        log.error(
            "No cluster ids in per-sample rows: headline metrics fall back to the "
            "per-chain macro, which is family-weighted. Check that "
            "data.chain_clusters_file reaches the val/test datasets. Re-aggregate "
            "per_sample_metrics.tsv offline rather than trusting test/* here."
        )
    headline = cluster_macro if balanced else chain_macro

    for key in _HEADLINE_KEYS:
        wandb_key = _WANDB_KEY.get(key, key)
        v, _ = headline(rows, key)
        log_fn(f"test/{wandb_key}", v,
               prog_bar=(key in ("P@L_long", "f1_long")), sync_dist=False)
        cv, _ = chain_macro(rows, key)
        log_fn(f"test/{wandb_key}_chainmacro", cv, prog_bar=False, sync_dist=False)

    av, an = headline(rows, "AUC-PR_long")
    log_fn("test/AUC-PR_long", av, prog_bar=True, sync_dist=False)
    cav, can = chain_macro(rows, "AUC-PR_long")
    log_fn("test/AUC-PR_long_chainmacro", cav, prog_bar=False, sync_dist=False)
    log_fn("test/AUC-PR_long_n_defined", float(an if balanced else can),
           prog_bar=False, sync_dist=False)

    n_long_defined = sum(int(r.get("n_valid_long_pairs", 0)) > 0 for r in rows)
    log_fn("test/n_proteins_long_defined", float(n_long_defined),
           prog_bar=False, sync_dist=False)
    log_fn("test/n_proteins", float(len(rows)), prog_bar=False, sync_dist=False)
    log_fn("test/n_clusters", float(n_clusters), prog_bar=False, sync_dist=False)
    log_fn("test/cluster_balanced", float(balanced), prog_bar=False, sync_dist=False)

    for subset_name in subset_names:
        srows = [r for r in rows if r.get("subset") == subset_name]
        if not srows:
            continue
        log_fn(f"test/{subset_name}/n_samples", float(len(srows)),
               prog_bar=False, sync_dist=False)
        n_long_defined = sum(int(r.get("n_valid_long_pairs", 0)) > 0 for r in srows)
        log_fn(f"test/{subset_name}/n_proteins_long_defined", float(n_long_defined),
               prog_bar=False, sync_dist=False)
        _, sn_clusters = cluster_macro(srows, "P@L_long")
        log_fn(f"test/{subset_name}/n_clusters", float(sn_clusters),
               prog_bar=False, sync_dist=False)
        for key in _SUBSET_KEYS:
            wandb_key = _WANDB_KEY.get(key, key)
            v, _ = headline(srows, key)
            log_fn(f"test/{subset_name}/{wandb_key}", v,
                   prog_bar=(key == "P@L_long"), sync_dist=False)
            cv, _ = chain_macro(srows, key)
            log_fn(f"test/{subset_name}/{wandb_key}_chainmacro", cv,
                   prog_bar=False, sync_dist=False)


def export_per_sample_tsv(rows, trainer):
    """Write per-sample rows to <log_dir>/per_sample_metrics.tsv, matching the
    location + TSV format ContactLitModule uses so paired_significance reads all
    runs identically. No-op when rows is empty."""
    from pathlib import Path
    if not rows:
        return
    try:
        from hydra.core.hydra_config import HydraConfig
        log_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        ld = getattr(trainer, "log_dir", None) if trainer is not None else None
        if ld is not None and not str(ld).startswith(".neptune"):
            log_dir = Path(ld)
        else:
            log_dir = Path("results")
    output_path = log_dir / "per_sample_metrics.tsv"
    headers = list(rows[0].keys())

    def _fmt(v):
        # Lossless: paired_significance.py re-reads this TSV. repr() round-trips
        # floats exactly (NaN -> 'nan'); 4-6-decimal rounding is for paper tables.
        if isinstance(v, float):
            return repr(v)
        return str(v)

    with open(output_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(_fmt(row[h]) for h in headers) + "\n")
    print(f"Per-sample metrics saved to {output_path}", flush=True)


def precision_at_k_by_range(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    k_mode: str = "L",
    seq_len=None,
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

        p_at_k = precision_at_k_masked(probs, targets, combined_mask, k_mode, seq_len=seq_len)
        results[range_name] = p_at_k
    
    return results


def auc_pr_masked(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    range_type: str = "all",
    undefined_value: float = 0.0,
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

    # Unique pairs only (i<j) — symmetric double-count would distort AUC-PR.
    mask = unique_pair_mask(mask)

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
        return undefined_value

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Filter out NaN values (can occur with short sequences / empty ranges)
    finite = np.isfinite(all_probs) & np.isfinite(all_targets)
    all_probs = all_probs[finite]
    all_targets = all_targets[finite]

    if len(all_probs) == 0:
        return undefined_value

    # Need at least one positive and one negative — else AUC-PR is undefined
    if all_targets.sum() == 0 or all_targets.sum() == len(all_targets):
        return undefined_value

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
    mask = unique_pair_mask(mask)  # unique pairs only (i<j)

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
