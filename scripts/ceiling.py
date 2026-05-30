"""
Non-training P@L ceiling analysis for Stage 2 diagnosis.

Runs val loader twice (first pass fits LR on flattened pair-level features,
second pass computes per-sample P@L by range). Feature sources:

    prior_alone       — rank by `prior` contact map
    esm_alone         — rank by `esm_contacts`
    weighted_sum      — α·prior + β·esm_contacts, grid-searched
    lr_minimal        — LR(prior, count)
    lr_plus_esm       — LR(prior, count, esm)
    lr_plus_dist      — LR(prior, count, esm, dist_bins 9ch)
    lr_plus_stats     — LR(..., dist_stats 2ch)
    lr_plus_agree     — LR(..., agreement 1ch)
    lr_all            — LR(everything)

Writes .temp/ceiling_results.tsv with P@L_{short,medium,long} and AUC-PR_long.

Usage:
    python scripts/ceiling.py experiment=diagnostics/ceiling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.utils.metrics import precision_at_k_by_range  # noqa: E402


FEATURE_SPECS: Dict[str, List[str]] = {
    "lr_minimal":   ["prior", "count"],
    "lr_plus_esm":  ["prior", "count", "esm"],
    "lr_plus_dist": ["prior", "count", "esm", "dist_bins"],
    "lr_all":       ["prior", "count", "esm", "dist_bins"],
}

FEATURE_CHANNELS = {
    "prior": 1, "count": 1, "esm": 1,
    "dist_bins": 9,
}


def _get_channel_tensor(batch: Dict, key: str, fallback_shape) -> torch.Tensor:
    """Return (B, C, L, L) tensor for a feature key, zero-filled if missing."""
    mapping = {
        "prior": "prior", "count": "count", "esm": "esm_contacts",
        "dist_bins": "tpl_dist_bins",
    }
    batch_key = mapping[key]
    if batch_key in batch:
        return batch[batch_key]
    # fallback: zero tensor with correct channel count
    B, _, H, W = fallback_shape
    C = FEATURE_CHANNELS[key]
    return torch.zeros((B, C, H, W), dtype=torch.float32)


def _stack_feature_set(batch: Dict, keys: List[str]) -> torch.Tensor:
    """Concatenate feature tensors along channel dim → (B, total_C, L, L)."""
    shape = batch["prior"].shape
    return torch.cat([_get_channel_tensor(batch, k, shape) for k in keys], dim=1)


def _flatten_valid_pairs(
    feat: torch.Tensor,      # (B, C, L, L)
    target: torch.Tensor,    # (B, L, L)
    mask: torch.Tensor,      # (B, L, L)  — long_mask (pair_mask * sep)
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect pair-level (N, C) features and (N,) targets from valid pairs."""
    m = mask.bool()
    # Broadcast mask over channel dim.
    feat_flat = feat.permute(0, 2, 3, 1)[m]           # (N, C)
    y_flat = target[m]                                # (N,)
    return feat_flat.cpu().numpy().astype(np.float32), y_flat.cpu().numpy().astype(np.int64)


def _predict_prob_maps(
    lr: LogisticRegression, feat: torch.Tensor
) -> torch.Tensor:
    """
    Run LR on full (B, C, L, L) tensor → (B, L, L) linear score.
    We skip the sigmoid because P@L only uses rank-ordering.
    """
    if len(lr.classes_) == 1:
        B, _, H, W = feat.shape
        const = float(lr.classes_[0])
        return torch.full((B, H, W), const, dtype=torch.float32)
    w = torch.from_numpy(lr.coef_[0].astype(np.float32))       # (C,)
    b = float(lr.intercept_[0])
    # (B, C, H, W) · (C,) → (B, H, W)
    score = torch.einsum("bchw,c->bhw", feat, w) + b
    return score


def _per_range_metrics(
    probs: torch.Tensor, contact: torch.Tensor, long_mask: torch.Tensor,
    with_auc: bool = False,
) -> Dict[str, float]:
    """P@L by range; AUC-PR is expensive, only compute when explicitly asked."""
    if probs.dim() == 3:
        probs_4d = probs.unsqueeze(1)
    else:
        probs_4d = probs
    ranges = precision_at_k_by_range(probs_4d, contact, long_mask, k_mode="L")
    if with_auc:
        ranges["AUC_PR_long"] = auc_pr_masked(probs_4d, contact, long_mask, range_type="long")
    return ranges


def _run_val_loader(datamodule, max_batches: int):
    """Yield val batches; respects max_batches (−1 = all)."""
    loader = datamodule.val_dataloader()
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def fit_logistic_regressions(
    datamodule,
    max_batches: int,
    max_pairs_per_batch: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, LogisticRegression], Dict[str, float]]:
    """
    Pass 1: accumulate balanced flat features from val batches, fit LR per
    feature set. Returns (dict of fitted LRs, grid-search best {α, β} +
    val-mean weighted sum P@L_long for the search).
    """
    print("=== Pass 1: fitting LRs on val pair-level data ===", flush=True)

    per_fs_X: Dict[str, List[np.ndarray]] = {fs: [] for fs in FEATURE_SPECS}
    per_fs_y: List[np.ndarray] = []

    n_batches = 0
    n_pairs_total = 0
    for i, batch in enumerate(_run_val_loader(datamodule, max_batches)):
        if i == 0:
            # Diagnostic: per-feature stats from the very first batch, so we
            # can tell at a glance whether prior/tpl_* are being populated.
            for key in ("prior", "count", "tpl_dist_bins", "esm_contacts"):
                if key in batch:
                    t = batch[key]
                    print(
                        f"  [batch0] {key} shape={tuple(t.shape)} "
                        f"min={float(t.min()):.4g} max={float(t.max()):.4g} "
                        f"mean={float(t.mean()):.4g} "
                        f"nonzero_frac={float((t != 0).float().mean()):.4g}",
                        flush=True,
                    )
                else:
                    print(f"  [batch0] {key} MISSING", flush=True)
        mask = batch["long_mask"]          # (B, L, L)
        if not mask.any():
            continue
        target = batch["contact"]

        any_feat = _stack_feature_set(batch, FEATURE_SPECS["lr_minimal"])
        _, y = _flatten_valid_pairs(any_feat, target, mask)

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if pos.size == 0 or neg.size == 0:
            continue
        per_class = min(max_pairs_per_batch // 2, pos.size, neg.size)
        pos_idx = rng.choice(pos, size=per_class, replace=False)
        neg_idx = rng.choice(neg, size=per_class, replace=False)
        sel = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(sel)

        per_fs_y.append(y[sel])
        for fs, keys in FEATURE_SPECS.items():
            feat = _stack_feature_set(batch, keys)
            X, _ = _flatten_valid_pairs(feat, target, mask)
            per_fs_X[fs].append(X[sel])
        n_batches += 1
        n_pairs_total += sel.size

    if n_batches == 0:
        raise RuntimeError("No valid batches encountered in val loader.")

    y_all = np.concatenate(per_fs_y)
    print(f"  collected {n_pairs_total} balanced pairs from {n_batches} batches", flush=True)

    lrs: Dict[str, LogisticRegression] = {}
    for fs, chunks in per_fs_X.items():
        X_all = np.concatenate(chunks, axis=0)
        lr = LogisticRegression(
            solver="lbfgs", max_iter=500, C=1.0, n_jobs=1,
        )
        lr.fit(X_all, y_all)
        lrs[fs] = lr
        print(f"  [{fs}] fit on {X_all.shape} — classes={lr.classes_.tolist()}", flush=True)

    return lrs


def weighted_sum_grid_search(
    datamodule, max_batches: int,
) -> Tuple[Tuple[float, float], float]:
    """Pass 1.5: grid search α, β ∈ [0, 3] for sigmoid-free weighted_sum P@L_long."""
    print("=== Grid search α·prior + β·esm for P@L_long ===", flush=True)
    alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    betas  = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    # Collect all batches once to avoid re-iterating the loader per (α, β).
    cached = []
    for batch in _run_val_loader(datamodule, max_batches):
        cached.append({
            "prior": batch["prior"].clone(),
            "esm": batch.get(
                "esm_contacts",
                torch.zeros_like(batch["prior"]),
            ).clone(),
            "contact": batch["contact"].clone(),
            "long_mask": batch["long_mask"].clone(),
        })

    best = (0.0, 0.0)
    best_score = -1.0
    for a in alphas:
        for b in betas:
            if a == 0.0 and b == 0.0:
                continue
            total = 0.0
            n = 0
            for c in cached:
                probs = a * c["prior"].squeeze(1) + b * c["esm"].squeeze(1)
                r = precision_at_k_by_range(
                    probs.unsqueeze(1), c["contact"], c["long_mask"], k_mode="L"
                )
                total += r["long"] * c["contact"].shape[0]
                n += c["contact"].shape[0]
            score = total / max(1, n)
            if score > best_score:
                best_score = score
                best = (a, b)
    print(f"  best α={best[0]}, β={best[1]}, P@L_long={best_score:.4f}", flush=True)
    return best, best_score


def evaluate_feature_sets(
    datamodule,
    lrs: Dict[str, LogisticRegression],
    weighted: Tuple[Tuple[float, float], float],
    max_batches: int,
) -> Dict[str, Dict[str, float]]:
    """Pass 2: compute per-sample range metrics for each method, return means."""
    print("=== Pass 2: per-sample P@L by range ===", flush=True)
    (best_a, best_b), _ = weighted

    # Aggregate per-batch means weighted by batch size.
    per_method_sum: Dict[str, Dict[str, float]] = {}
    per_method_n: Dict[str, int] = {}

    def _add(name: str, metrics: Dict[str, float], n: int):
        if name not in per_method_sum:
            per_method_sum[name] = {k: 0.0 for k in metrics}
            per_method_n[name] = 0
        for k, v in metrics.items():
            per_method_sum[name][k] += v * n
        per_method_n[name] += n

    for batch in _run_val_loader(datamodule, max_batches):
        contact = batch["contact"]
        long_mask = batch["long_mask"]
        B = contact.shape[0]

        # prior_alone
        prior = batch["prior"].squeeze(1)
        _add("prior_alone", _per_range_metrics(prior, contact, long_mask), B)

        # esm_alone
        esm = batch.get("esm_contacts", torch.zeros_like(batch["prior"])).squeeze(1)
        _add("esm_alone", _per_range_metrics(esm, contact, long_mask), B)

        # weighted_sum_best
        ws = best_a * prior + best_b * esm
        _add("weighted_sum_best", _per_range_metrics(ws, contact, long_mask), B)

        # LR feature sets
        for fs, keys in FEATURE_SPECS.items():
            feat = _stack_feature_set(batch, keys)
            probs = _predict_prob_maps(lrs[fs], feat)
            _add(fs, _per_range_metrics(probs, contact, long_mask), B)

    results = {}
    for name, sums in per_method_sum.items():
        n = max(1, per_method_n[name])
        results[name] = {k: v / n for k, v in sums.items()}
    return results


def write_results_tsv(
    results: Dict[str, Dict[str, float]],
    weighted: Tuple[Tuple[float, float], float],
    out_path: Path,
) -> None:
    (a, b), _ = weighted
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["P_L_short", "P_L_medium", "P_L_long"]
    key_map = {"P_L_short": "short", "P_L_medium": "medium", "P_L_long": "long"}
    lines = ["\t".join(["feature_set", *cols])]
    method_order = [
        "prior_alone", "esm_alone", "weighted_sum_best",
        "lr_minimal", "lr_plus_esm", "lr_plus_dist",
        "lr_plus_stats", "lr_plus_agree", "lr_all",
    ]
    for name in method_order:
        if name not in results:
            continue
        row = [name]
        for c in cols:
            v = results[name].get(key_map[c], float("nan"))
            row.append(f"{v:.4f}")
        if name == "weighted_sum_best":
            row[0] = f"weighted_sum_best(a={a},b={b})"
        lines.append("\t".join(row))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}", flush=True)
    print("\n".join(lines), flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ceiling_cfg = cfg.get("ceiling", {}) or {}
    max_batches = int(ceiling_cfg.get("max_batches", -1))
    max_pairs_per_batch = int(ceiling_cfg.get("max_pairs_per_batch", 20000))
    out_path = Path(ceiling_cfg.get("out_path", ".temp/ceiling_results.tsv"))

    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    lrs = fit_logistic_regressions(
        datamodule,
        max_batches=max_batches,
        max_pairs_per_batch=max_pairs_per_batch,
        rng=rng,
    )
    weighted = weighted_sum_grid_search(datamodule, max_batches=max_batches)
    results = evaluate_feature_sets(datamodule, lrs, weighted, max_batches=max_batches)
    write_results_tsv(results, weighted, out_path)


if __name__ == "__main__":
    main()
