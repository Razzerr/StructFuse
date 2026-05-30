"""
Feature correlation / redundancy analysis for Stage 2 diagnosis.

For each scalar view of the Stage 2 template features, quantify:
    - Pearson correlation with prior
    - MI with ground-truth contact
    - Conditional MI with contact GIVEN (prior, count)

Cond-MI < ~0.01 bit → feature redundant over what prior+count already provide.
Cond-MI > ~0.05 bit → feature has unique signal; poor training results point
at architecture, not information bottleneck.

Usage:
    python scripts/feature_correlation.py experiment=diagnostics/ceiling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.feature_selection import mutual_info_classif

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# Scalar views of each feature (everything reduced to 1D per pair so MI is tractable).
SCALAR_VIEWS: Dict[str, str] = {
    "prior":            "prior",
    "count":            "count",
    "esm":              "esm_contacts",
    "dist_argmax":      "tpl_dist_bins",       # argmax over 9 bins → ordinal
}


def _extract_pair_scalars(batch: Dict) -> Dict[str, np.ndarray]:
    """Return dict of (N_valid,) arrays, where N_valid = number of pairs in long_mask."""
    mask = batch["long_mask"].bool()
    target = batch["contact"][mask].cpu().numpy().astype(np.int64)
    out: Dict[str, np.ndarray] = {"_y": target}

    def _squeeze_mask(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(1)[mask].cpu().numpy().astype(np.float32)

    if "prior" in batch:
        out["prior"] = _squeeze_mask(batch["prior"])
    if "count" in batch:
        out["count"] = _squeeze_mask(batch["count"])
    if "esm_contacts" in batch:
        out["esm"] = _squeeze_mask(batch["esm_contacts"])
    if "tpl_dist_bins" in batch:
        # (B, 9, L, L) → argmax ordinal bin id.
        bins = batch["tpl_dist_bins"].argmax(dim=1)   # (B, L, L)
        out["dist_argmax"] = bins[mask].cpu().numpy().astype(np.float32)

    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _discretize(x: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Quantile discretization → ordinal bin id in [0, n_bins)."""
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)  # collapse degenerate edges
    if qs.size < 2:
        return np.zeros_like(x, dtype=np.int64)
    idx = np.clip(np.searchsorted(qs[1:-1], x, side="right"), 0, qs.size - 2)
    return idx.astype(np.int64)


def _entropy_from_counts(counts: np.ndarray, base: float) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * (np.log(p) / np.log(base))).sum())


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    """MI(x; y) in bits, both x/y discrete non-negative ints."""
    nx = int(x.max()) + 1
    ny = int(y.max()) + 1
    joint = np.zeros((nx, ny), dtype=np.int64)
    np.add.at(joint, (x, y), 1)
    total = joint.sum()
    if total == 0:
        return 0.0
    p_xy = joint / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    mask = p_xy > 0
    denom = p_x * p_y
    denom = np.where(denom > 0, denom, 1.0)
    log_term = np.log(p_xy / denom, where=mask, out=np.zeros_like(p_xy))
    mi = float((p_xy[mask] * log_term[mask]).sum() / np.log(base))
    return max(0.0, mi)


def _conditional_mutual_info(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, base: float = 2.0
) -> float:
    """
    I(x; y | z) = Σ_z p(z) I(x; y | Z=z).
    All inputs must be discrete non-negative ints. Small-sample bias corrected by
    dropping strata with <50 samples.
    """
    total = 0.0
    for z_val in np.unique(z):
        mask = z == z_val
        n = int(mask.sum())
        if n < 50:
            continue
        p_z = n / z.size
        mi_cond = _mutual_info_discrete(x[mask], y[mask], base=base)
        total += p_z * mi_cond
    return max(0.0, total)


def _run_val_loader(datamodule, max_batches: int):
    loader = datamodule.val_dataloader()
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def collect_pairs(
    datamodule, max_batches: int, max_pairs: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    print(f"=== Collecting pair-level features (max_pairs={max_pairs}) ===", flush=True)
    chunks: Dict[str, List[np.ndarray]] = {}
    total = 0
    batch_keys_seen: List[str] = []
    for i, batch in enumerate(_run_val_loader(datamodule, max_batches)):
        if i == 0:
            batch_keys_seen = sorted(batch.keys())
            print(f"  first batch keys: {batch_keys_seen}", flush=True)
            # Quick per-feature stats from the raw batch tensors.
            for key in ("prior", "count", "tpl_dist_bins", "esm_contacts"):
                if key in batch:
                    t = batch[key]
                    print(
                        f"  stats[{key}] shape={tuple(t.shape)} "
                        f"min={float(t.min()):.4g} max={float(t.max()):.4g} "
                        f"mean={float(t.mean()):.4g} nonzero_frac={float((t!=0).float().mean()):.4g}",
                        flush=True,
                    )
                else:
                    print(f"  stats[{key}] MISSING from batch", flush=True)
        d = _extract_pair_scalars(batch)
        n = d["_y"].size
        if n == 0:
            continue
        if total + n > max_pairs:
            take = max(0, max_pairs - total)
            if take == 0:
                break
            idx = rng.choice(n, size=take, replace=False)
            for k, v in d.items():
                chunks.setdefault(k, []).append(v[idx])
            total += take
            break
        else:
            for k, v in d.items():
                chunks.setdefault(k, []).append(v)
            total += n
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    print(f"  collected {total} valid pairs", flush=True)
    # Per-feature pair-level stats after masking (what MI will see).
    print("=== Pair-level feature stats (after long_mask) ===", flush=True)
    for name in ("prior", "count", "esm", "dist_argmax", "dist_mean", "dist_std", "agreement"):
        if name not in out:
            print(f"  {name}: MISSING", flush=True)
            continue
        x = out[name]
        print(
            f"  {name}: n={x.size} min={x.min():.4g} max={x.max():.4g} "
            f"mean={x.mean():.4g} std={x.std():.4g} nonzero_frac={(x != 0).mean():.4g}",
            flush=True,
        )
    return out


def compute_stats(pairs: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    y = pairs["_y"]  # already 0/1
    prior = pairs.get("prior")
    count = pairs.get("count")

    # Z = (prior_bin, count_bin) joint stratum id for conditional MI.
    if prior is not None and count is not None:
        prior_bin = _discretize(prior, n_bins=8)
        count_bin = _discretize(count, n_bins=8)
        z_joint = prior_bin * 16 + count_bin
    else:
        z_joint = None

    rows: List[Dict[str, float]] = []
    for name, src_key in SCALAR_VIEWS.items():
        if name not in pairs:
            continue
        x = pairs[name]
        r = {
            "feature": name,
            "source": src_key,
            "Pearson_w_prior": _pearson(x, prior) if prior is not None else float("nan"),
            "MI_w_contact_bits": float("nan"),
            "CondMI_given_prior_count_bits": float("nan"),
        }

        # MI(x; y) via sklearn (continuous x, discrete y).
        try:
            mi = mutual_info_classif(
                x.reshape(-1, 1),
                y,
                discrete_features=False,
                n_neighbors=3,
                random_state=42,
            )
            r["MI_w_contact_bits"] = float(mi[0] / np.log(2))  # nats → bits
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] MI failed for {name}: {e}", flush=True)

        # Conditional MI: discretize x and compute hist-based estimator.
        if z_joint is not None:
            try:
                x_bin = _discretize(x, n_bins=16)
                r["CondMI_given_prior_count_bits"] = _conditional_mutual_info(
                    x_bin, y, z_joint, base=2.0
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] CondMI failed for {name}: {e}", flush=True)

        rows.append(r)
    return rows


def write_tsv(rows: List[Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["feature", "source", "Pearson_w_prior", "MI_w_contact_bits", "CondMI_given_prior_count_bits"]
    lines = ["\t".join(cols)]
    for r in rows:
        row = []
        for c in cols:
            v = r.get(c, "")
            row.append(v if isinstance(v, str) else f"{v:.4f}")
        lines.append("\t".join(row))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}", flush=True)
    print("\n".join(lines), flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    corr_cfg = cfg.get("correlation", {}) or {}
    max_batches = int(corr_cfg.get("max_batches", 200))
    max_pairs = int(corr_cfg.get("max_pairs", 1_000_000))
    out_path = Path(corr_cfg.get("out_path", ".temp/feature_correlation.tsv"))

    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    pairs = collect_pairs(datamodule, max_batches, max_pairs, rng)
    rows = compute_stats(pairs)
    write_tsv(rows, out_path)


if __name__ == "__main__":
    main()
