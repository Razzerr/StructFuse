"""Paired significance test on per-protein metrics from two test runs.

Reads two `per_sample_metrics.tsv` files (one per model: e.g. frontier vs
no_templates baseline), matches chains by `sample_id`, and reports per-metric
per-subset:
  - n (matched chains)
  - mean_delta = mean(treatment - control)
  - 95% bootstrap CI of mean_delta (10k resamples, percentile method)
  - paired Wilcoxon signed-rank p-value (two-sided)

Used in paper Methods/Results to defend headline (frontier_650M vs B2 trained
no-templates) and ablation deltas (every R*/A* run vs frontier_8M baseline).

Usage:
    python scripts/paired_significance.py \\
        --treatment .temp/<run_a>/per_sample_metrics.tsv \\
        --control   .temp/<run_b>/per_sample_metrics.tsv \\
        --out       .temp/audit/<run_a>__vs__<run_b>.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Per-protein metrics worth testing. Keys must match column names produced by
# `_export_per_sample_metrics` in contact_lit_module.py.
METRICS = (
    "P@L",
    "P@L_long",
    "P@L/2_long",
    "P@L/5_long",
    "AUC-PR_long",
    "f1",
    "precision",
    "recall",
)
N_BOOTSTRAP = 10_000


def _bootstrap_ci(delta: np.ndarray, n_resamples: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(delta). Returns (lo, hi)."""
    rng = np.random.default_rng(seed=0)  # deterministic across reruns
    n = len(delta)
    if n < 2:
        return (float("nan"), float("nan"))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _wilcoxon_p(delta: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value. NaN for too-few-pairs / all-zeros."""
    nz = delta[delta != 0]
    if len(nz) < 6:
        return float("nan")
    try:
        _, p = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", correction=False)
        return float(p)
    except ValueError:
        return float("nan")


def _per_subset_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Yield ('whole', full_df) plus one slice per subset value (gold, casp16, ...)."""
    out = {"whole": df}
    if "subset" in df.columns:
        for subset, sub in df.groupby("subset", sort=True):
            out[str(subset)] = sub
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--treatment", required=True, type=Path, help="per_sample_metrics.tsv for the treatment run (e.g. frontier)")
    p.add_argument("--control", required=True, type=Path, help="per_sample_metrics.tsv for the control run (e.g. no_templates)")
    p.add_argument("--out", required=True, type=Path, help="Output TSV path")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help=f"Bootstrap resample count (default {N_BOOTSTRAP})")
    args = p.parse_args()

    a = pd.read_csv(args.treatment, sep="\t")
    b = pd.read_csv(args.control, sep="\t")

    if "sample_id" not in a.columns or "sample_id" not in b.columns:
        raise SystemExit("Both TSVs must have a 'sample_id' column")

    merged = a.merge(b, on="sample_id", suffixes=("_t", "_c"))
    if merged.empty:
        raise SystemExit(f"No overlapping sample_ids between {args.treatment} and {args.control}")

    # Carry subset from treatment side (assumes both files use the same subset assignment).
    if "subset_t" in merged.columns:
        merged["subset"] = merged["subset_t"]

    rows = []
    for subset_name, sub in _per_subset_groups(merged).items():
        for metric in METRICS:
            tcol, ccol = f"{metric}_t", f"{metric}_c"
            if tcol not in sub.columns or ccol not in sub.columns:
                continue
            t_vals = pd.to_numeric(sub[tcol], errors="coerce").to_numpy()
            c_vals = pd.to_numeric(sub[ccol], errors="coerce").to_numpy()
            mask = np.isfinite(t_vals) & np.isfinite(c_vals)
            if mask.sum() < 2:
                continue
            delta = t_vals[mask] - c_vals[mask]
            ci_lo, ci_hi = _bootstrap_ci(delta, args.n_bootstrap)
            p_val = _wilcoxon_p(delta)
            rows.append({
                "subset": subset_name,
                "metric": metric,
                "n": int(mask.sum()),
                "mean_treatment": round(float(t_vals[mask].mean()), 4),
                "mean_control": round(float(c_vals[mask].mean()), 4),
                "mean_delta": round(float(delta.mean()), 4),
                "ci95_lo": round(ci_lo, 4),
                "ci95_hi": round(ci_hi, 4),
                "wilcoxon_p": p_val,
            })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    if not out.empty:
        # Print headline P@L_long row(s) so the operator sees something useful immediately.
        for _, r in out[out["metric"] == "P@L_long"].iterrows():
            sig = "***" if r["wilcoxon_p"] is not None and r["wilcoxon_p"] < 1e-3 else (
                "**" if r["wilcoxon_p"] < 1e-2 else ("*" if r["wilcoxon_p"] < 5e-2 else "ns")
            )
            print(f"  [{r['subset']:>18}] P@L_long: Δ={r['mean_delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] p={r['wilcoxon_p']:.2e} {sig} (n={r['n']})")


if __name__ == "__main__":
    main()
