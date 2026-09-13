"""Paired significance test on per-protein metrics from two test runs.

Reads two `per_sample_metrics.tsv` files (one per model: e.g. frontier vs
no_templates baseline), matches chains by `sample_id`, and reports per-metric
per-subset:
  - n (matched chains), n_clusters (independent units)
  - mean_delta = cluster-balanced mean(treatment - control): the mean delta
    within each sequence cluster, averaged unweighted over clusters
  - mean_delta_chain = the old per-chain mean, kept for comparison
  - 95% CLUSTER bootstrap CI (resamples clusters, not chains, 10k resamples,
    percentile method)
  - paired Wilcoxon signed-rank p-value over per-cluster mean deltas

Chains inside a 30%-identity cluster are near-duplicates, so resampling chains
treats one protein family deposited a thousand times as a thousand independent
observations. On the 2026 test set that inflates the apparent sample size from
6,958 clusters to 165,412 chains and understates every interval by roughly an
order of magnitude. `--bootstrap chain` restores the old behaviour for
reproducing pre-2026 numbers; it is not a valid mode for new claims.

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
from collections.abc import Sequence
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
    # Long-range classification metrics (paper table is F1_long, not all-range f1).
    "f1_long",
    "precision_long",
    "recall_long",
)
N_BOOTSTRAP = 10_000


def _cluster_means(delta: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Mean delta within each sequence cluster — one value per independent unit."""
    _, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    return sums / counts


def _bootstrap_ci(
    delta: np.ndarray,
    n_resamples: int,
    alpha: float = 0.05,
    clusters: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean delta.

    With `clusters`, resamples CLUSTERS with replacement and takes the mean of
    their mean deltas, so the interval reflects the number of independent
    families rather than the number of deposited chains. Without it, falls back
    to the per-chain bootstrap.
    """
    units = _cluster_means(delta, clusters) if clusters is not None else delta
    n = len(units)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=seed)  # deterministic across reruns
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(units[idx].mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _wilcoxon_p(delta: np.ndarray, clusters: np.ndarray | None = None) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value over independent units.

    With `clusters`, the units are per-cluster mean deltas — the same units the
    bootstrap resamples, so the p-value and the interval agree about what an
    observation is. NaN for too-few-pairs / all-zeros.
    """
    units = _cluster_means(delta, clusters) if clusters is not None else delta
    nz = units[units != 0]
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


def _validate_per_sample_frame(df: pd.DataFrame, source: Path | str) -> None:
    if "sample_id" not in df.columns:
        raise ValueError(f"{source} is missing column 'sample_id'")
    duplicated = df["sample_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "sample_id"].astype(str).head(5).tolist()
        raise ValueError(f"{source} contains duplicate sample_id values: {examples}")


def _paired_frame(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    treatment_source: Path | str = "treatment",
    control_source: Path | str = "control",
) -> pd.DataFrame:
    """Strict one-to-one sample pairing with subset-consistency validation."""
    _validate_per_sample_frame(treatment, treatment_source)
    _validate_per_sample_frame(control, control_source)
    merged = treatment.merge(
        control,
        on="sample_id",
        suffixes=("_t", "_c"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"No overlapping sample_ids between {treatment_source} and {control_source}"
        )
    if "subset_t" in merged.columns and "subset_c" in merged.columns:
        left = merged["subset_t"].fillna("").astype(str)
        right = merged["subset_c"].fillna("").astype(str)
        mismatch = left != right
        if mismatch.any():
            examples = merged.loc[
                mismatch, ["sample_id", "subset_t", "subset_c"]
            ].head(5).to_dict("records")
            raise ValueError(f"Subset assignment mismatch between paired runs: {examples}")
        merged["subset"] = merged["subset_t"]
    if "cluster_id_t" in merged.columns and "cluster_id_c" in merged.columns:
        mismatch = merged["cluster_id_t"] != merged["cluster_id_c"]
        if mismatch.any():
            examples = merged.loc[
                mismatch, ["sample_id", "cluster_id_t", "cluster_id_c"]
            ].head(5).to_dict("records")
            raise ValueError(
                f"Cluster assignment mismatch between paired runs: {examples}. "
                "The two runs were evaluated against different chain_clusters.tsv "
                "files, so their clusters are not comparable."
            )
        merged["cluster_id"] = merged["cluster_id_t"]
    return merged


def paired_metric_rows(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    *,
    treatment_source: Path | str = "treatment",
    control_source: Path | str = "control",
    metrics: Sequence[str] = METRICS,
    n_bootstrap: int = N_BOOTSTRAP,
    unit: str = "cluster",
    seed: int = 0,
) -> list[dict[str, object]]:
    """Paired statistics for every (subset, metric) of one treatment/control pair.

    In `unit="cluster"` the point estimates, the bootstrap CI and the Wilcoxon
    input are ALL per-cluster means — one observation per sequence family. The
    `*_chain` columns keep the per-chain quantity beside them so the effect of
    re-weighting is visible rather than asserted.

    This is the one implementation; `main()` and the paper figure generators in
    `paper/results_2_*/generate_artifacts.py` all call it, so a figure CI and a
    table CI cannot drift apart. Values are returned UNROUNDED — rounding is a
    presentation choice for the caller.
    """
    if unit not in ("cluster", "chain"):
        raise ValueError(f"unit must be 'cluster' or 'chain', got {unit!r}")
    merged = _paired_frame(treatment, control, treatment_source, control_source)
    if unit == "cluster" and "cluster_id" not in merged.columns:
        raise ValueError(
            f"Cluster resampling requested but neither {treatment_source} nor "
            f"{control_source} carries a 'cluster_id' column. Re-run the "
            "evaluation with data.chain_clusters_file set, or ask for "
            "unit='chain' explicitly to reproduce a pre-2026 number."
        )

    rows: list[dict[str, object]] = []
    for subset_name, sub in _per_subset_groups(merged).items():
        for metric in metrics:
            tcol, ccol = f"{metric}_t", f"{metric}_c"
            if tcol not in sub.columns or ccol not in sub.columns:
                continue
            t_vals = pd.to_numeric(sub[tcol], errors="coerce").to_numpy(dtype=float)
            c_vals = pd.to_numeric(sub[ccol], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(t_vals) & np.isfinite(c_vals)
            if mask.sum() < 2:
                continue
            delta = t_vals[mask] - c_vals[mask]
            t_kept, c_kept = t_vals[mask], c_vals[mask]

            clusters = None
            if unit == "cluster":
                cid = pd.to_numeric(sub["cluster_id"], errors="coerce").to_numpy()[mask]
                # -1 marks an unknown cluster; such a chain is not an
                # independent unit and must not join a pseudo-cluster of its own.
                known = np.isfinite(cid) & (cid >= 0)
                if known.sum() < 2:
                    continue
                delta, clusters = delta[known], cid[known].astype(int)
                t_kept, c_kept = t_kept[known], c_kept[known]
                units = _cluster_means(delta, clusters)
                mean_delta = float(units.mean())
                mean_t = float(_cluster_means(t_kept, clusters).mean())
                mean_c = float(_cluster_means(c_kept, clusters).mean())
                n_pairs, n_clusters = int(known.sum()), int(len(units))
            else:
                mean_delta = float(delta.mean())
                mean_t = float(t_kept.mean())
                mean_c = float(c_kept.mean())
                n_pairs, n_clusters = int(mask.sum()), -1

            ci_lo, ci_hi = _bootstrap_ci(delta, n_bootstrap, clusters=clusters, seed=seed)
            rows.append({
                "subset": subset_name,
                "metric": metric,
                "n": n_pairs,
                "n_clusters": n_clusters,
                "unit": unit,
                "mean_treatment": mean_t,
                "mean_control": mean_c,
                "mean_delta": mean_delta,
                "mean_treatment_chain": float(t_kept.mean()),
                "mean_control_chain": float(c_kept.mean()),
                # Per-chain means over the SAME retained rows, so the only
                # difference from the cluster columns is the weighting.
                "mean_delta_chain": float(delta.mean()),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "wilcoxon_p": _wilcoxon_p(delta, clusters=clusters),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--treatment", required=True, type=Path, help="per_sample_metrics.tsv for the treatment run (e.g. frontier)")
    p.add_argument("--control", required=True, type=Path, help="per_sample_metrics.tsv for the control run (e.g. no_templates)")
    p.add_argument("--out", required=True, type=Path, help="Output TSV path")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help=f"Bootstrap resample count (default {N_BOOTSTRAP})")
    p.add_argument(
        "--bootstrap", choices=("cluster", "chain"), default="cluster",
        help="Resampling unit. 'cluster' (default) treats one sequence cluster "
             "as one observation, matching the headline aggregation. 'chain' "
             "reproduces pre-2026 numbers and understates every interval; it is "
             "not valid for new claims.",
    )
    args = p.parse_args()

    a = pd.read_csv(args.treatment, sep="\t")
    b = pd.read_csv(args.control, sep="\t")

    try:
        rows = paired_metric_rows(
            a,
            b,
            treatment_source=args.treatment,
            control_source=args.control,
            n_bootstrap=args.n_bootstrap,
            unit=args.bootstrap,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Rounding is presentation only; `paired_metric_rows` returns full precision.
    for row in rows:
        for key in ("mean_treatment", "mean_control", "mean_delta",
                    "mean_treatment_chain", "mean_control_chain",
                    "mean_delta_chain", "ci95_lo", "ci95_hi"):
            row[key] = round(float(row[key]), 4)

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
            unit = f"{r['n_clusters']} clusters / {r['n']} chains" if r["n_clusters"] >= 0 else f"{r['n']} chains"
            print(f"  [{r['subset']:>18}] P@L_long: Δ={r['mean_delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] p={r['wilcoxon_p']:.2e} {sig} ({unit})")


if __name__ == "__main__":
    main()
