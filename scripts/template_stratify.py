"""Stratify paired per-protein gains by reference template quality.

For model comparisons, chains are first matched by ``sample_id``. Bins are
defined only from the reference (StructFuse) run's ``best_tpl_sim`` or
``n_templates_retrieved``. Per-chain deltas are then aggregated inside those
reference bins. This avoids the invalid comparison where a no-template control
is assigned to bins using its own zero-valued retrieval metadata.

Aggregation unit (2026-09-11): ``--bootstrap cluster`` (default) averages each
bin per sequence cluster first (chains of one cluster that fall in the bin),
then unweighted over clusters, and bootstraps clusters. This is the same
estimator as ``paired_significance.py`` and the headline metrics; per-chain
means are kept beside it as ``*_chain`` columns. ``--bootstrap chain`` restores
the pre-2026 per-chain behaviour and is not valid for new claims. The bin is a
per-chain attribute of the reference run, so one cluster may contribute to
several bins.

Usage:
    python scripts/template_stratify.py \
        --inputs frontier=frontier.tsv no_templates=no_templates.tsv \
        --reference frontier \
        --out .temp/audit/frontier_vs_notpl_strat.tsv

Single-input descriptive mode:
    python scripts/template_stratify.py \
        --inputs frontier=frontier.tsv \
        --out .temp/audit/frontier_strat.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SIM_BIN_PRESETS = {
    "broad": (
        [-1e-9, 0.3, 0.5, 0.7, 1.0 + 1e-6],
        ["sim<0.3", "0.3-0.5", "0.5-0.7", "sim>0.7"],
    ),
    # The paper runs retrieve very close neighbours, so broad bins collapse
    # almost every test protein into sim>0.7. These bins resolve the high-sim
    # regime without using data-dependent quantiles.
    "high": (
        [-1e-9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0 + 1e-6],
        ["sim<0.95", "0.95-0.98", "0.98-0.99", "0.99-0.995", "0.995-0.999", "sim>=0.999"],
    ),
}
N_TPL_BINS = [-1, 0, 1, 4, 8, 16, 1_000_000]
N_TPL_LABELS = ["k=0", "k=1", "k=2-4", "k=5-8", "k=9-16", "k>16"]
DEFAULT_METRICS = (
    "P@L",
    "P@L_long",
    "P@L/2_long",
    "P@L/5_long",
    "AUC-PR_long",
    "f1_long",
)


def _parse_inputs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--inputs entries must be name=path, got {spec!r}")
        label, path_s = spec.split("=", 1)
        if not label or not path_s:
            raise SystemExit(f"Invalid --inputs entry: {spec!r}")
        if label in labels:
            raise SystemExit(f"Duplicate input label: {label!r}")
        labels.add(label)
        parsed.append((label, Path(path_s)))
    return parsed


def _bootstrap_ci(delta: np.ndarray, n_resamples: int) -> tuple[float, float]:
    """Percentile CI of the mean of ``delta`` — one entry per resampling unit."""
    if n_resamples <= 0 or len(delta) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=0)
    means = np.empty(n_resamples, dtype=np.float64)
    n = len(delta)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _require_cluster_ids(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Cluster mode refuses rather than degrades: no ``cluster_id`` or an unknown
    (-1) cluster cannot be pooled into a pseudo-cluster."""
    if "cluster_id" not in df.columns:
        raise SystemExit(
            f"{path} has no 'cluster_id' column; pre-2026-08-21 TSV. "
            "Use --bootstrap chain to reproduce the old per-chain numbers."
        )
    cid = pd.to_numeric(df["cluster_id"], errors="coerce")
    bad = ~np.isfinite(cid) | (cid < 0)
    if bad.any():
        print(f"[template_stratify] {path}: dropping {int(bad.sum())} rows with unknown cluster_id")
    out = df.loc[~bad].copy()
    out["cluster_id"] = cid[~bad].astype(int)
    return out


def _cluster_means(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-cluster means of ``cols`` over the chains of ``group`` (one bin)."""
    return group.groupby("cluster_id", sort=False)[cols].mean()


def _load(
    path: Path,
    *,
    require_retrieval: bool,
    sim_bins: list[float],
    sim_labels: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "sample_id" not in df.columns:
        raise SystemExit(f"{path} is missing column 'sample_id'")
    duplicated = df["sample_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "sample_id"].astype(str).head(5).tolist()
        raise SystemExit(f"{path} contains duplicate sample_id values: {examples}")
    if require_retrieval:
        missing = [
            col for col in ("best_tpl_sim", "n_templates_retrieved")
            if col not in df.columns
        ]
        if missing:
            raise SystemExit(
                f"{path} is missing reference retrieval columns: {', '.join(missing)}"
            )
        df = df.copy()
        df["sim_bin"] = pd.cut(
            pd.to_numeric(df["best_tpl_sim"], errors="coerce"),
            bins=sim_bins,
            labels=sim_labels,
            include_lowest=True,
        )
        df["k_bin"] = pd.cut(
            pd.to_numeric(df["n_templates_retrieved"], errors="coerce"),
            bins=N_TPL_BINS,
            labels=N_TPL_LABELS,
        )
    return df


def _aggregate_single(
    df: pd.DataFrame,
    by: str,
    label: str,
    metrics: tuple[str, ...],
    unit: str,
) -> pd.DataFrame:
    rows = []
    for bin_val, group in df.groupby(by, observed=True, sort=False):
        row = {
            "stratifier": by,
            "bin": str(bin_val),
            "model": label,
            "unit": unit,
            "n_chains": int(len(group)),
        }
        if unit == "cluster":
            row["n_clusters"] = int(group["cluster_id"].nunique())
        for metric in metrics:
            if metric not in group.columns:
                continue
            g = group[[c for c in ("cluster_id",) if c in group.columns] + [metric]].copy()
            g[metric] = pd.to_numeric(g[metric], errors="coerce")
            g = g[np.isfinite(g[metric])]
            chain_vals = g[metric].to_numpy(dtype=float)
            row[f"n_{metric}"] = int(len(chain_vals))
            row[f"mean_{metric}_chain"] = float(chain_vals.mean()) if len(chain_vals) else float("nan")
            if unit == "cluster":
                cm = _cluster_means(g, [metric])[metric].to_numpy(dtype=float)
                row[f"n_clusters_{metric}"] = int(len(cm))
                row[f"mean_{metric}"] = float(cm.mean()) if len(cm) else float("nan")
                row[f"std_{metric}"] = float(cm.std(ddof=0)) if len(cm) else float("nan")
            else:
                row[f"mean_{metric}"] = row[f"mean_{metric}_chain"]
                row[f"std_{metric}"] = float(chain_vals.std(ddof=0)) if len(chain_vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_paired(
    merged: pd.DataFrame,
    by: str,
    reference_label: str,
    control_label: str,
    metrics: tuple[str, ...],
    n_bootstrap: int,
    unit: str,
) -> pd.DataFrame:
    rows = []
    for bin_val, group in merged.groupby(by, observed=True, sort=False):
        row = {
            "stratifier": by,
            "bin": str(bin_val),
            "reference": reference_label,
            "control": control_label,
            "unit": unit,
            "n_pairs": int(len(group)),
        }
        if unit == "cluster":
            row["n_clusters"] = int(group["cluster_id"].nunique())
        for metric in metrics:
            ref_col = f"{metric}__reference"
            ctrl_col = f"{metric}__control"
            if ref_col not in group.columns or ctrl_col not in group.columns:
                continue
            g = group[[c for c in ("cluster_id",) if c in group.columns] + [ref_col, ctrl_col]].copy()
            g[ref_col] = pd.to_numeric(g[ref_col], errors="coerce")
            g[ctrl_col] = pd.to_numeric(g[ctrl_col], errors="coerce")
            g = g[np.isfinite(g[ref_col]) & np.isfinite(g[ctrl_col])]
            g["delta"] = g[ref_col] - g[ctrl_col]
            chain_delta = g["delta"].to_numpy(dtype=float)
            row[f"n_{metric}"] = int(len(chain_delta))
            row[f"mean_delta_{metric}_chain"] = (
                float(chain_delta.mean()) if len(chain_delta) else float("nan")
            )
            if unit == "cluster":
                cm = _cluster_means(g, [ref_col, ctrl_col, "delta"])
                ref = cm[ref_col].to_numpy(dtype=float)
                ctrl = cm[ctrl_col].to_numpy(dtype=float)
                delta = cm["delta"].to_numpy(dtype=float)
                row[f"n_clusters_{metric}"] = int(len(delta))
            else:
                ref = g[ref_col].to_numpy(dtype=float)
                ctrl = g[ctrl_col].to_numpy(dtype=float)
                delta = chain_delta
            ci_lo, ci_hi = _bootstrap_ci(delta, n_bootstrap)
            row[f"mean_{metric}__reference"] = (
                float(ref.mean()) if len(ref) else float("nan")
            )
            row[f"mean_{metric}__control"] = (
                float(ctrl.mean()) if len(ctrl) else float("nan")
            )
            row[f"mean_delta_{metric}"] = (
                float(delta.mean()) if len(delta) else float("nan")
            )
            row[f"ci95_lo_delta_{metric}"] = ci_lo
            row[f"ci95_hi_delta_{metric}"] = ci_hi
            row[f"std_delta_{metric}"] = (
                float(delta.std(ddof=0)) if len(delta) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_frame(
    reference: pd.DataFrame,
    control: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    reference_columns = [
        "sample_id",
        *(["cluster_id"] if "cluster_id" in reference.columns else []),
        "sim_bin",
        "k_bin",
        *[metric for metric in metrics if metric in reference.columns],
    ]
    control_columns = [
        "sample_id",
        *[metric for metric in metrics if metric in control.columns],
    ]
    return reference[reference_columns].merge(
        control[control_columns],
        on="sample_id",
        how="inner",
        suffixes=("__reference", "__control"),
        validate="one_to_one",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. frontier=frontier.tsv no_templates=control.tsv",
    )
    parser.add_argument(
        "--reference",
        help="Reference input label. Defaults to the first --inputs entry.",
    )
    parser.add_argument(
        "--sim-bin-preset",
        choices=sorted(SIM_BIN_PRESETS),
        default="broad",
        help="Similarity-bin preset. Use 'high' for paper runs with near-1.0 templates.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to aggregate.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for per-bin mean-delta CIs. Default 0 disables CIs.",
    )
    parser.add_argument(
        "--bootstrap",
        choices=("cluster", "chain"),
        default="cluster",
        help="Aggregation and resampling unit. 'cluster' (default) averages each bin "
        "per sequence cluster first and resamples clusters; 'chain' reproduces the "
        "pre-2026 per-chain numbers and is not valid for new claims.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output TSV path")
    args = parser.parse_args()
    unit = args.bootstrap

    sim_bins, sim_labels = SIM_BIN_PRESETS[args.sim_bin_preset]
    metrics = tuple(args.metrics)

    parsed = _parse_inputs(args.inputs)
    reference_label = args.reference or parsed[0][0]
    paths = dict(parsed)
    if reference_label not in paths:
        raise SystemExit(
            f"--reference {reference_label!r} is not one of: {', '.join(paths)}"
        )

    reference = _load(
        paths[reference_label],
        require_retrieval=True,
        sim_bins=sim_bins,
        sim_labels=sim_labels,
    )
    if unit == "cluster":
        reference = _require_cluster_ids(reference, paths[reference_label])
    controls = [(label, path) for label, path in parsed if label != reference_label]

    if not controls:
        out_df = pd.concat(
            [
                _aggregate_single(reference, "sim_bin", reference_label, metrics, unit),
                _aggregate_single(reference, "k_bin", reference_label, metrics, unit),
            ],
            ignore_index=True,
        )
    else:
        blocks = []
        for control_label, control_path in controls:
            control = _load(
                control_path,
                require_retrieval=False,
                sim_bins=sim_bins,
                sim_labels=sim_labels,
            )
            merged = _paired_frame(reference, control, metrics)
            if merged.empty:
                raise SystemExit(
                    f"No overlapping sample_ids between {reference_label} and "
                    f"{control_label}"
                )
            blocks.extend(
                [
                    _aggregate_paired(
                        merged,
                        "sim_bin",
                        reference_label,
                        control_label,
                        metrics,
                        args.n_bootstrap,
                        unit,
                    ),
                    _aggregate_paired(
                        merged,
                        "k_bin",
                        reference_label,
                        control_label,
                        metrics,
                        args.n_bootstrap,
                        unit,
                    ),
                ]
            )
        out_df = pd.concat(blocks, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    sim_rows = out_df[out_df["stratifier"] == "sim_bin"]
    if not sim_rows.empty:
        columns = [
            column for column in sim_rows.columns
            if column in ("bin", "control", "unit", "n_pairs", "n_chains", "n_clusters")
            or ("P@L_long" in column and "P@L_long_chain" not in column)
        ]
        print(f"\nP@L_long stratified by reference best_tpl_sim (unit={unit}):")
        print(sim_rows[columns].to_string(index=False))


if __name__ == "__main__":
    main()
