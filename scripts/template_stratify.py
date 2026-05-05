"""Stratify per-protein test metrics by template-quality bins.

Reads `per_sample_metrics.tsv` files and reports mean P@L_long (and other
metrics) within bins of:
  - `best_tpl_sim`  (top-K cosine similarity from FAISS, in [0, 1])
  - `n_templates_retrieved` (count of templates after holdout/cluster filter)

Used in paper supplementary to defend "retrieval carries signal": if the
delta (frontier - no_templates) grows monotonically with template quality,
the model is genuinely using templates, not just ESM2.

Usage (single run, one row per bin):
    python scripts/template_stratify.py \\
        --inputs frontier=path/to/frontier.tsv \\
        --out .temp/audit/frontier_strat.tsv

Usage (two runs, side-by-side per bin):
    python scripts/template_stratify.py \\
        --inputs frontier=path/to/frontier.tsv no_templates=path/to/notpl.tsv \\
        --out .temp/audit/frontier_vs_notpl_strat.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Bin edges chosen to span the meaningful range of cosine similarity in this
# project (FAISS topk_precomputed scores empirically fall in ~[0.0, 1.0],
# with 0.7+ being clear homologue territory and <0.3 being "remote/noise").
SIM_BINS = [-1e-9, 0.3, 0.5, 0.7, 1.0 + 1e-9]
SIM_LABELS = ["sim<0.3", "0.3-0.5", "0.5-0.7", "sim>0.7"]
N_TPL_BINS = [-1, 0, 1, 4, 8, 16, 1_000_000]
N_TPL_LABELS = ["k=0", "k=1", "k=2-4", "k=5-8", "k=9-16", "k>16"]
METRICS = ("P@L", "P@L_long", "P@L/2_long", "P@L/5_long", "AUC-PR_long", "f1")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for col in ("best_tpl_sim", "n_templates_retrieved"):
        if col not in df.columns:
            raise SystemExit(
                f"{path} is missing column '{col}'. Re-export the per-sample dump "
                f"with the updated contact_lit_module._export_per_sample_metrics."
            )
    df["sim_bin"] = pd.cut(df["best_tpl_sim"], bins=SIM_BINS, labels=SIM_LABELS, include_lowest=True)
    df["k_bin"] = pd.cut(df["n_templates_retrieved"], bins=N_TPL_BINS, labels=N_TPL_LABELS)
    return df


def _aggregate(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    rows = []
    for bin_val, group in df.groupby(by, observed=True, sort=False):
        row = {"stratifier": by, "bin": str(bin_val), "model": label, "n_chains": len(group)}
        for m in METRICS:
            if m in group.columns:
                vals = pd.to_numeric(group[m], errors="coerce").dropna()
                row[f"mean_{m}"] = round(float(vals.mean()), 4) if len(vals) else float("nan")
                row[f"std_{m}"] = round(float(vals.std(ddof=0)), 4) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more name=path pairs (e.g. frontier=run_a.tsv no_templates=run_b.tsv)",
    )
    p.add_argument("--out", required=True, type=Path, help="Output TSV path")
    args = p.parse_args()

    parsed = []
    for spec in args.inputs:
        if "=" not in spec:
            raise SystemExit(f"--inputs entries must be name=path, got {spec!r}")
        label, path_s = spec.split("=", 1)
        df = _load(Path(path_s))
        df["__model__"] = label
        parsed.append((label, df))

    blocks = []
    for label, df in parsed:
        blocks.append(_aggregate(df, "sim_bin", label))
        blocks.append(_aggregate(df, "k_bin", label))

    combined = pd.concat(blocks, ignore_index=True)

    # Long format → wide format with one row per (stratifier, bin) and one column per (metric, model)
    if len(parsed) >= 2:
        # Pivot so each model has its own column block; compute delta vs. first model.
        first_label = parsed[0][0]
        pivot = combined.pivot_table(
            index=["stratifier", "bin"], columns="model", values=[f"mean_{m}" for m in METRICS] + ["n_chains"]
        )
        pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()
        for m in METRICS:
            base = f"mean_{m}__{first_label}"
            if base in pivot.columns:
                for label, _ in parsed[1:]:
                    other = f"mean_{m}__{label}"
                    if other in pivot.columns:
                        pivot[f"delta_{m}__{first_label}_minus_{label}"] = pivot[base] - pivot[other]
        out_df = pivot
    else:
        out_df = combined

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    if "stratifier" in out_df.columns:
        # Print sim_bin slice for quick eyeball — the headline argument.
        sim_rows = out_df[out_df["stratifier"] == "sim_bin"]
        if not sim_rows.empty:
            cols = [c for c in sim_rows.columns if c == "bin" or c.startswith("mean_P@L_long") or c.startswith("delta_P@L_long") or c.startswith("n_chains")]
            print("\nP@L_long stratified by best_tpl_sim:")
            print(sim_rows[cols].to_string(index=False))


if __name__ == "__main__":
    main()
