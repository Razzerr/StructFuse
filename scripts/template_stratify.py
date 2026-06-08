"""Stratify paired per-protein gains by reference template quality.

For model comparisons, chains are first matched by ``sample_id``. Bins are
defined only from the reference (StructFuse) run's ``best_tpl_sim`` or
``n_templates_retrieved``. Per-chain deltas are then aggregated inside those
reference bins. This avoids the invalid comparison where a no-template control
is assigned to bins using its own zero-valued retrieval metadata.

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


SIM_BINS = [-1e-9, 0.3, 0.5, 0.7, 1.0 + 1e-9]
SIM_LABELS = ["sim<0.3", "0.3-0.5", "0.5-0.7", "sim>0.7"]
N_TPL_BINS = [-1, 0, 1, 4, 8, 16, 1_000_000]
N_TPL_LABELS = ["k=0", "k=1", "k=2-4", "k=5-8", "k=9-16", "k>16"]
METRICS = (
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


def _load(path: Path, *, require_retrieval: bool) -> pd.DataFrame:
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
            bins=SIM_BINS,
            labels=SIM_LABELS,
            include_lowest=True,
        )
        df["k_bin"] = pd.cut(
            pd.to_numeric(df["n_templates_retrieved"], errors="coerce"),
            bins=N_TPL_BINS,
            labels=N_TPL_LABELS,
        )
    return df


def _aggregate_single(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    rows = []
    for bin_val, group in df.groupby(by, observed=True, sort=False):
        row = {
            "stratifier": by,
            "bin": str(bin_val),
            "model": label,
            "n_chains": int(len(group)),
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"n_{metric}"] = int(len(vals))
            row[f"mean_{metric}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"std_{metric}"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_paired(
    merged: pd.DataFrame,
    by: str,
    reference_label: str,
    control_label: str,
) -> pd.DataFrame:
    rows = []
    for bin_val, group in merged.groupby(by, observed=True, sort=False):
        row = {
            "stratifier": by,
            "bin": str(bin_val),
            "reference": reference_label,
            "control": control_label,
            "n_pairs": int(len(group)),
        }
        for metric in METRICS:
            ref_col = f"{metric}__reference"
            ctrl_col = f"{metric}__control"
            if ref_col not in group.columns or ctrl_col not in group.columns:
                continue
            ref = pd.to_numeric(group[ref_col], errors="coerce").to_numpy(dtype=float)
            ctrl = pd.to_numeric(group[ctrl_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(ref) & np.isfinite(ctrl)
            ref = ref[finite]
            ctrl = ctrl[finite]
            delta = ref - ctrl
            row[f"n_{metric}"] = int(len(delta))
            row[f"mean_{metric}__reference"] = (
                float(ref.mean()) if len(ref) else float("nan")
            )
            row[f"mean_{metric}__control"] = (
                float(ctrl.mean()) if len(ctrl) else float("nan")
            )
            row[f"mean_delta_{metric}"] = (
                float(delta.mean()) if len(delta) else float("nan")
            )
            row[f"std_delta_{metric}"] = (
                float(delta.std(ddof=0)) if len(delta) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_frame(reference: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    reference_columns = [
        "sample_id",
        "sim_bin",
        "k_bin",
        *[metric for metric in METRICS if metric in reference.columns],
    ]
    control_columns = [
        "sample_id",
        *[metric for metric in METRICS if metric in control.columns],
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
    parser.add_argument("--out", required=True, type=Path, help="Output TSV path")
    args = parser.parse_args()

    parsed = _parse_inputs(args.inputs)
    reference_label = args.reference or parsed[0][0]
    paths = dict(parsed)
    if reference_label not in paths:
        raise SystemExit(
            f"--reference {reference_label!r} is not one of: {', '.join(paths)}"
        )

    reference = _load(paths[reference_label], require_retrieval=True)
    controls = [(label, path) for label, path in parsed if label != reference_label]

    if not controls:
        out_df = pd.concat(
            [
                _aggregate_single(reference, "sim_bin", reference_label),
                _aggregate_single(reference, "k_bin", reference_label),
            ],
            ignore_index=True,
        )
    else:
        blocks = []
        for control_label, control_path in controls:
            control = _load(control_path, require_retrieval=False)
            merged = _paired_frame(reference, control)
            if merged.empty:
                raise SystemExit(
                    f"No overlapping sample_ids between {reference_label} and "
                    f"{control_label}"
                )
            blocks.extend(
                [
                    _aggregate_paired(
                        merged, "sim_bin", reference_label, control_label
                    ),
                    _aggregate_paired(
                        merged, "k_bin", reference_label, control_label
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
            if column in ("bin", "control", "n_pairs", "n_chains")
            or "P@L_long" in column
        ]
        print("\nP@L_long stratified by reference best_tpl_sim:")
        print(sim_rows[columns].to_string(index=False))


if __name__ == "__main__":
    main()
