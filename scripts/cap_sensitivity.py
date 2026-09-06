#!/usr/bin/env python3
"""Reconstruct the evaluation cap offline and report cap sensitivity.

Pre-registered check (memory-decisions.md, 2026-08-22). The cap is a post-hoc
row subset, so ONE uncapped test pass per run yields the whole C = 4/8/16/full
curve without retraining and without re-validating: the checkpoints and the
val-selected thresholds are held fixed, and the only thing that varies is which
chains are averaged.

Acceptance criterion, fixed before the results were seen: C=8 stands if its
headline metric AND the paired delta each differ from the uncapped value by at
most 0.002, with no qualitative change in conclusions.

Sanity gate: offline C=8 must reproduce the ordinary capped run exactly. Pass
--expect-c8 to assert it. If that fails, nothing else in the output is meaningful.

Usage:
  python scripts/cap_sensitivity.py \
      --reference logs/.../uncapped_frontier/per_sample_metrics.tsv \
      --control   logs/.../uncapped_no_templates/per_sample_metrics.tsv \
      --caps 4,8,16,0 --out .temp/cap_sensitivity.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXEMPT_SUBSETS = ("casp16",)


def apply_cap(df: pd.DataFrame, cap: int, exempt=EXEMPT_SUBSETS) -> pd.DataFrame:
    """Mirror of ContactDataset's cap: sorted sample_id, first `cap` per cluster.

    Two details that must match the dataset or the reconstruction is not the same
    subset the model was evaluated on (`dataset.py`, "Eval cap"):
      * ordering is `sorted()` over the identifier, not file or row order;
      * exempt-subset chains are kept AND do not consume the cluster's quota.
    `cap <= 0` means uncapped.
    """
    if cap <= 0:
        return df
    out, seen = [], {}
    for _, row in df.sort_values("sample_id", kind="mergesort").iterrows():
        if str(row.get("subset", "")) in exempt:
            out.append(row)
            continue
        cid = int(row["cluster_id"])
        if cid < 0:
            continue
        if seen.get(cid, 0) < cap:
            seen[cid] = seen.get(cid, 0) + 1
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def cluster_macro(df: pd.DataFrame, metric: str) -> tuple[float, int]:
    """Per-chain -> mean within cluster -> unweighted mean over clusters."""
    v = pd.to_numeric(df[metric], errors="coerce")
    cid = pd.to_numeric(df["cluster_id"], errors="coerce")
    ok = v.notna() & cid.notna() & (cid >= 0)
    if not ok.any():
        return float("nan"), 0
    per_cluster = v[ok].groupby(cid[ok]).mean()
    return float(per_cluster.mean()), int(len(per_cluster))


def cluster_bootstrap_ci(delta: np.ndarray, cids: np.ndarray, n: int, seed: int):
    keep = np.isfinite(delta) & (cids >= 0)
    delta, cids = delta[keep], cids[keep]
    if len(delta) < 2:
        return float("nan"), float("nan")
    uniq, inv = np.unique(cids, return_inverse=True)
    per_cluster = np.array([delta[inv == i].mean() for i in range(len(uniq))])
    if len(per_cluster) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    k = len(per_cluster)
    means = np.array([per_cluster[rng.integers(0, k, size=k)].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="Uncapped per_sample TSV, treatment run")
    ap.add_argument("--control", default=None, help="Uncapped per_sample TSV, control run")
    ap.add_argument("--metric", default="P@L_long")
    ap.add_argument("--caps", default="4,8,16,0", help="Comma-separated; 0 = uncapped")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-c8", type=float, default=None,
                    help="Assert offline C=8 reproduces this reference value (sanity gate).")
    ap.add_argument("--tolerance", type=float, default=0.002,
                    help="Pre-registered acceptance threshold vs uncapped.")
    ap.add_argument("--out", default=".temp/cap_sensitivity.tsv")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference, sep="\t")
    if "cluster_id" not in ref.columns:
        raise SystemExit(
            f"{args.reference} has no cluster_id column — it predates 2026-08-21 and "
            "cannot be re-aggregated per cluster."
        )
    ctl = pd.read_csv(args.control, sep="\t") if args.control else None

    caps = [int(c) for c in args.caps.split(",")]
    rows = []
    for cap in caps:
        r = apply_cap(ref, cap)
        val, n_cl = cluster_macro(r, args.metric)
        row = {"cap": cap or "full", "n_chains": len(r), "n_clusters": n_cl,
               f"ref_{args.metric}": val}
        if ctl is not None:
            c = apply_cap(ctl, cap)
            cval, _ = cluster_macro(c, args.metric)
            m = r[["sample_id", "cluster_id", args.metric]].merge(
                c[["sample_id", args.metric]], on="sample_id", suffixes=("_r", "_c"))
            d = (pd.to_numeric(m[f"{args.metric}_r"], errors="coerce")
                 - pd.to_numeric(m[f"{args.metric}_c"], errors="coerce")).to_numpy(float)
            lo, hi = cluster_bootstrap_ci(
                d, pd.to_numeric(m["cluster_id"], errors="coerce").fillna(-1).to_numpy(int),
                args.n_bootstrap, args.seed)
            row.update({f"ctl_{args.metric}": cval, "paired_delta": float(np.nanmean(d)),
                        "ci95_lo": lo, "ci95_hi": hi, "n_paired": len(m)})
        rows.append(row)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False, float_format="%.6f")
    print(out.to_string(index=False))
    print(f"\nSaved {args.out}")

    full = out[out["cap"] == "full"]
    c8 = out[out["cap"] == 8]
    if args.expect_c8 is not None and not c8.empty:
        got = float(c8[f"ref_{args.metric}"].iloc[0])
        delta = abs(got - args.expect_c8)
        ok = delta < 1e-4
        print(f"\nSANITY GATE offline C=8 vs reported: {got:.6f} vs {args.expect_c8:.6f} "
              f"(|d|={delta:.2e}) {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("Offline C=8 does not reproduce the capped run — "
                             "the reconstruction is wrong, ignore the rest.")
    if not full.empty and not c8.empty:
        dm = abs(float(c8[f"ref_{args.metric}"].iloc[0]) - float(full[f"ref_{args.metric}"].iloc[0]))
        print(f"\nC=8 vs full, {args.metric}: |d| = {dm:.6f} "
              f"({'WITHIN' if dm <= args.tolerance else 'EXCEEDS'} tolerance {args.tolerance})")
        if "paired_delta" in out.columns:
            dd = abs(float(c8["paired_delta"].iloc[0]) - float(full["paired_delta"].iloc[0]))
            print(f"C=8 vs full, paired delta: |d| = {dd:.6f} "
                  f"({'WITHIN' if dd <= args.tolerance else 'EXCEEDS'} tolerance)")


if __name__ == "__main__":
    main()
