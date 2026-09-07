#!/usr/bin/env python3
"""How often does a training crop fall outside the cached ESM embedding?

`precompute_esm2_embeddings.py` truncates sequences at --max_len (1022, the ESM2
limit), so a chain longer than that has a SHORTER cached `rep` than its contact
map. `collate_padded` crops both with the same bounds taken from the full chain
length, so for a long chain `rep_full[start:end]` silently returns fewer rows —
or none at all — and the uncovered positions of `h_esm` stay zero while
`contact`, `mask` and `pair_mask` cover the whole crop.

The result is valid labels paired with zero features. It is silent: no warning,
no shape error. Training on those positions is learning from noise.

This script quantifies the exposure from chain lengths alone — no GPU, no model.

Usage:
  python scripts/check_esm_crop_coverage.py \
      --lengths data/processed_2026/npz_lengths.json \
      --split-dir data/output_splits_2026 --crop 384 --esm-max-len 1022
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def expected_uncovered(length: int, crop: int, esm_max: int) -> tuple[float, float]:
    """(P(crop is partly uncovered), mean uncovered fraction of the crop).

    Training uses a uniform random start in [0, length - crop]; evaluation uses
    the centre. Enumerating every start is exact and cheap at these sizes.
    """
    if length <= crop:
        starts = [0]
    else:
        starts = range(0, length - crop + 1)
    n = 0
    total_frac = 0.0
    n_any = 0
    for s in starts:
        end = min(s + crop, length)
        covered = max(0, min(end, esm_max) - s)
        want = end - s
        frac = 1.0 - covered / want if want else 0.0
        total_frac += frac
        n_any += frac > 0
        n += 1
    return n_any / n, total_frac / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="data/processed_2026/npz_lengths.json")
    ap.add_argument("--split-dir", default="data/output_splits_2026")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--esm-max-len", type=int, default=1022)
    args = ap.parse_args()

    lengths = json.loads(Path(args.lengths).read_text())
    lengths = {k: int(v) for k, v in lengths.items()}

    splits = {}
    for name, fn in (("train", "all_train_ids.txt"), ("val", "val_holdout_ids.txt"),
                     ("test", "test_ids.txt")):
        p = Path(args.split_dir) / fn
        if p.exists():
            splits[name] = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    print(f"crop={args.crop}  esm_max_len={args.esm_max_len}\n")
    print(f"{'split':>6} {'chains':>9} {'>max_len':>9} {'%':>7} "
          f"{'P(crop hit)':>12} {'mean lost':>10} {'fully empty':>12}")
    for name, ids in splits.items():
        ls = [lengths[i] for i in ids if i in lengths]
        long_ones = [L for L in ls if L > args.esm_max_len]
        p_hit = lost = fully = 0.0
        for L in long_ones:
            p, f = expected_uncovered(L, args.crop, args.esm_max_len)
            p_hit += p
            lost += f
            # a crop entirely past the embedding: start >= esm_max
            starts = max(1, L - args.crop + 1)
            fully += max(0, starts - args.esm_max_len) / starts
        n = len(ls)
        print(f"{name:>6} {n:9d} {len(long_ones):9d} {100*len(long_ones)/max(1,n):6.2f}% "
              f"{p_hit/max(1,n):12.5f} {lost/max(1,n):10.5f} {fully/max(1,n):12.5f}")

    print("\nP(crop hit)  = probability a uniformly random training crop includes "
          "at least one position with no ESM features")
    print("mean lost    = expected fraction of a crop's positions left as zeros")
    print("fully empty  = probability the ENTIRE crop has zero ESM features "
          "while its contact labels are intact")


if __name__ == "__main__":
    main()
