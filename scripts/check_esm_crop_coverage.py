#!/usr/bin/env python3
"""Exposure of training and evaluation to crops that fall outside the ESM cache.

`precompute_esm2_embeddings.py` truncates at --max_len (1022, the ESM2 limit), so
a longer chain has a SHORTER cached `rep` than its contact map. `collate_padded`
crops both with bounds taken from the full chain length, so `rep_full[s:e]`
silently returns fewer rows and the uncovered part of `h_esm` stays zero while
contact/mask/pair_mask cover the whole crop.

Masking the loss on those positions would NOT be a sufficient fix: zero rows
still enter the InstanceNorm2d statistics in PairFeatures and are still attended
to by the axial attention, which runs without attn_mask. This audit therefore
measures how much of the pipeline is affected in order to choose the fix, not to
decide whether silent zero-filling is tolerable.

Three things it does that a naive average does not:
  * TRAIN is aggregated per CLUSTER, because ClusterTrainValSampler draws one
    chain per cluster per epoch — a large family must not weigh more just for
    having more chains.
  * VAL/TEST use the CENTRE crop actually used at evaluation, not a random one,
    reported separately for the C=8 cap and the full split.
  * Cache lengths are READ, not assumed to be min(L, 1022), and missing ids are
    reported rather than skipped.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def cached_len(cache_dir: Path, stem: str) -> int | None:
    """Rows of `rep` from the .npy header only — no array decompression."""
    path = cache_dir / f"{stem}.npz"
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.startswith("rep"))
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
        return int(shape[0])
    except Exception:
        return None


def random_crop_exposure(length: int, crop: int, cov: int) -> tuple[float, float, float]:
    """(P(any uncovered), mean uncovered fraction, P(fully uncovered)) over all starts."""
    if length <= crop:
        starts = [0]
    else:
        starts = range(0, length - crop + 1)
    n = any_ = full = 0
    frac = 0.0
    for s in starts:
        end = min(s + crop, length)
        covered = max(0, min(end, cov) - s)
        want = end - s
        f = 1.0 - covered / want if want else 0.0
        frac += f
        any_ += f > 0
        full += covered == 0
        n += 1
    return any_ / n, frac / n, full / n


def centre_crop_exposure(length: int, crop: int, cov: int) -> tuple[bool, float]:
    """Evaluation uses the centre window; this is the exact single case."""
    start = max(0, (length - crop) // 2)
    end = min(start + crop, length)
    covered = max(0, min(end, cov) - start)
    want = end - start
    f = 1.0 - covered / want if want else 0.0
    return f > 0, f


def load_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="data/processed_2026/npz_lengths.json")
    ap.add_argument("--split-dir", default="data/output_splits_2026")
    ap.add_argument("--cache-dir", default="data/precomputed/esm_t6_8M_2026")
    ap.add_argument("--clusters", default="data/output_splits_2026/chain_clusters.tsv")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--esm-max-len", type=int, default=1022)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--verify-sample", type=int, default=2000,
                    help="Short chains whose cache length is read anyway, to test "
                         "the min(L, max_len) assumption instead of trusting it.")
    args = ap.parse_args()

    lengths = {k: int(v) for k, v in json.loads(Path(args.lengths).read_text()).items()}
    cache = Path(args.cache_dir)
    split_dir = Path(args.split_dir)

    chain2cluster: dict[str, int] = {}
    cl_path = Path(args.clusters)
    if cl_path.exists():
        for i, line in enumerate(cl_path.read_text().splitlines()):
            if i == 0 or not line.strip():
                continue
            f = line.split("\t")
            chain2cluster[f[0]] = int(f[1])

    splits = {}
    for name, fn in (("train", "all_train_ids.txt"), ("val", "val_holdout_ids.txt"),
                     ("test", "test_ids.txt")):
        p = split_dir / fn
        if p.exists():
            splits[name] = load_ids(p)

    rng = np.random.default_rng(0)
    print(f"crop={args.crop}  esm_max_len={args.esm_max_len}  cap=C{args.cap}\n")

    # --- assumption check: is the cached length really min(L, max_len)? ---
    all_ids = sorted({i for ids in splits.values() for i in ids})
    long_ids = [i for i in all_ids if lengths.get(i, 0) > args.esm_max_len]
    sample = list(rng.choice([i for i in all_ids if i not in set(long_ids)],
                             size=min(args.verify_sample, len(all_ids)), replace=False))
    cov: dict[str, int] = {}
    mismatch = missing_cache = 0
    for stem in long_ids + sample:
        c = cached_len(cache, stem)
        if c is None:
            missing_cache += 1
            continue
        cov[stem] = c
        if c != min(lengths.get(stem, 0), args.esm_max_len):
            mismatch += 1
    print(f"cache lengths read: {len(cov)}  (all {len(long_ids)} chains > max_len "
          f"+ {len(sample)} sampled short)")
    print(f"  missing cache files : {missing_cache}   {'OK' if not missing_cache else '<-- INVESTIGATE'}")
    print(f"  != min(L, max_len)  : {mismatch}   {'OK' if not mismatch else '<-- assumption broken'}")

    missing_len = sum(1 for ids in splits.values() for i in ids if i not in lengths)
    print(f"  ids absent from lengths: {missing_len}   "
          f"{'OK' if not missing_len else '<-- results are conditional'}\n")

    def coverage_of(stem: str) -> int:
        return cov.get(stem, min(lengths.get(stem, 0), args.esm_max_len))

    # --- TRAIN: random crop, aggregated per cluster ---
    if "train" in splits:
        per_cluster: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        n_nc = 0
        for stem in splits["train"]:
            L = lengths.get(stem)
            if L is None:
                continue
            cid = chain2cluster.get(stem, -1)
            if cid < 0:
                n_nc += 1
                continue
            per_cluster[cid].append(random_crop_exposure(L, args.crop, coverage_of(stem)))
        if per_cluster:
            means = np.array([np.mean(v, axis=0) for v in per_cluster.values()])
            m = means.mean(axis=0)
            print(f"TRAIN (random crop, one chain per cluster per epoch)")
            print(f"  clusters={len(per_cluster)}  chains={len(splits['train'])}"
                  f"  unclustered_skipped={n_nc}")
            print(f"  P(crop has uncovered positions) = {m[0]:.5f}")
            print(f"  mean uncovered fraction of crop = {m[1]:.5f}")
            print(f"  P(crop entirely uncovered)      = {m[2]:.5f}\n")

    # --- VAL/TEST: centre crop, capped and full ---
    for name in ("val", "test"):
        if name not in splits:
            continue
        ids = splits[name]
        capped, seen = [], defaultdict(int)
        for stem in sorted(ids):
            cid = chain2cluster.get(stem, -1)
            if cid < 0:
                continue
            if seen[cid] < args.cap:
                seen[cid] += 1
                capped.append(stem)
        for label, subset in ((f"C={args.cap}", capped), ("full", ids)):
            hits = [centre_crop_exposure(lengths[s], args.crop, coverage_of(s))
                    for s in subset if s in lengths]
            n_any = sum(h for h, _ in hits)
            frac = float(np.mean([f for _, f in hits])) if hits else 0.0
            print(f"{name.upper()} centre crop, {label}: chains={len(hits)}  "
                  f"affected={n_any} ({100*n_any/max(1,len(hits)):.3f}%)  "
                  f"mean uncovered fraction={frac:.5f}")
    print("\nMasking these positions in the loss would not be a fix: they still enter "
          "InstanceNorm2d statistics and axial attention, which runs without attn_mask.")


if __name__ == "__main__":
    main()
