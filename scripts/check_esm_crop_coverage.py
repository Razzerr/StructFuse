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


def build_dataset(split_file: Path, cfg, cap: int | None):
    """Chain list exactly as production builds it.

    Split files hold PDB ENTRY ids (`1abc`); lengths, cache and clusters are keyed
    by CHAIN (`1abc_A`). ContactDataset does that expansion — plus min_len, the
    skip lists, subset exclusion, and the evaluation cap with its CASP16 exemption.
    Reimplementing any of it here is how the first version of this audit came to
    compare entries against chains and silently match nothing.
    """
    from src.data.components.dataset import ContactDataset

    kwargs = dict(
        root=cfg["data_root"],
        min_len=cfg["min_len"],
        splits_json_path=cfg["splits_json_path"],
        skip_ids_files=cfg["skip_ids_files"],
    )
    if cap is not None:
        kwargs.update(
            max_chains_per_cluster=cap,
            chain_clusters_file=cfg["chain_clusters_file"],
            cap_exempt_subsets=cfg["cap_exempt_subsets"],
        )
    return ContactDataset(split_file, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="frontier_8M",
                    help="Hydra experiment, so paths/min_len/skip-lists/cap match production.")
    ap.add_argument("--cache-dir", default=None,
                    help="Override data.esm_embeddings_dir (e.g. the 650M cache).")
    ap.add_argument("--lengths", default=None, help="Override npz_lengths.json path.")
    ap.add_argument("--crop", type=int, default=None, help="Override data.crop_size.")
    ap.add_argument("--esm-max-len", type=int, default=1022)
    ap.add_argument("--verify-sample", type=int, default=2000,
                    help="Short chains whose cache length is read anyway, to test the "
                         "min(L, max_len) assumption instead of trusting it.")
    args = ap.parse_args()

    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3",
                               config_dir=str(Path.cwd() / "configs")):
        cfg_all = compose(config_name="train", overrides=[f"experiment={args.experiment}"])
    d = OmegaConf.to_container(cfg_all.data, resolve=True)

    crop = args.crop or int(d["crop_size"])
    cap = int(d["max_chains_per_cluster"]) if d.get("max_chains_per_cluster") else None
    cache = Path(args.cache_dir or d["esm_embeddings_dir"])
    split_dir = Path(d["split_dir"])
    lengths_path = Path(args.lengths or (Path(d["data_root"]) / "npz_lengths.json"))
    lengths = {k: int(v) for k, v in json.loads(lengths_path.read_text()).items()}

    cfg = {k: d[k] for k in ("data_root", "min_len", "splits_json_path",
                             "skip_ids_files", "chain_clusters_file",
                             "cap_exempt_subsets")}
    chain2cluster: dict[str, int] = {}
    cl_path = Path(cfg["chain_clusters_file"])
    if cl_path.exists():
        for i, line in enumerate(cl_path.read_text().splitlines()):
            if i and line.strip():
                f = line.split("\t")
                chain2cluster[f[0]] = int(f[1])

    print(f"experiment={args.experiment}  crop={crop}  esm_max_len={args.esm_max_len}  "
          f"cap=C{cap}\ncache={cache}\n")

    # Chain lists straight from production, so entry->chain expansion, min_len,
    # skip lists, subset exclusion and the CASP16 cap exemption are not re-derived.
    sets = {
        "train": build_dataset(split_dir / "all_train_ids.txt", cfg, None).ids,
        "val_full": build_dataset(split_dir / "val_holdout_ids.txt", cfg, None).ids,
        "test_full": build_dataset(split_dir / "test_ids.txt", cfg, None).ids,
    }
    if cap:
        sets[f"val_C{cap}"] = build_dataset(split_dir / "val_holdout_ids.txt", cfg, cap).ids
        sets[f"test_C{cap}"] = build_dataset(split_dir / "test_ids.txt", cfg, cap).ids

    rng = np.random.default_rng(0)
    all_ids = sorted({i for v in sets.values() for i in v})
    long_set = {i for i in all_ids if lengths.get(i, 0) > args.esm_max_len}
    short_ids = [i for i in all_ids if i not in long_set]
    sample = list(rng.choice(short_ids, size=min(args.verify_sample, len(short_ids)),
                             replace=False)) if short_ids else []

    cov: dict[str, int] = {}
    unreadable: list[str] = []

    def read_into(stems) -> int:
        bad = 0
        for stem in stems:
            c = cached_len(cache, stem)
            if c is None:
                unreadable.append(stem)
                continue
            cov[stem] = c
            bad += c != min(lengths.get(stem, 0), args.esm_max_len)
        return bad

    mismatch = read_into(sorted(long_set) + sample)
    print(f"cache lengths READ for {len(cov)} chains "
          f"(all {len(long_set)} over max_len + {len(sample)} sampled short)")
    if mismatch:
        print(f"  sample mismatch -> escalating to a full read")
        mismatch += read_into([i for i in all_ids if i not in cov])
    print(f"  != min(L, max_len)      : {mismatch}   "
          f"{'OK' if not mismatch else '<-- lengths are NOT min(L, max_len)'}")
    print(f"  unreadable/missing cache: {len(unreadable)}   "
          f"{'OK' if not unreadable else '<-- AUDIT INCOMPLETE'}")
    n_missing_len = sum(1 for i in all_ids if i not in lengths)
    print(f"  ids absent from lengths : {n_missing_len}   "
          f"{'OK' if not n_missing_len else '<-- results are conditional'}")

    assumed: set[str] = set()

    def coverage_of(stem: str) -> int:
        """Read length where available; otherwise fall back and COUNT it.

        Substituting min(L, max_len) for an unreadable file would quietly
        reinstate the assumption this audit exists to test, so those chains are
        tallied and reported rather than blending into the averages unremarked.
        """
        if stem in cov:
            return cov[stem]
        assumed.add(stem)
        return min(lengths.get(stem, 0), args.esm_max_len)

    # --- TRAIN: random crop, one chain per cluster per epoch ---
    per_cluster: dict[int, list] = defaultdict(list)
    n_nc = 0
    for stem in sets["train"]:
        L = lengths.get(stem)
        if L is None:
            continue
        cid = chain2cluster.get(stem, -1)
        if cid < 0:
            n_nc += 1
            continue
        per_cluster[cid].append(random_crop_exposure(L, crop, coverage_of(stem)))
    if per_cluster:
        m = np.array([np.mean(v, axis=0) for v in per_cluster.values()]).mean(axis=0)
        print(f"\nTRAIN — random crop, averaged within cluster then over clusters")
        print(f"  chains={len(sets['train'])}  clusters={len(per_cluster)}  "
              f"unclustered_skipped={n_nc}")
        print(f"  P(crop has uncovered positions) = {m[0]:.5f}")
        print(f"  mean uncovered fraction of crop = {m[1]:.5f}")
        print(f"  P(crop entirely uncovered)      = {m[2]:.5f}")

    # --- VAL/TEST: the centre crop actually used at evaluation ---
    print(f"\nVAL/TEST — centre crop (as evaluated)")
    for name in sorted(k for k in sets if k != "train"):
        hits = [centre_crop_exposure(lengths[s], crop, coverage_of(s))
                for s in sets[name] if s in lengths]
        n_any = sum(h for h, _ in hits)
        frac = float(np.mean([f for _, f in hits])) if hits else 0.0
        print(f"  {name:<12} chains={len(hits):>7}  affected={n_any:>5} "
              f"({100*n_any/max(1,len(hits)):.3f}%)  mean uncovered={frac:.5f}")

    if assumed:
        print(f"\n{len(assumed)} chains used an ASSUMED coverage of min(L, max_len); "
              f"their contribution above is not established.")
    print("\nMasking these positions in the loss would not be a fix: they still enter "
          "InstanceNorm2d statistics and axial attention, which runs without attn_mask.")


if __name__ == "__main__":
    main()
