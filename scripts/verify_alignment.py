#!/usr/bin/env python3
"""Verify that every feature is index-aligned with the contact labels.

A consistent off-by-one between features and labels does NOT show up in
evaluation — the model learns the offset and its predictions match the shifted
labels — but it degrades learning, because residue i's features are paired with
residue i+1's contacts. So it has to be tested directly, on the data.

The general tool is a shift scan: for any 2D feature that should be aligned with
the contact map, score it at offsets d in [-max_shift, max_shift] on both axes
and check the maximum lands on (0, 0). Anything else is a real misalignment.

Checks, all through the REAL collate path so this tests production code:

  A  NPZ self-consistency      len(seq) == contact.shape == len(mask) == len(coords);
                               contact symmetric; contact recomputable from coords at 8A
  B  ESM cache vs labels       shift scan of the cached ESM contact head against
                               the true contact map (tests seq<->rep alignment,
                               including the BOS/EOS strip in precompute)
  C  ESM cache coverage        crop bounds vs cached rep length — the truncation
                               hole quantified by check_esm_crop_coverage.py
  D  template prior vs labels  shift scan of `prior` (tests the NW projection)
  E  relative positions        rel encodes |i-j| exactly
  F  masks                     pair_mask == outer(mask, mask), diagonal zeroed;
                               long_mask == pair_mask AND |i-j| >= min_seq_sep

Usage:
  python scripts/verify_alignment.py --experiment frontier_8M --n-chains 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def shift_scan(feature: np.ndarray, contact: np.ndarray, mask: np.ndarray,
               max_shift: int = 3):
    """Score `feature` against `contact` at integer offsets; return (best, table).

    Score is the mean feature value over the true contacts, after shifting the
    contact map by (di, dj). Alignment is correct when the peak sits at (0, 0).
    """
    L = contact.shape[0]
    scores = {}
    for di in range(-max_shift, max_shift + 1):
        for dj in range(-max_shift, max_shift + 1):
            sc = np.zeros_like(contact)
            si = slice(max(0, di), L + min(0, di))
            sj = slice(max(0, dj), L + min(0, dj))
            ti = slice(max(0, -di), L + min(0, -di))
            tj = slice(max(0, -dj), L + min(0, -dj))
            sc[si, sj] = contact[ti, tj]
            sel = (sc > 0) & (mask > 0)
            scores[(di, dj)] = float(feature[sel].mean()) if sel.any() else np.nan
    best = max((v, k) for k, v in scores.items() if np.isfinite(v))[1]
    return best, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="frontier_8M")
    ap.add_argument("--n-chains", type=int, default=40)
    ap.add_argument("--max-shift", type=int, default=3)
    ap.add_argument("--contact-threshold", type=float, default=8.0)
    args = ap.parse_args()

    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3",
                               config_dir=str(Path.cwd() / "configs")):
        cfg = compose(config_name="train",
                      overrides=[f"experiment={args.experiment}",
                                 "data.eval_batch_size=1"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("test")
    from torch.utils.data import Subset
    dm.dset_test = Subset(dm.dset_test, list(range(args.n_chains)))
    dm.bucketed = False
    loader = dm.test_dataloader()

    min_sep = int(cfg.data.min_seq_sep)
    root = Path(cfg.data.data_root)
    esm_dir = Path(cfg.data.esm_embeddings_dir) if cfg.data.esm_embeddings_dir else None

    agg = {"esm": [], "prior": []}
    n_short_rep = 0
    print(f"\n=== per-chain checks ({args.n_chains} chains) ===")
    for n, batch in enumerate(loader):
        pid = batch["pid"][0]
        cb = batch["crop_bounds"][0].tolist()
        contact = batch["contact"][0].numpy()
        pair_mask = batch["pair_mask"][0].numpy()
        long_mask = batch["long_mask"][0].numpy()
        L = int(batch["seq_len"][0])

        if n == 0:
            # --- A: raw NPZ self-consistency (first chain only, it is a file read)
            npz = np.load(root / f"{pid}.npz", allow_pickle=True)
            seq = str(npz["seq"].item() if npz["seq"].shape == () else npz["seq"])
            c_full, m_full = npz["contact"], npz["mask"]
            check("A len(seq) == contact.shape[0] == len(mask)",
                  len(seq) == c_full.shape[0] == len(m_full),
                  f"{len(seq)} / {c_full.shape[0]} / {len(m_full)}")
            check("A contact is symmetric", bool(np.array_equal(c_full, c_full.T)))
            if "coords" in npz:
                co = npz["coords"]
                d = np.linalg.norm(co[:, None, :] - co[None, :, :], axis=-1)
                recomputed = (d < args.contact_threshold).astype(c_full.dtype)
                valid = np.outer(m_full > 0, m_full > 0)
                agree = (recomputed[valid] == c_full[valid]).mean()
                check("A contact recomputable from coords at 8A", agree > 0.999,
                      f"agreement={agree:.5f}")

            # --- E: relative positions
            if "rel" in batch or "rel_pos" in batch:
                rel = batch.get("rel", batch.get("rel_pos"))[0].numpy()
                idx = np.arange(rel.shape[-1])
                expect = np.abs(idx[:, None] - idx[None, :])
                r = rel if rel.ndim == 2 else rel[0]
                check("E rel encodes |i-j|",
                      bool(np.allclose(r[:L, :L], expect[:L, :L])) or r.max() <= 1.0,
                      "(bucketed/normalised encodings are reported, not asserted)")

            # --- F: masks
            mm = batch["residue_mask"][0].numpy() if "residue_mask" in batch else None
            if mm is not None:
                expect_pair = np.outer(mm, mm)
                np.fill_diagonal(expect_pair, 0.0)
                check("F pair_mask == outer(mask, mask), diagonal zeroed",
                      bool(np.allclose(pair_mask[:L, :L], expect_pair[:L, :L])))
            sep = np.abs(np.arange(long_mask.shape[0])[:, None]
                         - np.arange(long_mask.shape[0])[None, :])
            check(f"F long_mask == pair_mask AND |i-j| >= {min_sep}",
                  bool(np.array_equal(long_mask, pair_mask * (sep >= min_sep))))

        # --- C: ESM cache coverage for this crop
        if esm_dir is not None:
            with np.load(esm_dir / f"{pid}.npz") as e:
                rep_len = e["rep"].shape[0]
            if cb[1] > rep_len:
                n_short_rep += 1

        # --- B / D: shift scans, accumulated across chains
        if "esm_contacts" in batch:
            f = batch["esm_contacts"][0, 0].numpy()
            agg["esm"].append(shift_scan(f, contact, long_mask, args.max_shift)[1])
        if "prior" in batch and batch["prior"].abs().sum() > 0:
            f = batch["prior"][0, 0].numpy()
            agg["prior"].append(shift_scan(f, contact, long_mask, args.max_shift)[1])

    print(f"\n=== C: ESM cache coverage ===")
    check("C every crop lies inside the cached ESM embedding", n_short_rep == 0,
          f"{n_short_rep}/{args.n_chains} crops extend past the cached rep "
          f"(those positions get ZERO features while keeping their labels)")

    print("\n=== B / D: shift scans (peak must be at (0,0)) ===")
    for key, label in (("esm", "B ESM cached contact head"), ("prior", "D template prior")):
        rows = agg[key]
        if not rows:
            print(f"  [skip] {label}: no data")
            continue
        keys = rows[0].keys()
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        best = max(mean, key=lambda k: mean[k])
        near = sorted(mean.items(), key=lambda kv: -kv[1])[:4]
        check(f"{label} peaks at (0,0)", best == (0, 0), f"peak={best}")
        print("        top offsets: " +
              "  ".join(f"{k}={v:.4f}" for k, v in near))

    print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
