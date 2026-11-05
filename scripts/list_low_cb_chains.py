#!/usr/bin/env python3
"""List chains whose Cβ labels are mostly a Cα fallback (decided 2026-11-05: exclude).

A residue gets the Cα fallback when it is not glycine, has no CB atom, and lacks
N or C, so no virtual Cβ can be built (`pdb_utils.residue_cb`). That happens in
Cα-only models. For such chains the "Cβ" contact map is really a Cα map, so they
are excluded everywhere - dataset (`data.skip_ids_files`), ESM caches and FAISS
index (`build_index.py --skip_ids_files`) - like unclustered chains.

Rule: fraction of Cα-fallback residues among residues with a Cα > --threshold
(default 0.5). Read from each NPZ's `cb_source`; a Cα-generation NPZ (no
`cb_source`) is an error, not a pass.

Usage (server, inside an allocation):
    python scripts/list_low_cb_chains.py --processed-dir data/processed_2026cb \
        --out data/output_splits_2026cb/low_cb_ids.txt --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

CB_NONE, CB_CA_FALLBACK = 0, 4


def fallback_fraction(item: tuple) -> tuple:
    stem, root = item
    with np.load(Path(root) / f"{stem}.npz", allow_pickle=False) as z:
        if "cb_source" not in z.files:
            raise ValueError(f"{stem}: no cb_source (not a Cβ-generation NPZ)")
        src = z["cb_source"]
    with_ca = int((src != CB_NONE).sum())
    fb = int((src == CB_CA_FALLBACK).sum())
    return stem, with_ca, fb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    stems = sorted(json.loads((args.processed_dir / "npz_lengths.json").read_text()))
    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        rows = list(pool.imap_unordered(fallback_fraction, [(s, str(args.processed_dir)) for s in stems],
                                        chunksize=512))
    listed = sorted(s for s, n, fb in rows if n > 0 and fb / n > args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{s}\n" for s in listed))
    meta = {"processed_dir": str(args.processed_dir), "threshold": args.threshold, "chains_scanned": len(rows),
            "chains_listed": len(listed), "chains_with_any_fallback": sum(fb > 0 for _, _, fb in rows),
            "rule": "fraction of Cα-fallback residues among residues with a Cα > threshold"}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
