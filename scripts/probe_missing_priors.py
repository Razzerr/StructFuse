"""Diagnose why specific val chains retrieve zero templates.

Runs FaissIndex.topk_precomputed directly (filter_holdout=False, same as
eval collate) on a hand-picked list of PIDs, but with every filtering stage
logged so we can tell whether the miss is caused by:
  (a) the query itself being absent from the index embedding table,
  (b) same-PDB filter eating every hit (orphan subunit),
  (c) same-cluster filter (rare but possible for auto-clustered chains),
  (d) min_similarity threshold,
  (e) all candidates falling into the holdout set (only matters if
      filter_holdout=True, which we don't use for eval).

Usage (server):
    python scripts/probe_missing_priors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.utils.faiss import FaissIndex, _get_protein_id


PIDS_TO_PROBE = [
    "8fzd_p",
    "8fzd_q",
    "8fzd_r",
    "8fzd_t",
    "8fzd_u",
    # Include one that WORKED so we can compare.
    "8fzd_1",
]

INDEX_DIR = "data/index_t33"  # user renamed old → _bak, new index lives here
HOLDOUT_FILES = [
    "data/output_splits/val_holdout_ids.txt",
    "data/output_splits/test_ids.txt",
]
TOPK = 4
MIN_SIMILARITY = 0.0  # match default; inspect a wider net
SEARCH_K = 50  # over-fetch for diagnostic visibility


def load_holdout(files):
    out = set()
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"[warn] holdout file missing: {f}")
            continue
        with open(p) as fh:
            for line in fh:
                s = line.strip()
                if s:
                    out.add(s)
    return out


def probe(idx: FaissIndex, pid: str, filter_holdout: bool):
    print(f"\n{'='*80}")
    print(f"Query: {pid}   filter_holdout={filter_holdout}")
    print("=" * 80)

    query_prot_id = _get_protein_id(pid)
    query_cluster = idx.prot2cluster.get(query_prot_id, -1)
    print(f"  prot_id={query_prot_id}  cluster={query_cluster}")
    print(f"  in index embedding table: {pid in idx._id2row}")
    print(f"  in holdout set:           {query_prot_id in idx.holdout_prot_ids}")

    if idx._embeddings is None or pid not in idx._id2row:
        print("  !! query has no precomputed embedding — retrieval will return []")
        return

    row = idx._id2row[pid]
    x = idx._embeddings[row : row + 1].astype(np.float32)
    print(f"  query embedding norm: {np.linalg.norm(x):.4f}  "
          f"(should be ~1.0 for L2-normalised index)")

    sims, idxs = idx.index.search(x, SEARCH_K)
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()

    print(f"  raw top-{SEARCH_K} (before filters):")
    for rank, (sim, row_idx) in enumerate(zip(sims[:15], idxs[:15])):
        if row_idx < 0:
            print(f"    [{rank:2d}] --- no result ---")
            continue
        tpl_id = idx.row2id[row_idx]
        tpl_prot = _get_protein_id(tpl_id)
        tpl_cluster = idx.row2cluster[row_idx]
        flags = []
        if tpl_prot == query_prot_id:
            flags.append("SAME_PROT")
        if filter_holdout and tpl_prot in idx.holdout_prot_ids:
            flags.append("HOLDOUT")
        if query_cluster != -1 and tpl_cluster == query_cluster:
            flags.append("SAME_CLUSTER")
        if sim < MIN_SIMILARITY:
            flags.append(f"LOW_SIM<{MIN_SIMILARITY}")
        flag_str = "|".join(flags) if flags else "OK"
        print(f"    [{rank:2d}] sim={sim:.4f}  {tpl_id:20s}  cluster={tpl_cluster:>8d}  [{flag_str}]")

    kept = idx.topk_precomputed(
        pid,
        k=TOPK,
        min_similarity=MIN_SIMILARITY,
        random_retrieval=False,
        filter_holdout=filter_holdout,
    )
    print(f"  → kept after all filters: {len(kept)}")
    for tid, s in kept:
        print(f"      OK {tid}  sim={s:.4f}")


def main():
    print(f"Loading FaissIndex from {INDEX_DIR}")
    holdout = load_holdout(HOLDOUT_FILES)
    print(f"Holdout chains loaded: {len(holdout)}")
    idx = FaissIndex(INDEX_DIR, holdout_ids=holdout)
    print(f"Index size: {idx.index.ntotal}   dim={idx.d}")

    # Eval-style (what val/test loaders actually use).
    for pid in PIDS_TO_PROBE:
        probe(idx, pid, filter_holdout=False)


if __name__ == "__main__":
    sys.exit(main() or 0)
