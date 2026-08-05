#!/usr/bin/env python3
"""Gate the rebuilt data pipeline before a single GPU-hour is spent.

Everything downstream — the split guarantee, the same-cluster retrieval filter,
the paper's leakage claims — rests on two properties that are cheap to check and
expensive to discover late:

  * every chain the pipeline can load has a real sequence-cluster assignment, and
  * train, val and test are disjoint not only chain-by-chain but *cluster-wise*,
    so no training structure is a 30 %-identity homolog of an evaluated one.

The second check is the one the split never had. It is what turns "no structural
leak between the sets" from an assumption into a verified statement.

Run after `resolve_chain_clusters.py`, the split regeneration and the index
rebuild; `verify_no_leak.py` covers the remaining train-time retrieval check.

    python scripts/verify_data_integrity.py \
        --index-dirs data/index_t33 data/index_t6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]


class Report:
    """Collects pass/fail lines so every check runs before the script exits."""

    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def info(self, message: str) -> None:
        print(f"       {message}")


def read_stems(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def read_ids_json(index_dir: Path) -> List[dict]:
    with (index_dir / "ids.json").open() as handle:
        return json.load(handle)


def chain_stems_for_split(split_ids: Set[str], all_stems: Set[str]) -> Set[str]:
    """Split files list PDB entries; datasets expand them to chain stems."""
    by_entry: Dict[str, Set[str]] = {}
    for stem in all_stems:
        by_entry.setdefault(stem.split("_")[0].lower(), set()).add(stem)
    out: Set[str] = set()
    for pid in split_ids:
        out |= by_entry.get(pid.lower(), set())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dirs", nargs="+", default=["data/index_t33", "data/index_t6"])
    ap.add_argument("--chain-clusters-tsv", default="data/output_splits/chain_clusters.tsv")
    ap.add_argument("--no-cluster-ids", default="data/no_cluster_ids.txt")
    ap.add_argument("--corrupt-ids", default="data/corrupt_ids.txt")
    ap.add_argument("--train-ids", default="data/output_splits/all_train_ids.txt")
    ap.add_argument("--val-ids", default="data/output_splits/val_holdout_ids.txt")
    ap.add_argument("--test-ids", default="data/output_splits/test_ids.txt")
    ap.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Also load embeddings.npy and faiss.index to verify row alignment "
             "(several GB per index; skip for a fast metadata-only pass).",
    )
    args = ap.parse_args()

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    rep = Report()

    # ── chain → cluster map ────────────────────────────────────────────────
    tsv = resolve(args.chain_clusters_tsv)
    chain2cluster: Dict[str, int] = {}
    with tsv.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        i_id, i_cl = header.index("id"), header.index("cluster_id")
        for line in handle:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            chain2cluster[cols[i_id]] = int(cols[i_cl])
    excluded = read_stems(resolve(args.no_cluster_ids))
    corrupt = read_stems(resolve(args.corrupt_ids)) if resolve(args.corrupt_ids).exists() else set()
    rep.info(
        f"{len(chain2cluster)} chains in {args.chain_clusters_tsv}; "
        f"{len(excluded)} unclustered; {len(corrupt)} corrupt"
    )

    unresolved = {s for s, c in chain2cluster.items() if c == -1}
    rep.check(
        unresolved == excluded,
        "no_cluster_ids.txt matches the -1 rows of chain_clusters.tsv",
        f"tsv={len(unresolved)} file={len(excluded)}",
    )

    # ── splits ─────────────────────────────────────────────────────────────
    train_e, val_e, test_e = (
        read_stems(resolve(p)) for p in (args.train_ids, args.val_ids, args.test_ids)
    )
    all_stems = set(chain2cluster)
    train_c = chain_stems_for_split(train_e, all_stems) - excluded - corrupt
    val_c = chain_stems_for_split(val_e, all_stems) - excluded - corrupt
    test_c = chain_stems_for_split(test_e, all_stems) - excluded - corrupt
    rep.info(f"chains after exclusion — train={len(train_c)} val={len(val_c)} test={len(test_c)}")

    # (1) nothing excluded may survive anywhere
    for name, chains in (("train", train_c), ("val", val_c), ("test", test_c)):
        leaked = chains & excluded
        rep.check(not leaked, f"no unclustered chain in {name}", f"{len(leaked)} found")

    # (4) chain-level disjointness
    for a_name, a, b_name, b in (
        ("train", train_c, "val", val_c),
        ("train", train_c, "test", test_c),
        ("val", val_c, "test", test_c),
    ):
        overlap = a & b
        rep.check(
            not overlap,
            f"{a_name} ∩ {b_name} == ∅ (chains)",
            f"{len(overlap)} shared, e.g. {sorted(overlap)[:3]}",
        )

    # (5) cluster-level disjointness — the check the split never had
    def clusters_of(chains: Set[str]) -> Set[int]:
        return {chain2cluster[s] for s in chains if chain2cluster.get(s, -1) != -1}

    train_cl, val_cl, test_cl = clusters_of(train_c), clusters_of(val_c), clusters_of(test_c)
    rep.info(f"clusters — train={len(train_cl)} val={len(val_cl)} test={len(test_cl)}")
    for a_name, a, b_name, b in (
        ("train", train_cl, "val", val_cl),
        ("train", train_cl, "test", test_cl),
    ):
        overlap = a & b
        rep.check(
            not overlap,
            f"{a_name} ∩ {b_name} == ∅ (clusters)",
            f"{len(overlap)} shared clusters, e.g. {sorted(overlap)[:5]}",
        )

    # ── indexes ────────────────────────────────────────────────────────────
    for raw in args.index_dirs:
        index_dir = resolve(raw)
        if not (index_dir / "ids.json").exists():
            rep.check(False, f"{raw}/ids.json exists")
            continue
        meta = read_ids_json(index_dir)
        stems = [m["id"] for m in meta]

        # (2) zero -1 in the index
        bad = [m["id"] for m in meta if int(m.get("cluster_id", -1)) == -1]
        rep.check(
            not bad,
            f"{raw}: every ids.json row has cluster_id >= 0",
            f"{len(bad)} unclustered, e.g. {bad[:3]}",
        )

        # (1, index side) excluded chains must not be indexed
        indexed_excluded = set(stems) & excluded
        rep.check(
            not indexed_excluded,
            f"{raw}: no unclustered chain indexed",
            f"{len(indexed_excluded)} found",
        )

        # metadata must agree with the resolver
        mismatched = [
            m["id"]
            for m in meta
            if m["id"] in chain2cluster and int(m["cluster_id"]) != chain2cluster[m["id"]]
        ]
        rep.check(
            not mismatched,
            f"{raw}: cluster_id agrees with chain_clusters.tsv",
            f"{len(mismatched)} disagree, e.g. {mismatched[:3]}",
        )

        rep.check(len(set(stems)) == len(stems), f"{raw}: ids.json has no duplicate ids")

        # (3) row alignment across the three artifacts
        if args.check_embeddings:
            import faiss  # noqa: PLC0415 — heavy, only needed for this check
            import numpy as np  # noqa: PLC0415

            emb = np.load(index_dir / "embeddings.npy", mmap_mode="r")
            idx = faiss.read_index(str(index_dir / "faiss.index"))
            rep.check(
                len(meta) == emb.shape[0] == idx.ntotal,
                f"{raw}: ids.json / embeddings.npy / faiss.index row counts match",
                f"ids={len(meta)} emb={emb.shape[0]} faiss={idx.ntotal}",
            )
        else:
            rep.info(f"{raw}: row-count check skipped (pass --check-embeddings)")

    print()
    if rep.failures:
        print(f"FAILED {len(rep.failures)} check(s): {rep.failures}")
        return 1
    print("All data-integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
