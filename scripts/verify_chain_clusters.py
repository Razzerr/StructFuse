#!/usr/bin/env python3
"""Check every chain's cluster label by a route that shares no code with the resolver.

`chain_clusters.tsv` decides the same-cluster filter, the cluster-balanced metrics and
the evaluation cap, and three defects have already lived in the code that derives it
(the last one merged chain IDs differing only by case and mislabelled 59,240 chains).
This script re-derives each label independently and fails on any disagreement:

  * chain -> entity from `_entity_poly.pdbx_strand_id`, read with gemmi instead of the
    resolver's own CIF tokeniser; chain IDs are never normalised;
  * entity -> cluster from the raw RCSB file, cluster id = position among non-empty
    lines, tokens compared exactly as `<PDB upper>_<entity>`.

It imports nothing from `resolve_chain_clusters.py`, on purpose: a shared helper would
carry a shared bug. Read-only.

Usage (inside an allocation):
    python scripts/verify_chain_clusters.py --chain-clusters data/output_splits_2026cb/chain_clusters.tsv \
        --cluster-file data/clusters_30_2026.txt --out-dir .temp/verify_chain_clusters --workers 32
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MMCIF = "/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2026/mmCIF"


def read_entity_clusters(path: Path) -> dict:
    """`12E8_2` -> cluster id, the position of its line among non-empty lines."""
    out = {}
    cid = 0
    with path.open() as fh:
        for line in fh:
            toks = line.split()
            if not toks:
                continue
            for tok in toks:
                out.setdefault(tok, cid)
            cid += 1
    return out


def chain_to_entity(cif_path: str) -> dict:
    """AUTH chain id -> entity id for one entry, with gemmi; case is preserved."""
    import gemmi

    block = gemmi.cif.read(cif_path).sole_block()
    mapping = {}
    for row in block.find("_entity_poly.", ["entity_id", "pdbx_strand_id"]):
        entity = gemmi.cif.as_string(row[0])
        for chain in gemmi.cif.as_string(row[1]).replace(" ", "").split(","):
            if chain:
                mapping[chain] = entity
    return mapping


def _worker(item: tuple) -> tuple:
    entry, cif_path = item
    try:
        return entry, chain_to_entity(cif_path), None
    except Exception as exc:  # noqa: BLE001 - reported per entry, fails the check
        return entry, {}, f"{type(exc).__name__}: {exc}"


def compare(labels: dict, by_entry: dict, entity_clusters: dict) -> tuple:
    """Rows of chains whose label differs from the independent derivation, plus counts."""
    rows, counts = [], Counter()
    for stem, label in labels.items():
        entry, _, chain = stem.partition("_")
        entity = by_entry.get(entry, {}).get(chain)
        if entity is None:
            expected, kind = -1, "chain_not_in_entity_poly"
        else:
            expected = entity_clusters.get(f"{entry.upper()}_{entity}", -1)
            kind = "entity_in_cluster_file" if expected >= 0 else "entity_not_in_cluster_file"
        counts[kind] += 1
        if expected != label:
            counts["mismatch"] += 1
            rows.append({"id": stem, "label": label, "independent": expected, "entity": entity or "", "kind": kind})
    return rows, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-clusters", type=Path, required=True)
    ap.add_argument("--cluster-file", type=Path, required=True)
    ap.add_argument("--mmcif-dir", type=Path, default=Path(DEFAULT_MMCIF))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    labels = {}
    with args.chain_clusters.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            labels[row["id"]] = int(row["cluster_id"])
    entries = sorted({s.partition("_")[0] for s in labels})
    items = [(e, str(args.mmcif_dir / e[1:3] / f"{e}.cif.gz")) for e in entries]
    print(f"{len(labels)} chains in {len(entries)} entries", flush=True)

    by_entry, failed = {}, []
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = pool.imap_unordered(_worker, items, chunksize=64)
            for k, (entry, mapping, err) in enumerate(results, 1):
                by_entry[entry] = mapping
                if err:
                    failed.append((entry, err))
                if k % 20000 == 0:
                    print(f"  parsed {k}/{len(items)}", flush=True)
    else:
        for item in items:
            entry, mapping, err = _worker(item)
            by_entry[entry] = mapping
            if err:
                failed.append((entry, err))

    rows, counts = compare(labels, by_entry, read_entity_clusters(args.cluster_file))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "mismatches.tsv").open("w") as fh:
        fh.write("id\tlabel\tindependent\tentity\tkind\n")
        for r in rows:
            fh.write(f"{r['id']}\t{r['label']}\t{r['independent']}\t{r['entity']}\t{r['kind']}\n")
    per_kind = defaultdict(int)
    for r in rows:
        per_kind[r["kind"]] += 1
    summary = {
        "chains": len(labels), "entries": len(entries), "parse_failures": len(failed),
        "failure_examples": failed[:10], "counts": dict(counts), "mismatches_by_kind": dict(per_kind),
        "status": "PASS" if not rows and not failed else "FAIL",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
