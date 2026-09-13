#!/usr/bin/env python3
"""Split composition: PDB entries vs chains vs clusters, in one table.

READ-ONLY. It opens `mmcif_final_splits.json`, the split id files,
`npz_lengths.json` and `chain_clusters.tsv` and counts. It never regenerates a
split and never writes into the split directory — re-deriving the split would
change it (`prepare_data_splits.py --limit_files` defaults to 100, which
silently produces a structurally valid but wrong split).

**The unit confusion this exists to kill.** Three different things get called
"the test set", and the numbers differ by an order of magnitude:

  1. SPLIT-LEVEL entries — every PDB entry assigned to a split in
     `mmcif_final_splits.json`, including entries with no usable chain.
  2. EVALUATED entries — the entries actually listed in `test_ids.txt` /
     `val_holdout_ids.txt`, i.e. what the pipeline consumes.
  3. CHAINS — what the model sees, after `min_len` and the skip-id files.

`promoted_train_to_test` in `cluster_split_stats.txt` is a count of **PDB
entries** at level (1): `prepare_data_splits.py` increments it once per entry
whose split flips from train to test, guarded by `== "train"` so an entry in
several clusters is not double-counted. Quoting it beside a level-(2) or (3)
figure is what produced the impossible "test 23,164 entries incl. 146,386
promoted entries".

**The holdout union is entry-level.** `val_holdout_ids.txt` and `test_ids.txt`
hold PDB entry ids, and `FaissIndex.__init__` normalises them through
`_get_protein_id` (`rsplit("_", 1)[0].lower()`, a no-op on a bare entry id).
That union is what `filter_holdout=True` blocks at training time, so its size
is a Methods number and is reported here as the filter builds it, not as a sum
of two files that are assumed disjoint (the disjointness is checked).

Usage (2026 generation):
    python scripts/split_composition.py \\
        --split-dir data/output_splits_2026 \\
        --npz-lengths data/processed_2026/npz_lengths.json \\
        --skip-ids data/corrupt_ids.txt \\
        --skip-ids data/output_splits_2026/no_cluster_ids.txt \\
        --out .temp/split_composition_2026.tsv

Every input is optional except `--split-dir`; a missing one drops its rows from
the table with an explicit note rather than a silent zero.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

MIN_LEN_DEFAULT = 20
ID_FILES = {
    "train": "all_train_ids.txt",
    "val": "val_holdout_ids.txt",
    "test": "test_ids.txt",
}
# Mirrors src/models/utils/faiss.py::_get_protein_id — kept here rather than
# imported so this script stays stdlib-only and runnable without torch/faiss.
def protein_id(chain_or_entry: str) -> str:
    return chain_or_entry.rsplit("_", 1)[0].lower()


def read_ids(path: Path) -> Set[str]:
    with path.open() as fh:
        return {line.strip().lower() for line in fh if line.strip()}


def read_stems(path: Path) -> Set[str]:
    """Skip-id files hold chain stems (`8tz6_B`), one per line."""
    with path.open() as fh:
        return {line.strip() for line in fh if line.strip()}


def load_chain_clusters(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            out[row["id"]] = int(row["cluster_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--splits-json", type=Path, default=None,
                    help="default: <split-dir>/mmcif_final_splits.json")
    ap.add_argument("--cluster-stats", type=Path, default=None,
                    help="default: <split-dir>/cluster_split_stats.txt, for reconciliation")
    ap.add_argument("--npz-lengths", type=Path, default=None,
                    help="chain rows are omitted without it")
    ap.add_argument("--chain-clusters", type=Path, default=None,
                    help="default: <split-dir>/chain_clusters.tsv; cluster rows need it")
    ap.add_argument("--skip-ids", type=Path, action="append", default=[],
                    help="repeatable; same list as data.skip_ids_files")
    ap.add_argument("--no-cluster-entries", type=Path, default=None,
                    help="default: <split-dir>/no_cluster_entries.txt; verifies "
                         "why entries are missing from the splits JSON")
    ap.add_argument("--min-len", type=int, default=MIN_LEN_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.resolve().parent == args.split_dir.resolve():
        raise SystemExit("Refusing to write into the split directory; this script is read-only.")

    splits_json = args.splits_json or args.split_dir / "mmcif_final_splits.json"
    cluster_stats = args.cluster_stats or args.split_dir / "cluster_split_stats.txt"
    chain_clusters_path = args.chain_clusters or args.split_dir / "chain_clusters.tsv"

    rows: List[Dict[str, object]] = []
    notes: List[str] = []

    def add(unit: str, quantity: str, **counts):
        rows.append({"unit": unit, "quantity": quantity, **counts})

    # ── 1. SPLIT-LEVEL entries ────────────────────────────────────────────
    meta = json.loads(splits_json.read_text())
    by_split = collections.Counter(v.get("split") for v in meta.values())
    promoted = collections.Counter(
        v.get("split") for v in meta.values() if v.get("cluster_promoted")
    )
    casp16 = collections.Counter(
        v.get("split") for v in meta.values() if v.get("casp16_test_set")
    )
    add("pdb_entry", "assigned to split", **by_split, total=len(meta))
    add("pdb_entry", "of which cluster-promoted", **promoted, total=sum(promoted.values()))
    add("pdb_entry", "of which CASP16-flagged", **casp16, total=sum(casp16.values()))

    # Reconcile the archived counter against the file it describes.
    if cluster_stats.exists():
        stats = json.loads(cluster_stats.read_text())
        recorded = stats.get("promoted_train_to_test")
        observed = sum(promoted.values())
        verdict = "MATCH" if recorded == observed else f"MISMATCH (json says {observed})"
        notes.append(
            f"cluster_split_stats.promoted_train_to_test = {recorded:,} vs "
            f"{observed:,} entries carrying cluster_promoted in the splits JSON — {verdict}. "
            f"clusters_with_test={stats.get('clusters_with_test'):,}, "
            f"clusters_train_only={stats.get('clusters_train_only'):,}."
        )
    else:
        notes.append(f"{cluster_stats} absent — promotion counter not reconciled.")

    # ── 2. EVALUATED entries + the holdout union the filter actually uses ──
    id_sets: Dict[str, Set[str]] = {}
    for split, fname in ID_FILES.items():
        p = args.split_dir / fname
        if p.exists():
            id_sets[split] = read_ids(p)
        else:
            notes.append(f"{p} absent — '{split}' entry/chain rows omitted.")
    if id_sets:
        add("pdb_entry", "listed in the split id file",
            **{k: len(v) for k, v in id_sets.items()},
            total=len(set().union(*id_sets.values())))

    # The promoted/evaluated relation is EXACT, not a proportion:
    # split_test_val.py keeps `pid in test_ids and not cluster_promoted`, so the
    # evaluated pool is the split-level test set minus the promoted entries.
    # Checking it here is what turns a units error into a visible MISMATCH.
    if {"val", "test"} <= set(id_sets):
        expected = by_split.get("test", 0) - promoted.get("test", 0)
        observed = len(id_sets["val"]) + len(id_sets["test"])
        verdict = "MATCH" if expected == observed else "MISMATCH"
        notes.append(
            f"Evaluated pool: split-level test {by_split.get('test', 0):,} "
            f"− cluster-promoted {promoted.get('test', 0):,} = {expected:,}; "
            f"val + test id files = {observed:,} — {verdict}. "
            "split_test_val.py excludes promoted entries outright, so these must agree."
        )

    if "val" in id_sets and "test" in id_sets:
        overlap = id_sets["val"] & id_sets["test"]
        union = {protein_id(x) for x in (id_sets["val"] | id_sets["test"])}
        add("pdb_entry", "HOLDOUT UNION used by filter_holdout",
            val=len(id_sets["val"]), test=len(id_sets["test"]), total=len(union))
        notes.append(
            f"Holdout union (val_holdout_ids ∪ test_ids, normalised by _get_protein_id) "
            f"= {len(union):,} entries; val∩test = {len(overlap):,}"
            + (" (disjoint, so the union is the sum)." if not overlap
               else " — NOT disjoint, do not quote the sum.")
        )

    # ── 3. CHAINS, by the dataset's own rule ──────────────────────────────
    if args.npz_lengths and args.npz_lengths.exists() and id_sets:
        index: Dict[str, int] = json.loads(args.npz_lengths.read_text())
        skip_stems: Set[str] = set()
        for p in args.skip_ids:
            if p.exists():
                skip_stems |= read_stems(p)
            else:
                notes.append(f"{p} absent — its chains were NOT skipped in this count.")
        # ContactDataset: pdb_id = stem.split("_")[0]; keep if in the id set and
        # L >= min_len; then drop skip stems.
        raw = collections.Counter()
        kept = collections.Counter()
        chains_by_split: Dict[str, List[str]] = collections.defaultdict(list)
        for stem, length in index.items():
            pdb_id = stem.split("_")[0].lower()
            for split, ids in id_sets.items():
                if pdb_id in ids:
                    raw[split] += 1
                    if length >= args.min_len and stem not in skip_stems:
                        kept[split] += 1
                        chains_by_split[split].append(stem)
        add("chain", f"in npz_lengths, before filters", **raw, total=sum(raw.values()))
        add("chain", f"after min_len>={args.min_len} and skip-id files",
            **kept, total=sum(kept.values()))

        # ── 4. CLUSTERS ───────────────────────────────────────────────────
        if chain_clusters_path.exists():
            chain2cluster = load_chain_clusters(chain_clusters_path)
            cl_counts, unknown = {}, {}
            for split, stems in chains_by_split.items():
                cids = [chain2cluster.get(s, -1) for s in stems]
                cl_counts[split] = len({c for c in cids if c >= 0})
                unknown[split] = sum(1 for c in cids if c < 0)
            add("cluster", "distinct, among kept chains", **cl_counts,
                total=len({chain2cluster.get(s, -1)
                           for stems in chains_by_split.values() for s in stems
                           if chain2cluster.get(s, -1) >= 0}))
            if any(unknown.values()):
                notes.append(
                    "chains with no cluster among the kept set: "
                    + ", ".join(f"{k}={v:,}" for k, v in unknown.items())
                    + " — expected 0 on the 2026 generation."
                )
        else:
            notes.append(f"{chain_clusters_path} absent — cluster rows omitted.")
    else:
        notes.append("--npz-lengths not given or absent — chain and cluster rows omitted.")

    # ── 5. Why entries are missing from the splits JSON ───────────────────
    # drop_unclustered_entries removes entries with NO chain carrying a cluster
    # assignment (prepare_data_splits.py:344). Partially clustered entries are
    # kept. A missing cluster is a curation rule, not a parsing failure — so
    # verify the mechanism instead of asserting it.
    nce = args.no_cluster_entries or args.split_dir / "no_cluster_entries.txt"
    if nce.exists():
        dropped = read_ids(nce)
        still_present = {e for e in dropped if e in meta}
        add("pdb_entry", "dropped: no chain with a cluster (no_cluster_entries)",
            total=len(dropped))
        notes.append(
            f"no_cluster_entries.txt lists {len(dropped):,} entries; "
            f"{len(dropped) - len(still_present):,} are absent from the splits JSON "
            f"as drop_unclustered_entries intends"
            + (f" — but {len(still_present):,} ARE present, so the drop did not "
               f"fully apply." if still_present else ".")
        )
    else:
        notes.append(f"{nce} absent — the reason entries are missing from the "
                     "splits JSON was not verified.")

    # ── output ────────────────────────────────────────────────────────────
    fields = ["unit", "quantity", "train", "val", "test", "discarded", "total"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})

    width = max(len(str(r["quantity"])) for r in rows) + 2
    print(f"{'unit':<11}{'quantity':<{width}}" + "".join(f"{c:>14}" for c in fields[2:]))
    print("-" * (11 + width + 14 * (len(fields) - 2)))
    for r in rows:
        cells = "".join(
            f"{r.get(c, ''):>14,}" if isinstance(r.get(c), int) else f"{'':>14}"
            for c in fields[2:]
        )
        print(f"{r['unit']:<11}{str(r['quantity']):<{width}}{cells}")
    print("\nNotes:")
    for n in notes:
        print(f"  - {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
