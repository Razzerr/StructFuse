#!/usr/bin/env python3
"""How many evaluated chains are the same protein as a training chain?

RCSB's 30 % clustering can put identical sequences into different clusters
(`8RO2_37` and `9FMD_41`: identical 240 residues, two singleton clusters), so a
cluster-disjoint split is not automatically sequence-disjoint. A training chain
identical to a test chain is information in the WEIGHTS, unlike a retrieved template.

Read-only. For every validation and test chain, against all training chains:
  1. exact: the same modelled sequence (NPZ `seq`);
  2. near: otherwise the training chains sharing the most k-mers are aligned with the
     audit's own `alignment_identity` (production local alignment), best one kept.
A pair counts as the same protein at identity >= --identity over aligned pairs with
coverage >= --coverage: "both" = of both chains, "either" = of at least one (also
catches a fragment of a longer construct). Only the k-mer-ranked candidates are
aligned, so the near counts are a lower bound; the exact counts are complete.

Chain population = ContactDataset's rule: entry in the split list, length >= min_len,
stem not in a skip list. Capped rows use the evaluation cap: first C sorted stems per
cluster, CASP16 exempt. Shares come as chains and cluster-balanced (clusters weighted
equally), as everywhere in the paper.

Usage (inside an allocation):
    python scripts/sequence_overlap_audit.py --processed-dir data/processed_2026cb \
        --split-dir data/output_splits_2026cb --out-dir .temp/seq_overlap --workers 16 \
        --skip-ids-files data/corrupt_ids.txt data/output_splits_2026cb/no_cluster_ids.txt \
        data/output_splits_2026cb/low_cb_ids.txt
"""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_identity_audit import alignment_identity  # noqa: E402


def read_lines(path: Path) -> set:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def select_chains(lengths: dict, entries: set, skip: set, min_len: int) -> list:
    """ContactDataset's population rule: entry in the list, L >= min_len, not skipped."""
    return sorted(s for s, L in lengths.items()
                  if s.split("_")[0].lower() in entries and L >= min_len and s not in skip)


def apply_cap(stems: list, clusters: dict, cap: int, exempt: set) -> set:
    """Evaluation cap: first `cap` sorted stems per cluster; exempt stems always stay."""
    kept, seen = set(), Counter()
    for stem in sorted(stems):
        if stem in exempt:
            kept.add(stem)
            continue
        cid = clusters[stem]
        if seen[cid] < cap:
            seen[cid] += 1
            kept.add(stem)
    return kept


def _read_seq(item: tuple) -> tuple:
    stem, root = item
    with np.load(Path(root) / f"{stem}.npz", allow_pickle=True) as z:
        s = z["seq"]
        return stem, str(s.item()) if s.shape == () else str(s)


def kmers(seq: str, k: int) -> set:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def build_kmer_index(seqs: list, k: int, max_posting: int) -> dict:
    """k-mer -> indices into `seqs`. Very common k-mers (tags, repeats) are dropped."""
    index = defaultdict(list)
    for i, seq in enumerate(seqs):
        for km in kmers(seq, k):
            index[km].append(i)
    return {km: ids for km, ids in index.items() if len(ids) <= max_posting}


# Set in main() before the pool forks, so workers share them copy-on-write.
_G: dict = {}


def best_near_match(query: str) -> dict:
    """Best training sequence among the k-mer-ranked candidates.

    Candidates are ranked by (shared k-mers, index): `kmers` returns a set whose order
    depends on hash randomisation, so without the index a tie in the count could pick
    different candidates in different runs. The kept one is the first that meets the
    most of the rule: both coverages, then either, then simply the highest identity.
    A row keeps one candidate, so a high-identity fragment must not hide another
    candidate that passes the rule.
    """
    seqs, index, k, top = _G["seqs"], _G["index"], _G["k"], _G["top"]
    identity, coverage = _G["identity"], _G["coverage"]
    counts = Counter()
    for km in kmers(query, k):
        counts.update(index.get(km, ()))
    best = {"seq": query, "train_idx": -1, "identity": float("nan"), "query_cov": float("nan"),
            "train_cov": float("nan")}
    best_key = None
    for idx, _ in heapq.nsmallest(top, counts.items(), key=lambda kv: (-kv[1], kv[0])):
        stats = alignment_identity(query, seqs[idx])
        row = {"exact": False, "identity": stats["seq_identity_aligned"],
               "query_cov": stats["query_coverage"], "train_cov": stats["template_coverage"]}
        key = (same_protein(row, identity, coverage, "both"), same_protein(row, identity, coverage, "either"),
               row["identity"])
        if best_key is None or key > best_key:
            best_key = key
            best.update(train_idx=idx, identity=row["identity"], query_cov=row["query_cov"], train_cov=row["train_cov"])
    return best


def cluster_balanced(flags: dict, clusters: dict) -> float:
    per = defaultdict(list)
    for stem, flag in flags.items():
        per[clusters[stem]].append(float(flag))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def same_protein(r: dict, identity: float, coverage: float, mode: str = "either") -> bool:
    if r["exact"]:
        return True
    cov_q, cov_t = r["query_cov"], r["train_cov"]
    if not (r["identity"] >= identity):  # NaN (no candidate) is False
        return False
    return (cov_q >= coverage and cov_t >= coverage) if mode == "both" else (cov_q >= coverage or cov_t >= coverage)


def summarize(rows: list, clusters: dict, identity: float, coverage: float) -> dict:
    def same(r, mode):
        return same_protein(r, identity, coverage, mode)

    out = {"chains": len(rows), "clusters": len({clusters[r["stem"]] for r in rows})}
    tests = {"exact": lambda r: r["exact"], "same_protein_both_cov": lambda r: same(r, "both"),
             "same_protein_either_cov": lambda r: same(r, "either")}
    for name, fn in tests.items():
        flags = {r["stem"]: fn(r) for r in rows}
        n = sum(flags.values())
        out[name] = {"chains": int(n), "share_chains": n / len(rows) if rows else float("nan"),
                     "share_cluster_balanced": cluster_balanced(flags, clusters),
                     "clusters_with_any": len({clusters[s] for s, f in flags.items() if f})}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--skip-ids-files", nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--identity", type=float, default=0.95)
    ap.add_argument("--coverage", type=float, default=0.9)
    ap.add_argument("--kmer", type=int, default=10)
    ap.add_argument("--top-candidates", type=int, default=100)
    ap.add_argument("--max-posting", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-eval-chains", type=int, default=0, help="smoke runs only; 0 = all")
    ap.add_argument("--out-skip-list", type=Path, default=None,
                    help="write the val/test chains meeting the rule ('either' coverage), one stem per "
                         "line: they leave evaluation via data.skip_ids_files (decided 2026-11-05)")
    args = ap.parse_args()
    if args.out_skip_list is not None and args.out_skip_list.exists():
        raise SystemExit(f"{args.out_skip_list} exists; refusing to overwrite")
    if args.out_skip_list is not None and args.max_eval_chains:
        raise SystemExit("--out-skip-list needs the full population, not a smoke run")

    split = args.split_dir
    lengths = json.loads((args.processed_dir / "npz_lengths.json").read_text())
    skip = set()
    for raw in args.skip_ids_files:
        if not Path(raw).exists():
            raise SystemExit(f"skip ids file not found: {raw}")
        skip |= read_lines(Path(raw))
    train = select_chains(lengths, read_lines(split / "all_train_ids.txt"), skip, args.min_len)
    val = select_chains(lengths, read_lines(split / "val_holdout_ids.txt"), skip, args.min_len)
    test = select_chains(lengths, read_lines(split / "test_ids.txt"), skip, args.min_len)
    casp_entries = read_lines(split / "test_casp16_ids.txt")
    clusters = {}
    with (split / "chain_clusters.tsv").open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_id, i_c = header.index("id"), header.index("cluster_id")
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            clusters[cols[i_id]] = int(cols[i_c])
    if args.max_eval_chains:
        val, test = val[: args.max_eval_chains], test[: args.max_eval_chains]
    bad = [s for s in train + val + test if clusters.get(s, -1) < 0]
    if bad:
        raise SystemExit(f"{len(bad)} chains without a cluster (missing skip list?), e.g. {bad[:5]}")
    print(f"chains: train {len(train)}, val {len(val)}, test {len(test)}", flush=True)

    stems = train + val + test
    items = [(s, str(args.processed_dir)) for s in stems]
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            seq_of = dict(pool.imap_unordered(_read_seq, items, chunksize=256))
    else:
        seq_of = dict(map(_read_seq, items))

    train_first = {}
    for stem in train:  # sorted, so the representative of a sequence is reproducible
        train_first.setdefault(seq_of[stem], stem)
    train_seqs = list(train_first)
    eval_stems = val + test
    eval_seqs = {seq_of[s] for s in eval_stems}
    queries = sorted(eval_seqs - set(train_first))
    print(f"unique sequences: train {len(train_seqs)}, eval {len(eval_seqs)}, "
          f"eval without an exact training match {len(queries)}", flush=True)

    _G.update(seqs=train_seqs, index=build_kmer_index(train_seqs, args.kmer, args.max_posting),
              k=args.kmer, top=args.top_candidates, identity=args.identity, coverage=args.coverage)
    print(f"k-mer index: {len(_G['index'])} {args.kmer}-mers", flush=True)
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            found = list(pool.imap_unordered(best_near_match, queries, chunksize=64))
    else:
        found = [best_near_match(q) for q in queries]
    near = {f["seq"]: f for f in found}

    casp = {s for s in test if s.split("_")[0].lower() in casp_entries}
    capped = {"val": apply_cap(val, clusters, args.cap, set()), "test": apply_cap(test, clusters, args.cap, casp)}
    rows = []
    for name, group in (("val", val), ("test", test)):
        for stem in group:
            seq = seq_of[stem]
            exact = seq in train_first
            f = near.get(seq, {})
            idx = f.get("train_idx", -1)
            rows.append({
                "stem": stem, "split": name, "cluster_id": clusters[stem], "capped": stem in capped[name],
                "casp16": stem in casp, "L": len(seq), "exact": exact,
                "train_stem": train_first[seq] if exact else (train_first[train_seqs[idx]] if idx >= 0 else ""),
                "identity": 1.0 if exact else f.get("identity", float("nan")),
                "query_cov": 1.0 if exact else f.get("query_cov", float("nan")),
                "train_cov": 1.0 if exact else f.get("train_cov", float("nan")),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with (args.out_dir / "per_chain.tsv").open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(repr(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    populations = {
        "val": [r for r in rows if r["split"] == "val"],
        "val_capped": [r for r in rows if r["split"] == "val" and r["capped"]],
        "test": [r for r in rows if r["split"] == "test"],
        "test_capped": [r for r in rows if r["split"] == "test" and r["capped"]],
        "casp16": [r for r in rows if r["casp16"]],
    }
    eval_exact = {seq_of[s] for s in eval_stems} & set(train_first)
    touched = [s for s in train if seq_of[s] in eval_exact]
    summary = {
        "processed_dir": str(args.processed_dir), "split_dir": str(args.split_dir),
        "skip_ids_files": list(args.skip_ids_files),
        "rule": {"identity": args.identity, "coverage": args.coverage, "kmer": args.kmer,
                 "top_candidates": args.top_candidates, "max_posting": args.max_posting},
        "near_counts_are_a_lower_bound": True,
        "chains": {"train": len(train), "val": len(val), "test": len(test)},
        "unique_sequences": {"train": len(train_seqs), "eval": len(eval_seqs)},
        "populations": {k: summarize(v, clusters, args.identity, args.coverage) for k, v in populations.items()},
        "train_chains_with_an_exact_eval_copy": {"chains": len(touched),
                                                 "clusters": len({clusters[s] for s in touched})},
        "partial_smoke_run": bool(args.max_eval_chains),
    }
    if args.out_skip_list is not None:
        listed = sorted(r["stem"] for r in rows if same_protein(r, args.identity, args.coverage))
        args.out_skip_list.write_text("".join(f"{s}\n" for s in listed))
        summary["skip_list"] = {"path": str(args.out_skip_list), "chains": len(listed)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["populations"], indent=2))
    print(json.dumps(summary["train_chains_with_an_exact_eval_copy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
