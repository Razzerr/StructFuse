#!/usr/bin/env python3
"""Report the cluster composition of an evaluation split, and how it responds
to capping the number of chains taken from each cluster.

Chains inside one RCSB 30%-identity cluster are near-duplicates by construction,
so a mean over chains is a weighted mean in which a large family votes many
times. Two numbers describe the damage:

  n_eff  Kish effective sample size under the worst case of perfect
         intra-cluster correlation, N^2 / sum(n_i^2). A confidence interval
         computed over chains claims N independent observations; the truth lies
         between n_eff and N.

  cap    Keeping at most `cap` chains per cluster. Raising the cap admits more
         chains but *lowers* n_eff, because the chains it admits are duplicates:
         at cap=1 every cluster contributes once and n_eff == N, and efficiency
         falls from there. Under the worst case (rho=1) cap=1 is therefore both
         the cheapest and the statistically strongest choice, and every extra
         chain is pure cost. Real rho is below 1, so extra chains do carry some
         information — the question a cap answers is how much redundancy is
         worth paying for. Read the curve as a cost/precision trade-off, not as
         a peak to locate.

This is benchmark composition, not a leak. No retrieval filter changes it.

Usage:
    python scripts/cluster_composition.py
    python scripts/cluster_composition.py --out-tsv .temp/cluster_composition.tsv
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent


def _load_verify_module():
    """Reuse verify_data_integrity's readers so both tools agree on parsing."""
    path = ROOT / "scripts" / "verify_data_integrity.py"
    spec = importlib.util.spec_from_file_location("verify_data_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_n(counts: Sequence[int]) -> float:
    """Kish effective sample size assuming perfect intra-cluster correlation."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return total * total / sum(c * c for c in counts)


def cluster_sizes(chains, chain2cluster: Dict[str, int]) -> List[int]:
    sizes: Dict[int, int] = {}
    for stem in chains:
        cid = chain2cluster.get(stem, -1)
        if cid != -1:
            sizes[cid] = sizes.get(cid, 0) + 1
    return sorted(sizes.values(), reverse=True)


def describe(counts: Sequence[int]) -> str:
    total = sum(counts)
    return (
        f"{total} chains / {len(counts)} clusters; "
        f"largest={counts[0]} ({100 * counts[0] / total:.1f}%), "
        f"top10={100 * sum(counts[:10]) / total:.1f}%, "
        f"median={counts[len(counts) // 2]}, "
        f"mean={total / len(counts):.1f}, "
        f"n_eff={effective_n(counts):.0f}"
    )


def within_multiplier(counts: Sequence[int], cap: Optional[int]) -> float:
    """m_C = mean over clusters of 1 / min(n_k, C).

    For the cluster-balanced estimator (cluster means averaged over clusters),
        Var = (1/K) * (sigma_between^2 + sigma_within^2 * m_C)
    so the between-cluster term is fixed and the cap moves only the second one.
    """
    if not counts:
        return 0.0
    return sum(1.0 / (min(c, cap) if cap else c) for c in counts) / len(counts)


def cap_curve(counts: Sequence[int], caps: Sequence[Optional[int]]) -> List[dict]:
    full = sum(counts)
    # Widest-case reference: evaluating every chain. sigma_within == sigma_between
    # is the conservative assumption — chains inside a 30%-identity cluster
    # almost certainly vary less than clusters do, which makes any cap look
    # better than this reports, never worse.
    m_full = within_multiplier(counts, None)
    rows = []
    for cap in caps:
        capped = [min(c, cap) if cap else c for c in counts]
        n = sum(capped)
        # A cluster is "whole" when the cap does not truncate it: the cluster
        # mean is then computed from every chain we have, and raising the cap
        # cannot improve it. This is what makes a particular cap defensible
        # rather than arbitrary.
        whole = sum(1 for c in counts if not cap or c <= cap)
        m_cap = within_multiplier(counts, cap)
        rows.append(
            {
                "cap": cap if cap else "full",
                "chains": n,
                "n_eff": effective_n(capped),
                "n_eff_frac": effective_n(capped) / n if n else 0.0,
                "chains_frac_of_full": n / full if full else 0.0,
                "clusters_whole": whole,
                "clusters_whole_frac": whole / len(counts) if counts else 0.0,
                # How much wider the cluster-balanced CI gets versus evaluating
                # every chain. This is the quantity a cap should be chosen on.
                "ci_width_vs_full": ((1.0 + m_cap) / (1.0 + m_full)) ** 0.5,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default="data/output_splits_2026")
    ap.add_argument("--splits", nargs="+",
                    default=["val_holdout_ids.txt", "test_ids.txt"])
    ap.add_argument("--chain-clusters-tsv", default=None,
                    help="Defaults to <split-dir>/chain_clusters.tsv")
    ap.add_argument("--no-cluster-ids", default=None,
                    help="Defaults to <split-dir>/no_cluster_ids.txt")
    ap.add_argument("--corrupt-ids", default="data/corrupt_ids.txt")
    ap.add_argument("--caps", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--out-tsv", default=None,
                    help="Also write the cap curve here, for the paper.")
    args = ap.parse_args()

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    split_dir = resolve(args.split_dir)
    tsv = resolve(args.chain_clusters_tsv) if args.chain_clusters_tsv \
        else split_dir / "chain_clusters.tsv"
    no_cluster = resolve(args.no_cluster_ids) if args.no_cluster_ids \
        else split_dir / "no_cluster_ids.txt"

    verify = _load_verify_module()

    chain2cluster: Dict[str, int] = {}
    with tsv.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        i_id, i_cl = header.index("id"), header.index("cluster_id")
        for line in handle:
            if line.strip():
                cols = line.rstrip("\n").split("\t")
                chain2cluster[cols[i_id]] = int(cols[i_cl])

    drop = verify.read_stems(no_cluster)
    corrupt_path = resolve(args.corrupt_ids)
    if corrupt_path.exists():
        drop |= verify.read_stems(corrupt_path)

    all_stems = set(chain2cluster)
    caps: List[Optional[int]] = list(args.caps) + [None]
    out_rows = []

    for split in args.splits:
        entries = verify.read_stems(split_dir / split)
        chains = verify.chain_stems_for_split(entries, all_stems) - drop
        counts = cluster_sizes(chains, chain2cluster)
        if not counts:
            print(f"\n== {split}: no clustered chains")
            continue

        print(f"\n== {split}")
        print(f"   {describe(counts)}")
        print(f"   {'cap':>6}{'chains':>10}{'n_eff':>9}{'n_eff/N':>10}"
              f"{'vs full':>9}{'whole clusters':>17}{'CI vs full':>12}")
        for row in cap_curve(counts, caps):
            print(
                f"   {str(row['cap']):>6}{row['chains']:>10}{row['n_eff']:>9.0f}"
                f"{100 * row['n_eff_frac']:>9.1f}%{100 * row['chains_frac_of_full']:>8.1f}%"
                f"{row['clusters_whole']:>10} ({100 * row['clusters_whole_frac']:>4.1f}%)"
                f"{100 * (row['ci_width_vs_full'] - 1):>+11.1f}%"
            )
            out_rows.append({"split": split, **row})

    if args.out_tsv:
        out = resolve(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["split", "cap", "chains", "n_eff", "n_eff_frac",
                "chains_frac_of_full", "clusters_whole", "clusters_whole_frac",
                "ci_width_vs_full"]
        with out.open("w") as handle:
            handle.write("\t".join(cols) + "\n")
            for row in out_rows:
                handle.write("\t".join(repr(row[c]) if isinstance(row[c], float)
                                       else str(row[c]) for c in cols) + "\n")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
