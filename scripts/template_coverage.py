"""Template-retrieval coverage: per-chain export + cluster-balanced summary.

Answers the reviewer question "for what fraction of evaluated chains did
retrieval actually produce a usable structural prior, and over how much of the
contact map". Note the distinction the paper must keep:

    retrieval returned a template   !=   a non-zero prior was produced
    a non-zero prior was produced   !=   the prior is correct

This script measures the middle quantity. It says nothing about whether the
projected contacts are right — that is what the retrieval gain and the
identity stratification measure.

**Long-range is reported separately.** A prior that is non-zero only in the
local band (|i-j| < 24) does not count as available for the headline task, so
`has_prior_long` / `prior_nz_frac_long` are computed over valid unique pairs
with |i-j| >= 24. `has_prior` uses the min_seq_sep set the loss sees.

Aggregation (2026-09-13): every summary carries BOTH the per-chain mean and the
**cluster-balanced** mean (per-cluster mean first, then unweighted over
clusters) — the same estimand as every headline metric. Chains whose
`cluster_id` is -1 are excluded from the cluster-balanced column and counted in
`n_unknown_cluster`, never pooled into one pseudo-cluster.

Outputs, both in the Hydra run dir:
    coverage_per_chain.tsv   one row per evaluated chain (re-aggregate offline
                             without rebuilding priors)
    coverage_summary.tsv     per (split, subset) and a split-level ALL rollup

Population completeness is checked against the dataset length and, unless
`coverage.allow_partial=true`, a mismatch raises. A loader failure raises: a
run that silently wrote validation only is worse than no run.

Knobs:
    coverage.max_batches=all      # default full scan; an int gives a quick pass
                                  #   (implies allow_partial)
    coverage.include_train=false  # opt in to a cluster-sampled train epoch
    coverage.allow_partial=false  # accept a short/partial scan
    coverage.verbose=false        # per-chain stdout dump

Usage (headline 650M, C=8, centre crop, val+test, no training):
    python scripts/template_coverage.py experiment=trufor_fusion_with_dist_650M \
        data.crop_mode=center data.num_workers=8

8M row (the config behind paper_8m_trufor_full_k4):
    python scripts/template_coverage.py experiment=ablation/trufor_fusion_with_dist \
        data.crop_mode=center data.num_workers=8
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TRAIN_SAMPLE_SPLIT = "train_epoch_sample"
LONG_RANGE_MIN_SEP = 24

PER_CHAIN_FIELDS = [
    "split", "sample_id", "subset", "cluster_id", "seq_len",
    "n_valid_pairs", "n_valid_long_pairs",
    "prior_nz", "prior_nz_long",
    "has_prior", "has_prior_long",
    "prior_nz_frac", "prior_nz_frac_long",
    "prior_max", "count_max",
    "n_templates_retrieved", "best_tpl_sim",
]

# Quantities averaged in the summary. Each becomes two columns: a per-chain
# mean and a cluster-balanced mean.
SUMMARY_FIELDS = [
    "has_prior", "has_prior_long",
    "prior_nz_frac", "prior_nz_frac_long",
    "n_templates_retrieved", "best_tpl_sim",
]


def _upper(L: int, device) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    return idx.unsqueeze(0) > idx.unsqueeze(1)


def _per_chain_rows(batch: Dict, split: str) -> List[Dict]:
    """One row per chain. Masks are restricted to unique pairs (j>i), matching
    the evaluation protocol; counting both triangles would double every pair."""
    rows: List[Dict] = []
    prior = batch["prior"][:, 0]
    count = batch["count"][:, 0]
    pair_mask = batch["pair_mask"].bool()
    long_mask = batch["long_mask"].bool()  # pair_mask AND |i-j| >= min_seq_sep
    n_tpl = batch.get("n_templates_retrieved")
    best_sim = batch.get("best_tpl_sim")
    subsets = batch.get("subset")
    cluster_ids = batch.get("cluster_id")
    seq_lens = batch.get("seq_len")
    B, Lmax = prior.shape[0], prior.shape[-1]
    up = _upper(Lmax, prior.device)
    sep = (torch.arange(Lmax, device=prior.device).unsqueeze(0)
           - torch.arange(Lmax, device=prior.device).unsqueeze(1)).abs()
    far = sep >= LONG_RANGE_MIN_SEP

    for b in range(B):
        valid = long_mask[b] & up
        valid_long = pair_mask[b] & far & up
        nz = int(((prior[b] != 0) & valid).sum())
        nz_long = int(((prior[b] != 0) & valid_long).sum())
        n_valid = int(valid.sum())
        n_valid_long = int(valid_long.sum())
        rows.append({
            "split": split,
            "sample_id": batch["pid"][b],
            "subset": subsets[b] if subsets is not None else "all",
            "cluster_id": int(cluster_ids[b]) if cluster_ids is not None else -1,
            "seq_len": int(seq_lens[b]) if seq_lens is not None else Lmax,
            "n_valid_pairs": n_valid,
            "n_valid_long_pairs": n_valid_long,
            "prior_nz": nz,
            "prior_nz_long": nz_long,
            "has_prior": int(nz > 0),
            # NaN, not 0, when the chain has no valid long-range pair at all:
            # it cannot be scored on the headline task either, so it must not
            # count as a coverage failure.
            "has_prior_long": int(nz_long > 0) if n_valid_long else float("nan"),
            "prior_nz_frac": nz / n_valid if n_valid else float("nan"),
            "prior_nz_frac_long": nz_long / n_valid_long if n_valid_long else float("nan"),
            "prior_max": float(prior[b].max()),
            "count_max": float(count[b].max()),
            "n_templates_retrieved": int(n_tpl[b]) if n_tpl is not None else float("nan"),
            "best_tpl_sim": float(best_sim[b]) if best_sim is not None else float("nan"),
        })
    return rows


def _finite(values: List[float]) -> List[float]:
    return [v for v in values if v == v]  # drops NaN


def _mean(values: List[float]) -> float:
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def _cluster_balanced(rows: List[Dict], field: str) -> float:
    """Per-cluster mean first, then unweighted over clusters. Unknown clusters
    are dropped, never merged."""
    per: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        cid = r["cluster_id"]
        if cid < 0:
            continue
        v = r[field]
        if v == v:  # not NaN
            per[cid].append(float(v))
    if not per:
        return float("nan")
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def _summarize(rows: List[Dict], split: str) -> List[Dict]:
    by_subset: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_subset[r["subset"]].append(r)
    groups = sorted(by_subset.items())
    if len(groups) > 1:
        groups.append(("ALL", rows))

    out: List[Dict] = []
    for subset, grp in groups:
        row = {
            "split": split,
            "subset": subset,
            "n_chains": len(grp),
            "n_clusters": len({r["cluster_id"] for r in grp if r["cluster_id"] >= 0}),
            "n_unknown_cluster": sum(1 for r in grp if r["cluster_id"] < 0),
            "n_long_defined": sum(1 for r in grp if r["n_valid_long_pairs"] > 0),
        }
        for f in SUMMARY_FIELDS:
            # Full precision on disk (the TSV is re-read, not just eyeballed);
            # rounding happens only in the stdout digest below.
            row[f"{f}_chain"] = _mean([r[f] for r in grp])
            row[f] = _cluster_balanced(grp, f)
        out.append(row)
    return out


def _run_loader(loader, dataset_len: Optional[int], max_batches: Optional[int],
                split: str, verbose: bool, allow_partial: bool) -> List[Dict]:
    scope = "ALL" if max_batches is None else f"first {max_batches}"
    print(f"\n=== {split} loader — {scope} batches ===", flush=True)
    rows: List[Dict] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch_rows = _per_chain_rows(batch, split)
        rows.extend(batch_rows)
        if verbose:
            for r in batch_rows:
                print(f"  [{split} b{i}] {r['sample_id']:<12} "
                      f"prior_nz={r['prior_nz']:>7d}/{r['n_valid_pairs']:<7d} "
                      f"long={r['prior_nz_long']:>7d}/{r['n_valid_long_pairs']:<7d} "
                      f"k={r['n_templates_retrieved']} sim={r['best_tpl_sim']:.4f}",
                      flush=True)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1} batches, {len(rows)} chains", flush=True)

    seen = {r["sample_id"] for r in rows}
    if len(seen) != len(rows):
        raise RuntimeError(
            f"{split}: {len(rows) - len(seen)} duplicate sample_ids in the scan"
        )
    if dataset_len is not None and len(rows) != dataset_len:
        msg = (f"{split}: scanned {len(rows)} chains but the dataset holds "
               f"{dataset_len}. A partial scan is not a coverage measurement.")
        if not allow_partial:
            raise RuntimeError(msg + " Pass coverage.allow_partial=true to accept it.")
        print(f"  WARNING {msg}", flush=True)
    print(f"  {split}: {len(rows)} chains scanned", flush=True)
    return rows


def _write_tsv(path: Path, rows: List[Dict], fields: Optional[List[str]] = None) -> None:
    fields = fields or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(repr(r[k]) if isinstance(r[k], float) else str(r[k])
                              for k in fields) + "\n")
    print(f"Wrote {path}  ({len(rows)} rows)", flush=True)


def _dataset_len(datamodule, attr: str) -> Optional[int]:
    ds = getattr(datamodule, attr, None)
    try:
        return len(ds) if ds is not None else None
    except TypeError:
        return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.get("seed", 42)))
    cov = cfg.get("coverage", {}) or {}

    mb_raw = cov.get("max_batches", "all")
    if mb_raw in (None, "all", "ALL") or (isinstance(mb_raw, int) and mb_raw <= 0):
        max_batches: Optional[int] = None
    else:
        max_batches = int(mb_raw)
    verbose = bool(cov.get("verbose", False))
    include_train = bool(cov.get("include_train", False))
    # A capped scan is partial by construction.
    allow_partial = bool(cov.get("allow_partial", False)) or max_batches is not None

    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    print(f"max_batches={mb_raw} include_train={include_train} "
          f"allow_partial={allow_partial} crop_mode={cfg.data.get('crop_mode')} "
          f"cap={cfg.data.get('max_chains_per_cluster')} topk={cfg.data.get('topk')}",
          flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    per_chain: List[Dict] = []
    summary: List[Dict] = []

    if include_train:
        print(f"  [{TRAIN_SAMPLE_SPLIT}] cluster-sampled training epoch "
              "(one chain per cluster) — not the full training set.", flush=True)
        rows = _run_loader(datamodule.train_dataloader(), None, max_batches,
                           TRAIN_SAMPLE_SPLIT, verbose, allow_partial=True)
        per_chain += rows
        summary += _summarize(rows, TRAIN_SAMPLE_SPLIT)

    # No try/except: if a loader fails, the run must fail. Writing validation
    # alone as if the scan had completed is the failure mode this replaces.
    for split, loader_fn, attr in (
        ("val", datamodule.val_dataloader, "dset_val"),
        ("test", datamodule.test_dataloader, "dset_test"),
    ):
        rows = _run_loader(loader_fn(), _dataset_len(datamodule, attr), max_batches,
                           split, verbose, allow_partial)
        per_chain += rows
        summary += _summarize(rows, split)

    try:
        from hydra.core.hydra_config import HydraConfig
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # noqa: BLE001
        out_dir = Path("results")

    _write_tsv(out_dir / "coverage_per_chain.tsv", per_chain, PER_CHAIN_FIELDS)
    _write_tsv(out_dir / "coverage_summary.tsv", summary)

    print("\nUnsuffixed columns are cluster-balanced; *_chain are per-chain.", flush=True)
    fmt = lambda v: "nan" if v != v else f"{v:.4f}"
    for r in summary:
        print(f"  [{r['split']}/{r['subset']}] n={r['n_chains']} K={r['n_clusters']} "
              f"has_prior={fmt(r['has_prior'])} (chain {fmt(r['has_prior_chain'])}) "
              f"has_prior_long={fmt(r['has_prior_long'])} (chain {fmt(r['has_prior_long_chain'])}) "
              f"nz_frac_long={fmt(r['prior_nz_frac_long'])}", flush=True)


if __name__ == "__main__":
    main()
