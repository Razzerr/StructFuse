"""
Diagnose template retrieval coverage across splits + emit the paper coverage table.

Default behaviour (paper-grade): scan the FULL val/test loaders,
aggregate per (split, subset), and write `coverage_summary.tsv` to the Hydra run
dir. The summary answers the reviewer question "what fraction of test chains had
a usable template, and how good were they": frac_with_prior / mean_n_templates /
mean_best_tpl_sim per subset (gold / casp16 / ...).

The optional training loader is cluster-sampled (one chain per cluster per
epoch), so it is reported as `train_epoch_sample`, never as a full-train scan.

Knobs (override on the CLI):
    coverage.max_batches=all   # default — full scan. Set an int (e.g. 5) for a quick sanity pass.
    coverage.verbose=false     # default — set true to also print per-chain prior/count lines.
    coverage.include_train=false # default — opt in to a cluster-sampled train epoch.

Usage:
    python scripts/template_coverage.py experiment=diagnostics/ceiling
    python scripts/template_coverage.py experiment=diagnostics/ceiling coverage.max_batches=5   # quick
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


def _summarize(batch: Dict, batch_idx: int, tag: str) -> None:
    """Verbose per-chain dump (opt-in via coverage.verbose=true)."""
    B = batch["contact"].shape[0]
    for b in range(B):
        pid = batch["pid"][b]
        valid = batch["pair_mask"][b].bool()
        prior = batch["prior"][b, 0]
        count = batch["count"][b, 0]
        esm = batch["esm_contacts"][b, 0]
        n_valid = int(valid.sum())
        prior_nz = int(((prior != 0) & valid).sum())
        print(
            f"  [{tag} batch{batch_idx} b{b}] pid={pid:<10} "
            f"valid_pairs={n_valid:>7d} "
            f"prior nz={prior_nz:>6d}/{n_valid:<6d} "
            f"pmax={float(prior.max()):.3f}  "
            f"count max={float(count.max()):.0f}  "
            f"esm nz={int(((esm != 0) & valid).sum()):>6d} emax={float(esm.max()):.3f}",
            flush=True,
        )


def _accumulate(batch: Dict, acc: Dict[str, Dict[str, float]]) -> None:
    """Fold one batch into per-subset running sums."""
    B = batch["contact"].shape[0]
    n_tpl = batch.get("n_templates_retrieved")
    best_sim = batch.get("best_tpl_sim")
    subsets = batch.get("subset", ["all"] * B)
    for b in range(B):
        subset = subsets[b] if b < len(subsets) else "all"
        valid = batch["pair_mask"][b].bool()
        denom = max(1, int(valid.sum()))
        prior_b = batch["prior"][b, 0]
        nz = int(((prior_b != 0) & valid).sum())
        has_prior = nz > 0
        s = acc[subset]
        s["n"] += 1
        s["n_with_prior"] += 1.0 if has_prior else 0.0
        s["sum_nz_frac"] += nz / denom
        if n_tpl is not None:
            s["sum_n_tpl"] += float(int(n_tpl[b]))
        if best_sim is not None:
            s["sum_best_sim"] += float(best_sim[b])


def _run_loader(loader, max_batches: Optional[int], tag: str, verbose: bool) -> List[Dict]:
    scope = "ALL" if max_batches is None else f"first {max_batches}"
    print(f"\n=== {tag} loader — {scope} batches ===", flush=True)
    acc: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if verbose:
            _summarize(batch, i, tag)
        _accumulate(batch, acc)
    rows: List[Dict] = []
    # Per-subset rows + a split-level "ALL" rollup.
    roll: Dict[str, float] = defaultdict(float)
    for subset, s in sorted(acc.items()):
        n = int(s["n"]) or 1
        rows.append({
            "split": tag,
            "subset": subset,
            "n_samples": int(s["n"]),
            "frac_with_prior": round(s["n_with_prior"] / n, 4),
            "mean_n_templates": round(s["sum_n_tpl"] / n, 3),
            "mean_best_tpl_sim": round(s["sum_best_sim"] / n, 4),
            "mean_prior_nz_frac": round(s["sum_nz_frac"] / n, 5),
        })
        for k in ("n", "n_with_prior", "sum_n_tpl", "sum_best_sim", "sum_nz_frac"):
            roll[k] += s[k]
    if len(acc) > 1:
        n = int(roll["n"]) or 1
        rows.append({
            "split": tag, "subset": "ALL", "n_samples": int(roll["n"]),
            "frac_with_prior": round(roll["n_with_prior"] / n, 4),
            "mean_n_templates": round(roll["sum_n_tpl"] / n, 3),
            "mean_best_tpl_sim": round(roll["sum_best_sim"] / n, 4),
            "mean_prior_nz_frac": round(roll["sum_nz_frac"] / n, 5),
        })
    for r in rows:
        print(
            f"  [{r['split']}/{r['subset']}] n={r['n_samples']} "
            f"with_prior={r['frac_with_prior']} mean_n_tpl={r['mean_n_templates']} "
            f"mean_best_sim={r['mean_best_tpl_sim']}",
            flush=True,
        )
    return rows


def _write_tsv(rows: List[Dict]) -> None:
    if not rows:
        return
    try:
        from hydra.core.hydra_config import HydraConfig
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "coverage_summary.tsv"
    headers = list(rows[0].keys())
    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(str(r[h]) for h in headers) + "\n")
    print(f"\nCoverage summary written to {path}", flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.get("seed", 42)))
    cov_cfg = cfg.get("coverage", {}) or {}

    # Default = full scan (paper-grade). Override with coverage.max_batches=<int>.
    mb_raw = cov_cfg.get("max_batches", "all")
    if mb_raw in (None, "all", "ALL") or (isinstance(mb_raw, int) and mb_raw <= 0):
        max_batches: Optional[int] = None
    else:
        max_batches = int(mb_raw)
    verbose = bool(cov_cfg.get("verbose", False))
    include_train = bool(cov_cfg.get("include_train", False))

    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    print(f"max_batches={mb_raw}  verbose={verbose}  include_train={include_train}", flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    all_rows: List[Dict] = []

    # The training loader is cluster-sampled: one chain per cluster for the
    # current sampler epoch. It is useful diagnostically, but is not the full
    # training set and must not be labelled as such in a paper table.
    if include_train and (datamodule.dset_trainval is not None or datamodule.dset_train is not None):
        print(
            f"  [{TRAIN_SAMPLE_SPLIT}] NOTE: cluster-sampled training epoch "
            "(one chain per cluster), not the full training set.",
            flush=True,
        )
        try:
            all_rows += _run_loader(
                datamodule.train_dataloader(), max_batches, TRAIN_SAMPLE_SPLIT, verbose
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{TRAIN_SAMPLE_SPLIT}] dataloader error: {e}", flush=True)

    all_rows += _run_loader(datamodule.val_dataloader(), max_batches, "val", verbose)

    try:
        all_rows += _run_loader(datamodule.test_dataloader(), max_batches, "test", verbose)
    except Exception as e:  # noqa: BLE001
        print(f"  [test] dataloader error: {e}", flush=True)

    _write_tsv(all_rows)


if __name__ == "__main__":
    main()
