"""
Diagnose template retrieval coverage across splits.

For each of {train, val, test} iterate a handful of batches and print
per-sample prior/count statistics. Goal: find out whether the "prior is all
zeros" symptom seen by scripts/ceiling.py on the val set is specific to val
or universal.

Usage:
    python scripts/template_coverage.py experiment=diagnostics/ceiling
"""

from __future__ import annotations

from typing import Dict

import hydra
import torch
from omegaconf import DictConfig

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _summarize(batch: Dict, batch_idx: int, tag: str) -> None:
    B = batch["contact"].shape[0]
    for b in range(B):
        pid = batch["pid"][b]
        Lm = int(batch["pair_mask"][b].sum(dim=1).gt(0).sum().item())
        prior = batch["prior"][b, 0, :Lm, :Lm]
        count = batch["count"][b, 0, :Lm, :Lm]
        esm = batch["esm_contacts"][b, 0, :Lm, :Lm]
        print(
            f"  [{tag} batch{batch_idx} b{b}] pid={pid:<10} L={Lm:4d} "
            f"prior nz={int((prior != 0).sum()):>6d}/{Lm*Lm:<6d} "
            f"pmax={float(prior.max()):.3f}  "
            f"count max={float(count.max()):.0f}  "
            f"esm nz={int((esm != 0).sum()):>6d} emax={float(esm.max()):.3f}",
            flush=True,
        )


def _run_loader(loader, max_batches: int, tag: str) -> None:
    print(f"\n=== {tag} loader — first {max_batches} batches ===", flush=True)
    n_prior_nz = 0
    n_total_samples = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        _summarize(batch, i, tag)
        # Aggregate: count samples where prior has any non-zero pair.
        for b in range(batch["contact"].shape[0]):
            n_total_samples += 1
            if float(batch["prior"][b].abs().sum()) > 0:
                n_prior_nz += 1
    print(f"  [{tag}] samples with any prior≠0: {n_prior_nz}/{n_total_samples}", flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.get("seed", 42)))
    cov_cfg = cfg.get("coverage", {}) or {}
    max_batches = int(cov_cfg.get("max_batches", 5))

    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    # train — only if cluster_sampler / trainval is available
    if datamodule.dset_trainval is not None or datamodule.dset_train is not None:
        try:
            train_loader = datamodule.train_dataloader()
            _run_loader(train_loader, max_batches, tag="train")
        except Exception as e:  # noqa: BLE001
            print(f"  [train] dataloader error: {e}", flush=True)

    val_loader = datamodule.val_dataloader()
    _run_loader(val_loader, max_batches, tag="val")

    try:
        test_loader = datamodule.test_dataloader()
        _run_loader(test_loader, max_batches, tag="test")
    except Exception as e:  # noqa: BLE001
        print(f"  [test] dataloader error: {e}", flush=True)


if __name__ == "__main__":
    main()
