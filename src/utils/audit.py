"""Audit-manifest dump for paper-grade test runs.

After `trainer.test(...)` finishes, write a single JSON manifest to
`.temp/audit/<run_id>/manifest.json` capturing the retrieval+model config
snapshot, holdout filter settings, and checkpoint identity. Companion to
`scripts/verify_no_leak.py` and `scripts/template_coverage.py` (which are
manually launched pre-train); this manifest fixes the post-hoc record of
*what was actually loaded for inference*. Used in supplementary Methods.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _resolve_run_id(trainer) -> str:
    """Best-effort run_id resolution: W&B → Lightning logger version → timestamp."""
    loggers = getattr(trainer, "loggers", None) or [getattr(trainer, "logger", None)]
    for lg in loggers:
        if lg is None:
            continue
        # W&B logger
        exp = getattr(lg, "experiment", None)
        if exp is not None and hasattr(exp, "id"):
            return str(exp.id)
        # Generic Lightning version
        version = getattr(lg, "version", None)
        if version is not None:
            return str(version)
    return time.strftime("local_%Y%m%d_%H%M%S")


def _safe_select(cfg: DictConfig, dotted: str, default: Any = None) -> Any:
    """Read `cfg.foo.bar.baz`-style key path, returning default on miss."""
    cur: Any = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        try:
            cur = cur[part] if part in cur else default
        except (TypeError, KeyError):
            return default
    return cur if cur is not None else default


def dump_audit_manifest(
    cfg: DictConfig,
    trainer,
    ckpt_path: Optional[str] = None,
    output_root: str = ".temp/audit",
) -> Path:
    """Write `manifest.json` with the retrieval+model snapshot and return its path."""
    run_id = _resolve_run_id(trainer)
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert holdout list to plain Python (OmegaConf ListConfig → list).
    holdout_files = _safe_select(cfg, "data.holdout_id_files")
    if holdout_files is not None:
        holdout_files = list(OmegaConf.to_container(holdout_files, resolve=True))

    # Convert fusion_feature_groups (DictConfig → dict) for JSON serialization.
    feature_groups = _safe_select(cfg, "model.fusion_feature_groups")
    if feature_groups is not None:
        try:
            feature_groups = dict(OmegaConf.to_container(feature_groups, resolve=True))
        except Exception:
            feature_groups = str(feature_groups)

    manifest = {
        "run_id": run_id,
        "task_name": _safe_select(cfg, "task_name"),
        "seed": _safe_select(cfg, "seed"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt_path": ckpt_path,
        "tags": list(OmegaConf.to_container(_safe_select(cfg, "tags", []) or [], resolve=True)),
        "retrieval": {
            "index_dir": _safe_select(cfg, "data.index_dir"),
            "topk": _safe_select(cfg, "data.topk"),
            "random_retrieval": _safe_select(cfg, "data.random_retrieval"),
            "min_template_similarity": _safe_select(cfg, "data.min_template_similarity"),
            "holdout_id_files": holdout_files,
            "esm_embeddings_dir": _safe_select(cfg, "data.esm_embeddings_dir"),
            "skip_ids_file": _safe_select(cfg, "data.skip_ids_file"),
            # filter_holdout policy is hard-wired in DataModule:
            # train collate uses True, val/test use False (see contact_lit_datamodule.py).
            "filter_holdout_train": True,
            "filter_holdout_eval": False,
        },
        "model": {
            "esm_model": _safe_select(cfg, "model.esm_model"),
            "head_type": _safe_select(cfg, "model.head_type"),
            "fusion_strategy": _safe_select(cfg, "model.fusion_strategy"),
            "fusion_feature_groups": feature_groups,
            "use_tpl_dist_bins": _safe_select(cfg, "model.use_tpl_dist_bins"),
            "triangle_c": _safe_select(cfg, "model.triangle_c"),
            "lr": _safe_select(cfg, "model.lr"),
            "warmup_steps": _safe_select(cfg, "model.warmup_steps"),
            "warmup_fraction": _safe_select(cfg, "model.warmup_fraction"),
            "use_tversky": _safe_select(cfg, "model.use_tversky"),
            "tversky_weight": _safe_select(cfg, "model.tversky_weight"),
            "lambda_disto": _safe_select(cfg, "model.lambda_disto"),
            "compile_model": _safe_select(cfg, "model.compile_model"),
        },
        "trainer": {
            "max_epochs": _safe_select(cfg, "trainer.max_epochs"),
            "limit_train_batches": _safe_select(cfg, "trainer.limit_train_batches"),
            "deterministic": _safe_select(cfg, "trainer.deterministic"),
            "accumulate_grad_batches": _safe_select(cfg, "trainer.accumulate_grad_batches"),
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    log.info(f"Audit manifest written: {manifest_path}")
    return manifest_path
