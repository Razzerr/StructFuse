"""Audit-manifest dump for paper-grade test runs.

After `trainer.test(...)` finishes, write a single JSON manifest to
`.temp/audit/<run_id>/manifest.json` capturing the retrieval+model config
snapshot, holdout filter settings, and checkpoint identity. Companion to
`scripts/verify_no_leak.py` and `scripts/template_coverage.py` (which are
manually launched pre-train); this manifest fixes the post-hoc record of
*what was actually loaded for inference*. Used in supplementary Methods.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

_HASH_CHUNK_SIZE = 1024 * 1024
_FULL_HASH_LIMIT = 64 * 1024 * 1024


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


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return list(value)


def _git_identity() -> tuple[Optional[str], Optional[bool]]:
    """Resolve the launched commit and whether its checkout had local changes."""
    commit = os.environ.get("STRUCTFUSE_GIT_COMMIT")
    try:
        if commit is None:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return commit, None


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_sha256(path: Path, size: int) -> str:
    """Stable low-I/O fingerprint for large immutable artifacts."""
    digest = hashlib.sha256()
    digest.update(f"size={size}".encode("ascii"))
    offsets = sorted({
        0,
        max(0, size // 2 - _HASH_CHUNK_SIZE // 2),
        max(0, size - _HASH_CHUNK_SIZE),
    })
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(_HASH_CHUNK_SIZE)
            digest.update(f"offset={offset};len={len(chunk)}".encode("ascii"))
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(
    path_value: Optional[str | Path],
    *,
    force_full_hash: bool = False,
) -> Optional[dict[str, Any]]:
    """Describe a file without failing the run when an optional path is absent."""
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    identity: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return identity
    if not path.is_file():
        identity["type"] = "directory" if path.is_dir() else "other"
        stat = path.stat()
        identity.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return identity

    stat = path.stat()
    identity.update({
        "type": "file",
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    })
    if force_full_hash or stat.st_size <= _FULL_HASH_LIMIT:
        identity["fingerprint_type"] = "sha256"
        identity["fingerprint"] = _sha256_stream(path)
    else:
        identity["fingerprint_type"] = "sampled_sha256_head_mid_tail_1MiB"
        identity["fingerprint"] = _sampled_sha256(path, stat.st_size)
    return identity


def _ids_json_identity(index_dir: Optional[str | Path]) -> Optional[dict[str, Any]]:
    if index_dir is None:
        return None
    path = (Path(index_dir) / "ids.json").expanduser().resolve()
    identity = _file_identity(path, force_full_hash=True)
    if identity is None or not identity.get("exists"):
        return identity

    # build_index.py writes an indented JSON list with one `"id"` field per
    # entry. Count those fields in a streaming pass to avoid loading a large
    # metadata list into memory solely for the audit.
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.lstrip().startswith(b'"id"'):
                count += 1
    identity["entry_count"] = count
    identity["entry_count_method"] = "lines_starting_with_id_key"
    return identity


def _index_identity(index_dir: Optional[str | Path]) -> Optional[dict[str, Any]]:
    if index_dir is None:
        return None
    root = Path(index_dir).expanduser().resolve()
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "ids_json": _ids_json_identity(root),
        "faiss_index": _file_identity(root / "faiss.index"),
        "embeddings": _file_identity(root / "embeddings.npy"),
    }


def _split_identities(cfg: DictConfig) -> dict[str, Any]:
    split_dir = Path(_safe_select(cfg, "data.split_dir", "data/output_splits"))
    train_path = _safe_select(cfg, "data.train_ids") or split_dir / "all_train_ids.txt"
    holdout_paths = _plain_list(_safe_select(cfg, "data.holdout_id_files"))
    skip_paths = _plain_list(_safe_select(cfg, "data.skip_ids_files"))
    return {
        "train": _file_identity(train_path, force_full_hash=True),
        "validation": _file_identity(
            _safe_select(cfg, "data.val_ids"), force_full_hash=True
        ),
        "test": _file_identity(
            _safe_select(cfg, "data.test_ids"), force_full_hash=True
        ),
        "subset_membership": _file_identity(
            _safe_select(cfg, "data.splits_json_path"), force_full_hash=True
        ),
        # The evaluation cap changes WHICH chains are scored, so the chain ->
        # cluster file it reads is part of the result's identity.
        "chain_clusters": _file_identity(
            _safe_select(cfg, "data.chain_clusters_file"), force_full_hash=True
        ),
        "holdout_filters": [
            _file_identity(path, force_full_hash=True) for path in holdout_paths
        ],
        "skip_ids": [
            _file_identity(path, force_full_hash=True) for path in skip_paths
        ],
    }


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
    git_commit, git_dirty = _git_identity()

    # Convert holdout list to plain Python (OmegaConf ListConfig → list).
    holdout_files = _safe_select(cfg, "data.holdout_id_files")
    if holdout_files is not None:
        holdout_files = _plain_list(holdout_files)
    skip_files = _safe_select(cfg, "data.skip_ids_files")
    if skip_files is not None:
        skip_files = _plain_list(skip_files)

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
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "ckpt_path": ckpt_path,
        "checkpoint_identity": _file_identity(ckpt_path),
        "tags": _plain_list(_safe_select(cfg, "tags")),
        "data": {
            "data_root": _safe_select(cfg, "data.data_root"),
            "esm_embeddings_dir": _file_identity(
                _safe_select(cfg, "data.esm_embeddings_dir")
            ),
            "splits": _split_identities(cfg),
        },
        "retrieval": {
            "index_dir": _safe_select(cfg, "data.index_dir"),
            "index_identity": _index_identity(_safe_select(cfg, "data.index_dir")),
            "topk": _safe_select(cfg, "data.topk"),
            "random_retrieval": _safe_select(cfg, "data.random_retrieval"),
            "min_template_similarity": _safe_select(cfg, "data.min_template_similarity"),
            "holdout_id_files": holdout_files,
            "esm_embeddings_dir": _safe_select(cfg, "data.esm_embeddings_dir"),
            "skip_ids_files": skip_files,
            # filter_holdout policy is hard-wired in DataModule:
            # train collate uses True, val/test use False (see contact_lit_datamodule.py).
            "filter_holdout_train": True,
            "filter_holdout_eval": False,
        },
        "evaluation": {
            # Eval-only cap on chains per sequence cluster; see Methods 4.11.
            "max_chains_per_cluster": _safe_select(cfg, "data.max_chains_per_cluster"),
            "cap_exempt_subsets": _plain_list(
                _safe_select(cfg, "data.cap_exempt_subsets")
            ),
            "chain_clusters_file": _safe_select(cfg, "data.chain_clusters_file"),
        },
        "model": {
            "esm_model": _safe_select(cfg, "model.esm_model"),
            "head_type": _safe_select(cfg, "model.head_type"),
            "fusion_strategy": _safe_select(cfg, "model.fusion_strategy"),
            "fusion_feature_groups": feature_groups,
            "use_template_features": _safe_select(
                cfg, "model.use_template_features"
            ),
            "use_tpl_dist_bins": _safe_select(cfg, "model.use_tpl_dist_bins"),
            "triangle_c": _safe_select(cfg, "model.triangle_c"),
            "lr": _safe_select(cfg, "model.lr"),
            "warmup_steps": _safe_select(cfg, "model.warmup_steps"),
            "warmup_fraction": _safe_select(cfg, "model.warmup_fraction"),
            "use_tversky": _safe_select(cfg, "model.use_tversky"),
            "tversky_weight": _safe_select(cfg, "model.tversky_weight"),
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
