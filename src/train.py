from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import torch
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# PyTorch 2.6+ defaults weights_only=True; allow numpy types stored in checkpoints
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _dump_metrics_json(metrics: Dict[str, Any], cfg: DictConfig, filename: str) -> None:
    """Serialize Lightning callback_metrics to <output_dir>/<filename>.

    Used by scripts/smoke_repro_test.py (bit-determinism check) and any post-hoc
    analysis that wants the exact final-epoch numbers without re-fetching from W&B.
    Defensive: silently skips on missing output_dir or non-serialisable values.
    """
    try:
        import json
        import math
        from pathlib import Path

        output_dir = cfg.get("paths", {}).get("output_dir") if hasattr(cfg, "get") else None
        if not output_dir:
            return

        def _coerce(v: Any) -> Any:
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        coerced = {str(k): _coerce(v) for k, v in dict(metrics).items()}
        out = Path(output_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coerced, indent=2, default=str, sort_keys=True))
        log.info(f"Wrote {filename} to {out}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"Failed to dump {filename}: {e}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    train_metrics = trainer.callback_metrics

    # Dump train metrics for downstream tooling (e.g. scripts/smoke_repro_test.py).
    # This is a no-op if there's no output_dir configured (defensive).
    _dump_metrics_json(train_metrics, cfg, "train_metrics.json")

    if cfg.get("test"):
        log.info("Starting testing!")
        
        ckpt_path = cfg.get("ckpt_path")

        if not ckpt_path:
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                    checkpoint_callback = callback
                    break
            
            if checkpoint_callback is not None and checkpoint_callback.best_model_path:
                ckpt_path = checkpoint_callback.best_model_path
            else:
                ckpt_path = None
                log.warning("No checkpoint found! Using current model weights for testing...")

        # Calibrate the decision threshold on VALIDATION before testing, under the
        # exact (best-checkpoint) weights that will be tested. Otherwise eval-only
        # baselines never run validation and test at the 0.5 default, and the
        # per-range test metrics would have to tune the threshold on the test set.
        # validate(best) then test(in-memory, ckpt_path=None) so the test reload
        # cannot overwrite the freshly-calibrated threshold.
        if cfg.get("validate_before_test", True):
            log.info("Validating (threshold calibration) before testing!")
            trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                weights_only=False,
            )
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=None,
                weights_only=False,
            )
        else:
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                weights_only=False,
            )

        # Paper-grade audit: snapshot retrieval+model config to .temp/audit/<run_id>/
        # so supplementary Methods can cite "what was actually loaded for inference"
        # (companion to verify_no_leak.py / template_coverage.py launched pre-train).
        try:
            from src.utils.audit import dump_audit_manifest
            dump_audit_manifest(cfg, trainer, ckpt_path=ckpt_path)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Audit manifest dump failed (non-fatal): {e}")

    test_metrics = trainer.callback_metrics

    _dump_metrics_json(test_metrics, cfg, "test_metrics.json")

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
