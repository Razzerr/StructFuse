from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def _to_uint8_rgb(mat: torch.Tensor) -> np.ndarray:
    x = mat.detach().float().cpu().numpy()
    if x.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    x_min, x_max = float(x.min()), float(x.max())
    x = (x - x_min) / (x_max - x_min + 1e-8)
    x = (x * 255.0).astype(np.uint8)
    return np.stack([x, x, x], axis=-1)  # HxWx3

@rank_zero_only
def log_contact_triplet(
    trainer,
    *,
    step: int,
    pred: torch.Tensor,         # (L,L) prob or logits (set as_logits=True if logits)
    true: torch.Tensor,         # (L,L) 0/1
    mask: Optional[torch.Tensor] = None,  # (L,L) 0/1
    prefix: str = "val",
    as_logits: bool = False,
) -> None:
    logger = trainer.logger
    if logger is None:
        return
    try:
        if as_logits:
            pred = torch.sigmoid(pred)

        img_pred = _to_uint8_rgb(pred)
        img_true = _to_uint8_rgb(true)
        img_mask = _to_uint8_rgb(mask) if (mask is not None) else None

        # Neptune
        if logger.__class__.__name__ == "NeptuneLogger":
            from neptune.types import File
            run = getattr(logger, "run", None) or getattr(logger, "experiment", None)
            if run is not None:
                run[f"{prefix}/example_pred"].append(File.as_image(img_pred), step=step)
                run[f"{prefix}/example_true"].append(File.as_image(img_true), step=step)
                if img_mask is not None:
                    run[f"{prefix}/example_mask"].append(File.as_image(img_mask), step=step)

        # TensorBoard
        elif logger.__class__.__name__ == "TensorBoardLogger":
            logger.experiment.add_image(f"{prefix}/example_pred", img_pred.transpose(2, 0, 1), step)
            logger.experiment.add_image(f"{prefix}/example_true", img_true.transpose(2, 0, 1), step)
            if img_mask is not None:
                logger.experiment.add_image(f"{prefix}/example_mask", img_mask.transpose(2, 0, 1), step)

        # CSV / others: skip images
    except Exception:
        pass


@rank_zero_only
def log_first_batch_example(trainer, *, step: int, probs: torch.Tensor, batch: Dict[str, torch.Tensor], prefix: str = "val"):
    """Convenience wrapper: log first sample’s pred/true/mask as images."""
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    try:
        b = 0
        pred = probs[b]
        true = batch["contact"][b]
        mask = batch.get("long_mask", None)
        if isinstance(mask, torch.Tensor):
            mask = mask[b]
        log_contact_triplet(trainer, step=step, pred=pred, true=true, mask=mask, prefix=prefix, as_logits=False)
    except Exception:
        pass