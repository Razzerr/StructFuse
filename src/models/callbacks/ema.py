from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class EMACallback(Callback):
    """Exponential moving average of trainable parameters, swapped in for validation.

    Standard pattern from AF2/ESMFold: track an EMA shadow of the live weights with
    bias-corrected decay, swap shadow → live at validation start, restore live at the
    next training batch. ModelCheckpoint therefore observes EMA metrics AND saves the
    EMA snapshot (since the swap persists past on_validation_end).

    Frozen parameters (requires_grad=False, e.g. ESM2 backbone) are skipped — keeps
    memory ≈ trainable params only (~1.7M for ContactLitModule, negligible).
    """

    def __init__(
        self,
        decay: float = 0.999,
        warmup_steps: int = 1000,
        validate_with_ema: bool = True,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.validate_with_ema = bool(validate_with_ema)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

    @staticmethod
    def _trainable(pl_module: LightningModule):
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                yield name, p

    def _current_decay(self) -> float:
        bias_corrected = (1.0 + self.num_updates) / (self.warmup_steps + self.num_updates)
        return min(self.decay, bias_corrected)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.shadow:
            return
        for name, p in self._trainable(pl_module):
            self.shadow[name] = p.detach().clone()
        log.info(
            f"EMA initialised: {len(self.shadow)} param tensors, "
            f"decay={self.decay}, warmup_steps={self.warmup_steps}, "
            f"validate_with_ema={self.validate_with_ema}"
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.shadow:
            return
        self.num_updates += 1
        d = self._current_decay()
        with torch.no_grad():
            for name, p in self._trainable(pl_module):
                shadow = self.shadow.get(name)
                if shadow is None:
                    self.shadow[name] = p.detach().clone()
                    continue
                if shadow.device != p.device:
                    shadow = shadow.to(p.device)
                    self.shadow[name] = shadow
                shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.validate_with_ema or not self.shadow:
            return
        if self.backup:
            return
        for name, p in self._trainable(pl_module):
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(shadow.to(p.device))

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.backup:
            return
        for name, p in self._trainable(pl_module):
            live = self.backup.pop(name, None)
            if live is not None:
                p.data.copy_(live.to(p.device))
        self.backup.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "num_updates": self.num_updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "validate_with_ema": self.validate_with_ema,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.get("shadow", {}).items()}
        self.num_updates = int(state_dict.get("num_updates", 0))
