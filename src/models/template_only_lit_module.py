"""
Template-only baseline: Uses only template retrieval without ESM2 embeddings.

This module demonstrates the contribution of templates alone, without the
protein language model. Used for ablation studies to show that both
components (pLM + templates) are necessary for best performance.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule

from src.data.components.dataset import PriorBuilder
from src.models.components.pair2d_head import relpos_buckets
from src.models.utils.loss import masked_bce_balanced
from src.models.utils.metrics import (
    precision_at_k_masked,
    precision_at_k_by_range,
    _create_range_mask,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

_INIT_THRESHOLD = 0.5


# Number of relpos channels produced by relpos_buckets with default cuts
_RELPOS_CHANNELS = len((0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64)) + 1  # 14


class TemplateOnlyHead(nn.Module):
    """Simple CNN head for template-only baseline."""

    def __init__(
        self,
        in_channels: int = 2,  # template prior + count
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        relpos_channels: int = _RELPOS_CHANNELS,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels + relpos_channels  # + relpos buckets

        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(current_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ])
            current_channels = hidden_dim

        # Final projection
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        template_prior: torch.Tensor,
        count: torch.Tensor,
        relpos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            template_prior: (B, 1, L, L) - average template contact map
            count: (B, 1, L, L) - number of templates covering each position
            relpos: (R, L, L) - relative position encoding

        Returns:
            logits: (B, 1, L, L)
        """
        B = template_prior.shape[0]

        # Combine inputs
        relpos_batch = relpos.unsqueeze(0).expand(B, -1, -1, -1)  # (B, R, L, L)
        x = torch.cat([template_prior, count, relpos_batch], dim=1)  # (B, 2+R, L, L)

        return self.net(x)


class TemplateOnlyLitModule(LightningModule):
    """
    Template-only baseline for ablation studies.

    Uses FAISS retrieval + template priors, but NO ESM2 embeddings.
    Shows the contribution of templates alone.

    Template priors are built by PriorBuilder (same as the main model)
    and relative position encoding reuses ``relpos_buckets`` from
    ``pair2d_head.py`` (no duplicate implementation).
    """

    def __init__(
        self,
        # Retrieval settings
        index_dir: str = "data/index_t6",
        topk: int = 4,
        min_seq_sep: int = 6,
        # Head settings
        hidden_dim: int = 64,
        num_layers: int = 4,
        # Training settings
        lr: float = 2e-4,
        wd: float = 1e-2,
        # Loss settings
        pos_weight_scale: float = 0.8,
        label_smoothing: float = 0.05,
        # Alignment
        use_blosum: bool = True,
        # Cache
        max_tpl_cache: int = 1000,
        # Scheduler
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        cosine_restarts: bool = False,
        restart_period: int = 1000,
        restart_mult: float = 2.0,
        restart_decay: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Settings
        self.topk = topk
        self.min_seq_sep = min_seq_sep
        self.lr = lr
        self.wd = wd
        self.pos_weight_scale = pos_weight_scale
        self.label_smoothing = label_smoothing
        self.use_blosum = use_blosum
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.cosine_restarts = bool(cosine_restarts)
        self.restart_period = int(restart_period)
        self.restart_mult = float(restart_mult)
        self.restart_decay = float(restart_decay)

        # PriorBuilder — lazy init (needs FAISS index on disk)
        self.index_dir = Path(index_dir)
        self.max_tpl_cache = max_tpl_cache
        self._prior_builder: Optional[PriorBuilder] = None

        # Build head
        self.head = TemplateOnlyHead(
            in_channels=2,  # prior + count
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Threshold
        self.pred_threshold = _INIT_THRESHOLD

        # ── Streaming validation metrics (mirrors ContactLitModule) ──
        self._val_thresholds = torch.linspace(0.05, 0.99, 20)
        self._val_range_tp: Dict[str, torch.Tensor] = {}
        self._val_range_fp: Dict[str, torch.Tensor] = {}
        self._val_range_fn: Dict[str, torch.Tensor] = {}
        self._val_range_tn: Dict[str, torch.Tensor] = {}
        self._val_pL_range: Dict[str, List[float]] = {}
        self._range_defs = {"short": (6, 12), "medium": (12, 24), "long": (24, None)}

        # ── Streaming test metrics ──
        self._test_range_tp: Dict[str, torch.Tensor] = {}
        self._test_range_fp: Dict[str, torch.Tensor] = {}
        self._test_range_fn: Dict[str, torch.Tensor] = {}
        self._test_range_tn: Dict[str, torch.Tensor] = {}
        self._test_pL_range: Dict[str, List[float]] = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0

    # ── PriorBuilder (lazy init) ──────────────────────────────────────

    def _ensure_prior_builder(self):
        if self._prior_builder is None:
            self._prior_builder = PriorBuilder(
                index_dir=str(self.index_dir),
                topk=self.topk,
                use_blosum=self.use_blosum,
                only_positive_transfer=False,
                min_seq_sep=self.min_seq_sep,
                max_tpl_cache=self.max_tpl_cache,
            )

    # ── Forward ───────────────────────────────────────────────────────

    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ):
        """Forward pass."""
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]
        contact = batch["contact"].to(self.device)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)

        # Use precomputed priors from DataLoader if available, else build here
        if "prior" in batch and "count" in batch:
            prior = batch["prior"].to(self.device)
            count = batch["count"].to(self.device)
        else:
            self._ensure_prior_builder()
            Lmax = contact.shape[-1]
            priors, counts = [], []
            for pid, seq, bounds in zip(pids, seqs, crop_bounds):
                p_np, c_np, _d_np = self._prior_builder.build_one(
                    pid, seq, bounds[0].item(), bounds[1].item()
                )
                # Pad to Lmax
                Lc = p_np.shape[0]
                p_pad = np.zeros((Lmax, Lmax), dtype=np.float32)
                c_pad = np.zeros((Lmax, Lmax), dtype=np.float32)
                p_pad[:Lc, :Lc] = p_np
                c_pad[:Lc, :Lc] = c_np
                priors.append(torch.from_numpy(p_pad))
                counts.append(torch.from_numpy(c_pad))
            prior = torch.stack(priors).unsqueeze(1).to(self.device)
            count = torch.stack(counts).unsqueeze(1).to(self.device)

        Lmax = contact.shape[-1]

        # Relative position encoding (shared, cached via lru_cache)
        relpos = relpos_buckets(Lmax, self.device)  # (R, L, L)

        # Forward through head
        logits = self.head(prior, count, relpos)  # (B, 1, L, L)

        # Loss
        valid_mask = long_mask * pair_mask
        loss = masked_bce_balanced(
            logits,
            contact,
            valid_mask,
            pos_weight_scale=self.pos_weight_scale,
            label_smoothing=self.label_smoothing,
        )

        # Batch-level P@L (lightweight)
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pL = precision_at_k_masked(prob, contact, valid_mask, k_mode="L")

        prog_bar = stage == "val"
        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/P@L", pL, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)

        return loss, prob, contact, valid_mask

    # ── Training ──────────────────────────────────────────────────────

    def training_step(self, batch: Dict, batch_idx: int):
        loss, _, _, _ = self._step(batch, stage="train")
        return loss

    # ── Validation (streaming) ────────────────────────────────────────

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, prob, contact, mask = self._step(batch, stage="val")
        self._accumulate_streaming(
            prob, contact, mask,
            tp_dict=self._val_range_tp,
            fp_dict=self._val_range_fp,
            fn_dict=self._val_range_fn,
            tn_dict=self._val_range_tn,
            pL_dict=self._val_pL_range,
        )
        return loss

    def on_validation_epoch_start(self):
        self._val_range_tp = {}
        self._val_range_fp = {}
        self._val_range_fn = {}
        self._val_range_tn = {}
        self._val_pL_range = {}

    def on_validation_epoch_end(self):
        if not self._val_range_tp:
            return
        if self.trainer.sanity_checking:
            return

        self._log_range_metrics("val", self._val_range_tp, self._val_range_fp,
                                self._val_range_fn, self._val_range_tn,
                                self._val_pL_range)

        # Set threshold from long-range best-F1
        if "long" in self._val_range_tp:
            tp_l = self._val_range_tp["long"].cpu()
            fp_l = self._val_range_fp["long"].cpu()
            fn_l = self._val_range_fn["long"].cpu()
            f1_l = 2 * tp_l / (2 * tp_l + fp_l + fn_l + 1e-8)
            self.pred_threshold = self._val_thresholds[int(f1_l.argmax())].item()
        self.log("val/optimal_threshold", self.pred_threshold, prog_bar=True, sync_dist=False)

    # ── Test (streaming) ──────────────────────────────────────────────

    def test_step(self, batch: Dict, batch_idx: int):
        loss, prob, contact, mask = self._step(batch, stage="test")

        if prob.dim() == 4:
            prob = prob.squeeze(1)

        # Global TP/FP/FN at pred_threshold
        preds_bin = (prob >= self.pred_threshold).float()
        valid_preds = preds_bin[mask > 0]
        valid_targets = contact[mask > 0]
        self._test_tp += ((valid_preds == 1) & (valid_targets == 1)).sum().item()
        self._test_fp += ((valid_preds == 1) & (valid_targets == 0)).sum().item()
        self._test_fn += ((valid_preds == 0) & (valid_targets == 1)).sum().item()

        self._accumulate_streaming(
            prob.unsqueeze(1) if prob.dim() == 3 else prob,
            contact, mask,
            tp_dict=self._test_range_tp,
            fp_dict=self._test_range_fp,
            fn_dict=self._test_range_fn,
            tn_dict=self._test_range_tn,
            pL_dict=self._test_pL_range,
        )
        return loss

    def on_test_epoch_start(self):
        self._test_range_tp = {}
        self._test_range_fp = {}
        self._test_range_fn = {}
        self._test_range_tn = {}
        self._test_pL_range = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0

    def on_test_epoch_end(self):
        if not self._test_range_tp:
            return

        # Global precision / recall / F1
        tp, fp, fn = self._test_tp, self._test_fp, self._test_fn
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        self.log("test/precision", precision, prog_bar=True, sync_dist=False)
        self.log("test/recall", recall, prog_bar=True, sync_dist=False)
        self.log("test/f1", f1, prog_bar=True, sync_dist=False)
        self.log("test/threshold_used", self.pred_threshold, prog_bar=False, sync_dist=False)

        self._log_range_metrics("test", self._test_range_tp, self._test_range_fp,
                                self._test_range_fn, self._test_range_tn,
                                self._test_pL_range)

        # Clear
        self._test_range_tp = {}
        self._test_range_fp = {}
        self._test_range_fn = {}
        self._test_range_tn = {}
        self._test_pL_range = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0

    # ── Streaming helpers ─────────────────────────────────────────────

    def _accumulate_streaming(
        self,
        prob: torch.Tensor,
        contact: torch.Tensor,
        mask: torch.Tensor,
        tp_dict: Dict[str, torch.Tensor],
        fp_dict: Dict[str, torch.Tensor],
        fn_dict: Dict[str, torch.Tensor],
        tn_dict: Dict[str, torch.Tensor],
        pL_dict: Dict[str, List[float]],
    ):
        """Accumulate streaming TP/FP/FN/TN and P@L by range."""
        prob_sq = prob.detach()
        if prob_sq.dim() == 4:
            prob_sq = prob_sq.squeeze(1)
        contact_d = contact.detach()
        mask_d = mask.detach()
        B, L, _ = prob_sq.shape
        thresholds = self._val_thresholds.to(prob_sq.device)

        for rname, (min_sep, max_sep) in self._range_defs.items():
            range_mask = _create_range_mask(L, min_sep, max_sep, prob_sq.device).float()
            combined = mask_d * range_mask.unsqueeze(0)
            p_flat = prob_sq[combined > 0].flatten()
            t_flat = contact_d[combined > 0].flatten()
            if p_flat.numel() == 0:
                continue

            preds = (p_flat.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
            t_exp = t_flat.unsqueeze(0).expand_as(preds)
            tp = (preds * t_exp).sum(dim=1)
            fp = (preds * (1 - t_exp)).sum(dim=1)
            fn = ((1 - preds) * t_exp).sum(dim=1)
            tn = ((1 - preds) * (1 - t_exp)).sum(dim=1)

            if rname not in tp_dict:
                tp_dict[rname] = tp
                fp_dict[rname] = fp
                fn_dict[rname] = fn
                tn_dict[rname] = tn
            else:
                tp_dict[rname] += tp
                fp_dict[rname] += fp
                fn_dict[rname] += fn
                tn_dict[rname] += tn

        # P@L by range
        range_metrics = precision_at_k_by_range(prob_sq, contact_d, mask_d, k_mode="L")
        for rname, val in range_metrics.items():
            pL_dict.setdefault(rname, []).append(val)

    def _log_range_metrics(
        self,
        stage: str,
        tp_dict: Dict[str, torch.Tensor],
        fp_dict: Dict[str, torch.Tensor],
        fn_dict: Dict[str, torch.Tensor],
        tn_dict: Dict[str, torch.Tensor],
        pL_dict: Dict[str, List[float]],
    ):
        """Aggregate streaming stats into per-range metrics."""
        for rname in ("short", "medium", "long"):
            if rname not in tp_dict:
                continue
            tp = tp_dict[rname].cpu()
            fp = fp_dict[rname].cpu()
            fn = fn_dict[rname].cpu()
            tn = tn_dict[rname].cpu()

            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            best_idx = int(f1.argmax())
            best_f1 = f1[best_idx].item()
            prec = (tp[best_idx] / (tp[best_idx] + fp[best_idx] + 1e-8)).item()
            rec = (tp[best_idx] / (tp[best_idx] + fn[best_idx] + 1e-8)).item()

            is_long = rname == "long"
            self.log(f"{stage}/f1_{rname}", best_f1, prog_bar=is_long, sync_dist=False)
            self.log(f"{stage}/precision_{rname}", prec, prog_bar=False, sync_dist=False)
            self.log(f"{stage}/recall_{rname}", rec, prog_bar=False, sync_dist=False)

            # MCC
            mcc_num = tp * tn - fp * fn
            mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            mcc_all = mcc_num / mcc_den
            self.log(f"{stage}/MCC_{rname}", mcc_all[best_idx].item(), prog_bar=False, sync_dist=False)

        # P@L by range
        for rname, vals in pL_dict.items():
            avg = float(np.mean(vals)) if vals else 0.0
            self.log(f"{stage}/P@L_{rname}", avg, prog_bar=(rname == "long"), sync_dist=False)

    # ── Optimizer ─────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        total_steps = (
            self.trainer.estimated_stepping_batches
            if self.trainer is not None and self.trainer.estimated_stepping_batches
            else 10000
        )

        _warmup = self.warmup_steps
        _min_ratio = self.min_lr_ratio
        _restarts = self.cosine_restarts
        _T0 = self.restart_period
        _Tmult = self.restart_mult
        _decay = self.restart_decay

        def lr_lambda(step: int) -> float:
            if step < _warmup:
                return float(step) / float(max(1, _warmup))

            s = step - _warmup

            if _restarts:
                if _Tmult == 1.0:
                    cycle_idx = s // _T0
                    cycle_pos = (s % _T0) / float(_T0)
                else:
                    t_cur = _T0
                    cumul = 0
                    cycle_idx = 0
                    while cumul + t_cur <= s:
                        cumul += t_cur
                        t_cur = int(t_cur * _Tmult)
                        cycle_idx += 1
                    cycle_pos = (s - cumul) / float(max(1, t_cur))
                amplitude = _decay ** cycle_idx
            else:
                cycle_pos = s / float(max(1, total_steps - _warmup))
                amplitude = 1.0

            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * min(cycle_pos, 1.0)))
            return max(_min_ratio, amplitude * cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
