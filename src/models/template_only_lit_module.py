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
    per_sample_metric_rows,
    export_per_sample_tsv,
    log_macro_test_metrics,
    range_metrics_at_threshold,
    unique_pair_mask,
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
        index_dir: str = "data/index_t6_2026",
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
        warmup_steps: int = 0,  # 0 → use warmup_fraction
        warmup_fraction: float = 0.02,
        min_lr_ratio: float = 0.1,
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
        self.warmup_fraction = float(warmup_fraction)
        self.min_lr_ratio = float(min_lr_ratio)

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
        # Per-chain rows for per_sample_metrics.tsv (paired significance vs frontier).
        self._test_per_sample: List[Dict] = []

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
                p_np, c_np = self._prior_builder.build_one(
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
        # Test P@L here is a batch-average → log under *_batchavg so the canonical
        # test/P@L is the per-protein macro from the helper.
        pl_sfx = "_batchavg" if stage == "test" else ""
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/P@L{pl_sfx}", pL, prog_bar=prog_bar, on_step=on_step, on_epoch=True, sync_dist=True)

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

        # Global TP/FP/FN at pred_threshold — unique pairs only (sanity micro).
        umask = unique_pair_mask(mask)
        preds_bin = (prob >= self.pred_threshold).float()
        valid_preds = preds_bin[umask > 0]
        valid_targets = contact[umask > 0]
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
        self._test_per_sample.extend(
            per_sample_metric_rows(
                prob if prob.dim() == 3 else prob.squeeze(1),
                contact, mask,
                batch["pid"],
                batch.get("subset", ["all"] * prob.shape[0]),
                self.pred_threshold,
                seq_lens=batch.get("seq_len"),
                n_templates=batch.get("n_templates_retrieved"),
                best_sims=batch.get("best_tpl_sim"),
                cluster_ids=batch.get("cluster_id"),
            )
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
        self._test_per_sample = []

    def on_test_epoch_end(self):
        rows = self._test_per_sample
        if not rows:
            return

        export_per_sample_tsv(rows, self.trainer)
        self.log("test/threshold_used", self.pred_threshold, prog_bar=False, sync_dist=False)

        # HEADLINE = per-protein MACRO (+ per-subset), shared helper. Same
        # canonical keys + definition as the frontier ContactLitModule.
        from src.data.components.dataset import (
            SUBSET_GOLD, SUBSET_CASP16, SUBSET_CLUSTER_PROMOTED,
        )
        log_macro_test_metrics(
            self.log, rows, [SUBSET_GOLD, SUBSET_CASP16, SUBSET_CLUSTER_PROMOTED]
        )

        # Pooled-at-val-threshold per-range (sanity, *_micro) — NOT argmax-on-test.
        for rname in ("short", "medium", "long"):
            if rname not in self._test_range_tp:
                continue
            rm = range_metrics_at_threshold(
                self._test_range_tp[rname].cpu(), self._test_range_fp[rname].cpu(),
                self._test_range_fn[rname].cpu(), self._test_range_tn[rname].cpu(),
                self._val_thresholds, self.pred_threshold,
            )
            self.log(f"test/f1_{rname}_micro", rm["f1"], prog_bar=False, sync_dist=False)
            self.log(f"test/precision_{rname}_micro", rm["precision"], prog_bar=False, sync_dist=False)
            self.log(f"test/recall_{rname}_micro", rm["recall"], prog_bar=False, sync_dist=False)

        # Global pooled at val threshold (sanity, *_micro)
        tp, fp, fn = self._test_tp, self._test_fp, self._test_fn
        g_prec = tp / max(1, tp + fp)
        g_rec = tp / max(1, tp + fn)
        self.log("test/f1_micro", 2 * g_prec * g_rec / max(1e-8, g_prec + g_rec),
                 prog_bar=False, sync_dist=False)
        self.log("test/precision_micro", g_prec, prog_bar=False, sync_dist=False)
        self.log("test/recall_micro", g_rec, prog_bar=False, sync_dist=False)

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

        effective_warmup = (
            self.warmup_steps
            if self.warmup_steps > 0
            else max(1, int(total_steps * self.warmup_fraction))
        )
        warmup_source = "explicit" if self.warmup_steps > 0 else f"{self.warmup_fraction:.1%} of total"
        log.info(
            f"LR schedule: total_steps={total_steps}, warmup={effective_warmup} ({warmup_source})"
        )

        def lr_lambda(step: int) -> float:
            if step < effective_warmup:
                return float(step) / float(max(1, effective_warmup))
            progress = float(step - effective_warmup) / float(max(1, total_steps - effective_warmup))
            return max(self.min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
