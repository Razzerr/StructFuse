"""
ESM2-only baseline: Uses ESM2's built-in contact predictions directly.

This module is used for evaluation only (not training). It serves as a
lower-bound baseline showing what ESM2 attention-based contacts achieve
without any template retrieval or learned head.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from lightning import LightningModule

from src.models.components.esm_backbone import ESM2Backbone
from src.models.utils.loss import masked_bce_balanced
from src.models.utils.metrics import (
    precision_at_k_masked,
    precision_at_k_by_range,
    _create_range_mask,
)
from src.models.utils.visualize import plot_contact_map_comparison, plot_precision_recall_curve
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

_INIT_THRESHOLD = 0.5


class ESM2OnlyLitModule(LightningModule):
    """
    ESM2-only baseline: Uses ESM2's built-in contact predictions directly.

    This module is used for evaluation only (not training).
    """

    def __init__(
        self,
        esm_model: str = "esm2_t6_8M_UR50D",
        min_seq_sep: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.esm = ESM2Backbone(esm_model, finetune=False)
        self.esm_alphabet = self.esm.alphabet
        self.min_seq_sep = int(min_seq_sep)
        self.pred_threshold = _INIT_THRESHOLD

        # For visualization
        self._val_viz_logged = False

        # ── Streaming validation metrics ──
        self._val_thresholds = torch.linspace(0.05, 0.95, 19)
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

    def state_dict(self):
        """Save state including the optimal threshold."""
        state = super().state_dict()
        state['pred_threshold'] = self.pred_threshold
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load state including the optimal threshold."""
        if 'pred_threshold' in state_dict:
            self.pred_threshold = state_dict.pop('pred_threshold')
            log.info(f"Loaded optimal threshold from checkpoint: {self.pred_threshold:.4f}")
        return super().load_state_dict(state_dict, strict=strict)

    def _step(self, batch: Dict[str, torch.Tensor], stage: str, return_visualization: bool = False):
        """Simply extract ESM2 contact predictions and compute metrics."""
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]
        contact = batch["contact"].to(self.device)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)

        # Get ESM2 predictions — crop sequences BEFORE passing to ESM
        # (matches main ContactLitModule which passes cropped seq[b0:b1])
        seq_list = [(pid, seq[bounds[0]:bounds[1]])
                    for pid, seq, bounds in zip(pids, seqs, crop_bounds)]
        _, esm_contacts_batch = self.esm(seq_list, self.device)  # (B, 1, Lmax_crop, Lmax_crop)

        # Pad to Lmax (contact target size)
        Lmax = contact.shape[-1]
        if esm_contacts_batch.shape[-1] < Lmax:
            pad = Lmax - esm_contacts_batch.shape[-1]
            esm_contacts = torch.nn.functional.pad(esm_contacts_batch, (0, pad, 0, pad))
        else:
            esm_contacts = esm_contacts_batch[:, :, :Lmax, :Lmax]

        # ESM2 outputs are already probabilities (sigmoid applied in backbone)
        # Convert back to logits for loss computation
        logits = torch.logit(esm_contacts.clamp(1e-7, 1 - 1e-7))

        valid_mask = long_mask * pair_mask

        # Compute loss (just for monitoring, no gradients)
        with torch.no_grad():
            loss = masked_bce_balanced(
                logits,
                contact,
                valid_mask,
                pos_weight_scale=1.0,
                label_smoothing=0.0,
            )

        if stage == "val" or stage == "test":
            with torch.no_grad():
                prob = esm_contacts.squeeze(1)  # (B, L, L)
                pL = precision_at_k_masked(prob, contact, valid_mask, k_mode="L")
                pL2 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/2")
                pL5 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/5")

            prog_bar = stage == "val"
            self.log(f"{stage}/loss", loss, prog_bar=prog_bar, on_step=False, on_epoch=True, sync_dist=False)
            self.log(f"{stage}/P@L", pL, prog_bar=prog_bar, on_step=False, on_epoch=True, sync_dist=False)
            self.log(f"{stage}/P@L2", pL2, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log(f"{stage}/P@L5", pL5, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)

        viz_cache = None
        if return_visualization:
            prob = esm_contacts.squeeze(1)
            viz_cache = {
                'prob': prob,
                'contact': contact,
                'valid_mask': valid_mask,
                'seq': seqs,
                'crop_bounds': crop_bounds,
                'pid': pids,
            }

        return loss, viz_cache

    # ── Streaming helper ──────────────────────────────────────────────

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

    # ── Validation ────────────────────────────────────────────────────

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="val", return_visualization=True)

        # Streaming accumulation
        self._accumulate_streaming(
            viz_cache['prob'], viz_cache['contact'], viz_cache['valid_mask'],
            tp_dict=self._val_range_tp,
            fp_dict=self._val_range_fp,
            fn_dict=self._val_range_fn,
            tn_dict=self._val_range_tn,
            pL_dict=self._val_pL_range,
        )

        # Log one visualization per epoch
        if not self._val_viz_logged and batch_idx == 0:
            self._log_visualization(viz_cache)
            self._val_viz_logged = True

        return loss

    def on_validation_epoch_start(self):
        self._val_viz_logged = False
        self._val_range_tp = {}
        self._val_range_fp = {}
        self._val_range_fn = {}
        self._val_range_tn = {}
        self._val_pL_range = {}

    def on_validation_epoch_end(self):
        if not self._val_range_tp:
            return

        # Skip during sanity check
        if self.trainer.sanity_checking:
            log.info("Skipping threshold optimization during sanity check")
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
            best_idx = int(f1_l.argmax())
            self.pred_threshold = self._val_thresholds[best_idx].item()
            self.log("val/optimal_threshold", self.pred_threshold, prog_bar=True, sync_dist=False)
            self.log("val/f1", f1_l[best_idx].item(), prog_bar=True, sync_dist=False)

    # ── Test ──────────────────────────────────────────────────────────

    def on_test_start(self):
        log.info(f"Starting test with threshold: {self.pred_threshold:.4f}")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="test", return_visualization=True)

        prob = viz_cache['prob'].detach()
        contact = viz_cache['contact'].detach()
        mask = viz_cache['valid_mask'].detach()

        if prob.dim() == 4:
            prob = prob.squeeze(1)

        # Global TP/FP/FN at pred_threshold
        preds_bin = (prob >= self.pred_threshold).float()
        valid_preds = preds_bin[mask > 0]
        valid_targets = contact[mask > 0]
        self._test_tp += ((valid_preds == 1) & (valid_targets == 1)).sum().item()
        self._test_fp += ((valid_preds == 1) & (valid_targets == 0)).sum().item()
        self._test_fn += ((valid_preds == 0) & (valid_targets == 1)).sum().item()

        # Per-range streaming
        self._accumulate_streaming(
            prob.unsqueeze(1) if prob.dim() == 3 else prob,
            contact, mask,
            tp_dict=self._test_range_tp,
            fp_dict=self._test_range_fp,
            fn_dict=self._test_range_fn,
            tn_dict=self._test_range_tn,
            pL_dict=self._test_pL_range,
        )

        # Save visualizations
        self._save_test_batch_visualizations(viz_cache)

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

    # ── Visualizations ────────────────────────────────────────────────

    def _log_visualization(self, viz_cache: Dict):
        """Log a contact map visualization for the first sample in batch."""
        idx = 0

        seq = viz_cache['seq'][idx]
        crop_bounds = viz_cache['crop_bounds'][idx]
        pid = viz_cache['pid'][idx]

        seq_crop = seq[crop_bounds[0]:crop_bounds[1]]
        L = len(seq_crop)

        prob = viz_cache['prob'][idx, :L, :L]
        contact = viz_cache['contact'][idx, :L, :L]
        valid_mask = viz_cache['valid_mask'][idx, :L, :L]

        fig = plot_contact_map_comparison(
            pred_prob=prob,
            target=contact,
            mask=valid_mask,
            seq=seq_crop,
            pid=pid,
            threshold=self.pred_threshold,
        )

        prc_fig = plot_precision_recall_curve(
            pred_prob=prob,
            target=contact,
            mask=valid_mask,
            pid=pid,
        )

        if self.logger is not None:
            from io import BytesIO
            from PIL import Image

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)

            prc_buf = BytesIO()
            prc_fig.savefig(prc_buf, format="png", dpi=100, bbox_inches="tight")
            prc_buf.seek(0)
            prc_img = Image.open(prc_buf)

            logger_name = self.logger.__class__.__name__

            try:
                if hasattr(self.logger.experiment, "add_figure"):
                    self.logger.experiment.add_figure(
                        f"val/contact_map_{pid}", fig, global_step=self.current_epoch,
                    )
                    self.logger.experiment.add_figure(
                        f"val/pr_curve_{pid}", prc_fig, global_step=self.current_epoch,
                    )
                elif logger_name == "NeptuneLogger":
                    import neptune.types as neptune_types
                    self.logger.experiment[f"val/contact_map_{pid}"].append(
                        neptune_types.File.as_image(img), step=self.current_epoch
                    )
                    self.logger.experiment[f"val/pr_curve_{pid}"].append(
                        neptune_types.File.as_image(prc_img), step=self.current_epoch
                    )
                else:
                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(
                            key=f"val/contact_map_{pid}", images=[img], step=self.current_epoch,
                        )
                        self.logger.log_image(
                            key=f"val/pr_curve_{pid}", images=[prc_img], step=self.current_epoch,
                        )
            except Exception as e:
                log.warning(f"Failed to log visualization with {logger_name}: {e}")

            buf.close()
            prc_buf.close()

        plt.close(fig)
        plt.close(prc_fig)

    def _save_test_batch_visualizations(self, viz_cache: Dict):
        """Save contact map visualizations for all samples in test batch."""
        from pathlib import Path
        from hydra.core.hydra_config import HydraConfig

        try:
            hydra_cfg = HydraConfig.get()
            log_dir = Path(hydra_cfg.runtime.output_dir)
        except Exception:
            if self.trainer.log_dir is not None and not str(self.trainer.log_dir).startswith('.neptune'):
                log_dir = Path(self.trainer.log_dir)
            else:
                raise ValueError("Cannot determine log directory for saving test visualizations.")

        save_dir = log_dir / "test_visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(len(viz_cache["pid"])):
            prob = viz_cache["prob"][idx]
            contact = viz_cache["contact"][idx]
            valid_mask = viz_cache["valid_mask"][idx]
            seq = viz_cache["seq"][idx]
            crop_bounds = viz_cache["crop_bounds"][idx]
            pid = viz_cache["pid"][idx]

            seq_crop = seq[crop_bounds[0]:crop_bounds[1]]
            L = len(seq_crop)
            prob_crop = prob[:L, :L]
            contact_crop = contact[:L, :L]
            valid_crop = valid_mask[:L, :L]

            save_path = save_dir / f"test_{pid}.png"
            plot_contact_map_comparison(
                pred_prob=prob_crop,
                target=contact_crop,
                mask=valid_crop,
                seq=seq_crop,
                pid=pid,
                save_path=save_path,
                show=False,
                threshold=self.pred_threshold,
            )

            prc_save_path = save_dir / f"test_{pid}_PRC.png"
            prc_fig = plot_precision_recall_curve(
                pred_prob=prob_crop,
                target=contact_crop,
                mask=valid_crop,
                pid=pid,
            )
            prc_fig.savefig(prc_save_path, dpi=100, bbox_inches="tight")
            plt.close(prc_fig)
