from typing import Dict

import torch
import matplotlib.pyplot as plt
from lightning import LightningModule

from src.models.components.esm_backbone import ESM2Backbone
from src.models.utils.loss import masked_bce_balanced
from src.models.utils.metrics import precision_at_k_masked
from src.models.utils.visualize import plot_contact_map_comparison, plot_precision_recall_curve


_INIT_THRESHOLD = 0.5

class ESM2OnlyLitModule(LightningModule):
    """
    ESM2-only baseline: Uses ESM2's built-in contact predictions directly
    
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
        
        # For threshold optimization
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []
        
        self._test_probs = []
        self._test_targets = []
        self._test_masks = []
    
    def state_dict(self):
        """Save state including the optimal threshold"""
        state = super().state_dict()
        state['pred_threshold'] = self.pred_threshold
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state including the optimal threshold"""
        if 'pred_threshold' in state_dict:
            self.pred_threshold = state_dict.pop('pred_threshold')
            print(f"Loaded optimal threshold from checkpoint: {self.pred_threshold:.4f}")
        return super().load_state_dict(state_dict, strict=strict)
    
    def _step(self, batch: Dict[str, torch.Tensor], stage: str, return_visualization: bool = False):
        """
        Simply extract ESM2 contact predictions and compute metrics.
        """
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]
        contact = batch["contact"].to(self.device)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)
        
        # Get ESM2 predictions
        esm_contacts_list = []
        for pid, seq, bounds in zip(pids, seqs, crop_bounds):
            _, esm_cont = self.esm([(pid, seq)], self.device)
            # Crop to bounds
            esm_cont_crop = esm_cont[:, :, bounds[0]:bounds[1], bounds[0]:bounds[1]]
            esm_contacts_list.append(esm_cont_crop)
        
        # Pad to batch max length
        Lmax = max([cont.shape[2] for cont in esm_contacts_list])
        esm_contacts = torch.zeros(len(esm_contacts_list), 1, Lmax, Lmax, device=self.device)
        for i, cont in enumerate(esm_contacts_list):
            L = cont.shape[2]
            esm_contacts[i, :, :L, :L] = cont
        
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
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="val", return_visualization=True)
        
        # Accumulate predictions for optimal threshold finding
        self._val_probs.append(viz_cache['prob'].detach().cpu())
        self._val_targets.append(viz_cache['contact'].detach().cpu())
        self._val_masks.append(viz_cache['valid_mask'].detach().cpu())
        
        # Log one visualization per epoch
        if not self._val_viz_logged and batch_idx == 0:
            self._log_visualization(viz_cache)
            self._val_viz_logged = True
        
        return loss
    
    def on_validation_epoch_start(self):
        self._val_viz_logged = False
    
    def on_validation_epoch_end(self):
        if len(self._val_probs) == 0:
            return
        
        # Skip threshold optimization during sanity check
        if self.trainer.sanity_checking:
            print("Skipping threshold optimization during sanity check")
            self._val_probs = []
            self._val_targets = []
            self._val_masks = []
            return
        
        # Find max length across all batches and pad
        max_L = max(batch.shape[-1] for batch in self._val_probs)
        padded_probs = []
        padded_targets = []
        padded_masks = []
        
        for prob_batch, target_batch, mask_batch in zip(self._val_probs, self._val_targets, self._val_masks):
            if prob_batch.dim() == 4:
                prob_batch = prob_batch.squeeze(1)
            B, L, _ = prob_batch.shape
            if L < max_L:
                pad_size = max_L - L
                prob_batch = torch.nn.functional.pad(prob_batch, (0, pad_size, 0, pad_size), value=0)
                target_batch = torch.nn.functional.pad(target_batch, (0, pad_size, 0, pad_size), value=0)
                mask_batch = torch.nn.functional.pad(mask_batch, (0, pad_size, 0, pad_size), value=0)
            padded_probs.append(prob_batch)
            padded_targets.append(target_batch)
            padded_masks.append(mask_batch)
        
        all_probs = torch.cat(padded_probs, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_masks = torch.cat(padded_masks, dim=0)
        
        optimal_threshold, optimal_f1 = self._find_optimal_threshold(all_probs, all_targets, all_masks)
        self.pred_threshold = optimal_threshold
        
        all_preds = (all_probs >= optimal_threshold).float()
        valid_preds = all_preds[all_masks > 0]
        valid_targets = all_targets[all_masks > 0]
        
        tp = ((valid_preds == 1) & (valid_targets == 1)).sum().float()
        fp = ((valid_preds == 1) & (valid_targets == 0)).sum().float()
        fn = ((valid_preds == 0) & (valid_targets == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        self.log("val/optimal_threshold", optimal_threshold, prog_bar=True, sync_dist=False)
        self.log("val/precision", precision, prog_bar=True, sync_dist=False)
        self.log("val/recall", recall, prog_bar=True, sync_dist=False)
        self.log("val/f1", optimal_f1, prog_bar=True, sync_dist=False)
        
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []
    
    def _find_optimal_threshold(self, prob, target, mask):
        thresholds = torch.linspace(0.05, 0.95, 19)
        best_f1 = 0.0
        best_threshold = 0.5
        
        for thresh in thresholds:
            pred = (prob >= thresh).float()
            valid_pred = pred[mask > 0]
            valid_target = target[mask > 0]
            
            tp = ((valid_pred == 1) & (valid_target == 1)).sum().float()
            fp = ((valid_pred == 1) & (valid_target == 0)).sum().float()
            fn = ((valid_pred == 0) & (valid_target == 1)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh.item()
        
        return best_threshold, best_f1
    
    def on_test_start(self):
        print(f"Starting test with threshold: {self.pred_threshold:.4f}")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="test", return_visualization=True)
        
        self._test_probs.append(viz_cache['prob'].detach().cpu())
        self._test_targets.append(viz_cache['contact'].detach().cpu())
        self._test_masks.append(viz_cache['valid_mask'].detach().cpu())
        
        self._save_test_batch_visualizations(viz_cache)
        
        return loss
    
    def on_test_epoch_end(self):
        if len(self._test_probs) == 0:
            return
        
        # Find max length across all batches and pad
        max_L = max(batch.shape[-1] for batch in self._test_probs)
        padded_probs = []
        padded_targets = []
        padded_masks = []
        
        for prob_batch, target_batch, mask_batch in zip(self._test_probs, self._test_targets, self._test_masks):
            if prob_batch.dim() == 4:
                prob_batch = prob_batch.squeeze(1)
            B, L, _ = prob_batch.shape
            if L < max_L:
                pad_size = max_L - L
                prob_batch = torch.nn.functional.pad(prob_batch, (0, pad_size, 0, pad_size), value=0)
                target_batch = torch.nn.functional.pad(target_batch, (0, pad_size, 0, pad_size), value=0)
                mask_batch = torch.nn.functional.pad(mask_batch, (0, pad_size, 0, pad_size), value=0)
            padded_probs.append(prob_batch)
            padded_targets.append(target_batch)
            padded_masks.append(mask_batch)
        
        all_probs = torch.cat(padded_probs, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_masks = torch.cat(padded_masks, dim=0)
        
        all_preds = (all_probs >= self.pred_threshold).float()
        valid_preds = all_preds[all_masks > 0]
        valid_targets = all_targets[all_masks > 0]
        
        tp = ((valid_preds == 1) & (valid_targets == 1)).sum().float()
        fp = ((valid_preds == 1) & (valid_targets == 0)).sum().float()
        fn = ((valid_preds == 0) & (valid_targets == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        self.log("test/precision", precision, prog_bar=True, sync_dist=False)
        self.log("test/recall", recall, prog_bar=True, sync_dist=False)
        self.log("test/f1", f1, prog_bar=True, sync_dist=False)
        self.log("test/threshold_used", self.pred_threshold, prog_bar=False, sync_dist=False)
        
        self._test_probs = []
        self._test_targets = []
        self._test_masks = []
    
    def _log_visualization(self, viz_cache: Dict):
        """Log a contact map visualization for the first sample in batch"""
        # Extract relevant data from cache
        idx = 0  # First sample
        
        seq = viz_cache['seq'][idx]
        crop_bounds = viz_cache['crop_bounds'][idx]
        pid = viz_cache['pid'][idx]
        
        # Get cropped sequence
        seq_crop = seq[crop_bounds[0]:crop_bounds[1]]
        L = len(seq_crop)
        
        # Crop tensors to actual length
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
        
        # Generate precision-recall curve
        prc_fig = plot_precision_recall_curve(
            pred_prob=prob,
            target=contact,
            mask=valid_mask,
            pid=pid,
        )
        
        # Log figures - handle different logger types
        if self.logger is not None:
            from io import BytesIO
            from PIL import Image
            
            # Convert contact map figure to image
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            
            # Convert PRC figure to image
            prc_buf = BytesIO()
            prc_fig.savefig(prc_buf, format="png", dpi=100, bbox_inches="tight")
            prc_buf.seek(0)
            prc_img = Image.open(prc_buf)
            
            # Try different logger APIs
            logger_name = self.logger.__class__.__name__
            
            try:
                if hasattr(self.logger.experiment, "add_figure"):
                    # TensorBoard - log with global step
                    self.logger.experiment.add_figure(
                        f"val/contact_map_{pid}",
                        fig,
                        global_step=self.current_epoch,
                    )
                    self.logger.experiment.add_figure(
                        f"val/pr_curve_{pid}",
                        prc_fig,
                        global_step=self.current_epoch,
                    )
                elif logger_name == "NeptuneLogger":
                    # Neptune - log as series with step for slider
                    import neptune.types as neptune_types
                    
                    self.logger.experiment[f"val/contact_map_{pid}"].append(
                        neptune_types.File.as_image(img), step=self.current_epoch
                    )
                    self.logger.experiment[f"val/pr_curve_{pid}"].append(
                        neptune_types.File.as_image(prc_img), step=self.current_epoch
                    )
                else:
                    # Fallback: try generic log_image if available
                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(
                            key=f"val/contact_map_{pid}",
                            images=[img],
                            step=self.current_epoch,
                        )
                        self.logger.log_image(
                            key=f"val/pr_curve_{pid}",
                            images=[prc_img],
                            step=self.current_epoch,
                        )
            except Exception as e:
                print(f"Warning: Failed to log visualization with {logger_name}: {e}")
            
            buf.close()
            prc_buf.close()
        
        plt.close(fig)
        plt.close(prc_fig)
    
    def _save_test_batch_visualizations(self, viz_cache: Dict):
        """Save contact map visualizations for all samples in test batch"""
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
        
        # Process each sample in the batch
        for idx in range(len(viz_cache["pid"])):
            prob = viz_cache["prob"][idx]
            contact = viz_cache["contact"][idx]
            valid_mask = viz_cache["valid_mask"][idx]
            seq = viz_cache["seq"][idx]
            crop_bounds = viz_cache["crop_bounds"][idx]
            pid = viz_cache["pid"][idx]
            
            # Get cropped sequence and tensors
            seq_crop = seq[crop_bounds[0]:crop_bounds[1]]
            L = len(seq_crop)
            prob_crop = prob[:L, :L]
            contact_crop = contact[:L, :L]
            valid_crop = valid_mask[:L, :L]
            
            # Save contact map visualization
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
            
            # Save precision-recall curve
            prc_save_path = save_dir / f"test_{pid}_PRC.png"
            prc_fig = plot_precision_recall_curve(
                pred_prob=prob_crop,
                target=contact_crop,
                mask=valid_crop,
                pid=pid,
            )
            prc_fig.savefig(prc_save_path, dpi=100, bbox_inches="tight")

            plt.close(prc_fig)
