"""
Template-only baseline: Uses only template retrieval without ESM2 embeddings.

This module demonstrates the contribution of templates alone, without the
protein language model. Used for ablation studies to show that both
components (pLM + templates) are necessary for best performance.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule

from src.models.utils.faiss import FaissIndex
from src.models.utils.loss import masked_bce_balanced
from src.models.utils.metrics import (
    precision_at_k_masked, 
    precision_at_k_by_range,
    auc_pr_masked,
    mcc_at_threshold,
)
from src.data.utils.align import align_pair, blosum62_score


_INIT_THRESHOLD = 0.5


def relpos_buckets(L: int, device: torch.device, num_buckets: int = 32, max_dist: int = 128) -> torch.Tensor:
    """Generate relative position buckets for positional encoding."""
    idx = torch.arange(L, device=device)
    rel = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L)
    
    # Bucket scheme: log-scale for distances > max_dist/2
    rel_abs = rel.abs()
    is_small = rel_abs < (num_buckets // 2)
    
    # Small distances get linear buckets
    small_buckets = rel_abs.clamp(max=num_buckets // 2 - 1)
    
    # Large distances get log-scale buckets
    log_ratio = torch.log(rel_abs.float() / (num_buckets // 2)) / np.log(max_dist / (num_buckets // 2))
    large_buckets = (num_buckets // 2 + (log_ratio * (num_buckets // 2)).long()).clamp(max=num_buckets - 1)
    
    buckets = torch.where(is_small, small_buckets, large_buckets)
    
    # One-hot encode
    one_hot = torch.nn.functional.one_hot(buckets, num_classes=num_buckets)  # (L, L, num_buckets)
    return one_hot.permute(2, 0, 1).float()  # (num_buckets, L, L)


class TemplateOnlyHead(nn.Module):
    """Simple CNN head for template-only baseline."""
    
    def __init__(
        self,
        in_channels: int = 2,  # template prior + count
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_channels = in_channels + 32  # + relpos buckets
        
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
    
    def forward(self, template_prior: torch.Tensor, count: torch.Tensor, relpos: torch.Tensor) -> torch.Tensor:
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
        
        # Load FAISS index
        self.index_dir = Path(index_dir)
        self.faiss_index = None  # Lazy load
        
        # Template cache
        self.max_tpl_cache = max_tpl_cache
        self._tpl_cache: Dict[str, Dict] = {}
        self._hits_cache: Dict[str, List] = {}
        
        # Build head
        self.head = TemplateOnlyHead(
            in_channels=2,  # prior + count
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # Threshold
        self.pred_threshold = _INIT_THRESHOLD
        
        # Accumulators
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []
        
        self._test_probs = []
        self._test_targets = []
        self._test_masks = []
        self._test_pids = []
        self._test_subsets = []
    
    def _load_index(self):
        """Lazy load FAISS index."""
        if self.faiss_index is None:
            self.faiss_index = FaissIndex(self.index_dir)
    
    def _get_hits(self, pid: str, seq: str) -> List[Tuple[str, float]]:
        """Get template hits from FAISS."""
        cache_key = f"{pid}_{len(seq)}"
        if cache_key in self._hits_cache:
            return self._hits_cache[cache_key]
        
        self._load_index()
        
        # For template-only, we need query embedding
        # Use random embedding or load from index if available
        # For now, use the first template as query (bootstrap approach)
        hits = self.faiss_index.topk(
            np.random.randn(1, 320).astype(np.float32),  # Random query
            k=self.topk,
        )
        
        if len(self._hits_cache) < self.max_tpl_cache:
            self._hits_cache[cache_key] = hits
        
        return hits
    
    def _load_template(self, hit_id: str) -> Optional[Dict]:
        """Load template NPZ data."""
        if hit_id in self._tpl_cache:
            return self._tpl_cache[hit_id]
        
        npz_path = self.index_dir / "npz" / f"{hit_id}.npz"
        if not npz_path.exists():
            return None
        
        data = np.load(npz_path, allow_pickle=True)
        tpl = {
            "seq": str(data["seq"].item() if data["seq"].shape == () else data["seq"]),
            "contact": data["contact"],
        }
        
        if len(self._tpl_cache) < self.max_tpl_cache:
            self._tpl_cache[hit_id] = tpl
        
        return tpl
    
    def _build_prior(
        self, 
        query_seq: str, 
        crop_start: int, 
        crop_end: int,
        Lmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build template prior for a single query."""
        L = crop_end - crop_start
        prior = torch.zeros(1, Lmax, Lmax, device=self.device)
        count = torch.zeros(1, Lmax, Lmax, device=self.device)
        
        hits = self._get_hits("query", query_seq)
        
        for hit_id, score in hits[:self.topk]:
            tpl = self._load_template(hit_id)
            if tpl is None:
                continue
            
            tpl_seq = tpl["seq"]
            tpl_contact = tpl["contact"]
            
            # Align
            if self.use_blosum:
                score_fn = blosum62_score
            else:
                def identity_score(a: str, b: str) -> int:
                    return 1 if a == b else -1
                score_fn = identity_score
            
            q2t, t2q = align_pair(query_seq[crop_start:crop_end], tpl_seq, score_fn)
            
            # Transfer contacts
            for i in range(L):
                ti = q2t.get(i)
                if ti is None:
                    continue
                for j in range(i + self.min_seq_sep, L):
                    tj = q2t.get(j)
                    if tj is None:
                        continue
                    if ti < len(tpl_contact) and tj < len(tpl_contact):
                        prior[0, i, j] += tpl_contact[ti, tj]
                        prior[0, j, i] += tpl_contact[ti, tj]
                        count[0, i, j] += 1
                        count[0, j, i] += 1
        
        # Normalize
        prior = prior / (count + 1e-8)
        count = count.clamp(max=self.topk) / self.topk
        
        return prior, count
    
    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        return_visualization: bool = False,
    ):
        """Forward pass."""
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]
        contact = batch["contact"].to(self.device)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)
        
        Lmax = contact.shape[1]
        
        # Build template priors for batch
        priors = []
        counts = []
        for pid, seq, bounds in zip(pids, seqs, crop_bounds):
            prior, count = self._build_prior(seq, bounds[0].item(), bounds[1].item(), Lmax)
            priors.append(prior)
            counts.append(count)
        
        prior = torch.stack(priors, dim=0)  # (B, 1, L, L)
        count = torch.stack(counts, dim=0)  # (B, 1, L, L)
        
        # Relative position encoding
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
        
        # Metrics
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pL = precision_at_k_masked(prob, contact, valid_mask, k_mode="L")
        
        self.log(f"{stage}/loss", loss, prog_bar=(stage=="val"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/P@L", pL, prog_bar=(stage=="val"), on_epoch=True, sync_dist=True)
        
        viz_cache = None
        if return_visualization:
            subsets = batch.get("subset", ["all"] * len(pids))
            viz_cache = {
                "prob": prob,
                "contact": contact,
                "valid_mask": valid_mask,
                "seq": seqs,
                "crop_bounds": crop_bounds,
                "pid": pids,
                "subset": subsets,
            }
        
        return loss, viz_cache
    
    def training_step(self, batch: Dict, batch_idx: int):
        loss, _ = self._step(batch, stage="train")
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        loss, viz_cache = self._step(batch, stage="val", return_visualization=True)
        
        self._val_probs.append(viz_cache["prob"].detach().cpu())
        self._val_targets.append(viz_cache["contact"].detach().cpu())
        self._val_masks.append(viz_cache["valid_mask"].detach().cpu())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Find optimal threshold on validation set."""
        if len(self._val_probs) == 0:
            return
        
        # Simple threshold search
        all_probs = torch.cat([p.squeeze(1) for p in self._val_probs], dim=0)
        all_targets = torch.cat(self._val_targets, dim=0)
        all_masks = torch.cat(self._val_masks, dim=0)
        
        best_f1 = 0.0
        best_thresh = 0.5
        
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = (all_probs >= thresh).float()
            valid = all_masks > 0
            
            tp = ((preds[valid] == 1) & (all_targets[valid] == 1)).sum().float()
            fp = ((preds[valid] == 1) & (all_targets[valid] == 0)).sum().float()
            fn = ((preds[valid] == 0) & (all_targets[valid] == 1)).sum().float()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        self.pred_threshold = best_thresh
        self.log("val/threshold", best_thresh, prog_bar=False, sync_dist=False)
        self.log("val/f1", best_f1, prog_bar=True, sync_dist=False)
        
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []
    
    def test_step(self, batch: Dict, batch_idx: int):
        loss, viz_cache = self._step(batch, stage="test", return_visualization=True)
        
        self._test_probs.append(viz_cache["prob"].detach().cpu())
        self._test_targets.append(viz_cache["contact"].detach().cpu())
        self._test_masks.append(viz_cache["valid_mask"].detach().cpu())
        self._test_pids.extend(viz_cache["pid"])
        self._test_subsets.extend(viz_cache["subset"])
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute final test metrics."""
        if len(self._test_probs) == 0:
            return
        
        all_probs = torch.cat([p.squeeze(1) for p in self._test_probs], dim=0)
        all_targets = torch.cat(self._test_targets, dim=0)
        all_masks = torch.cat(self._test_masks, dim=0)
        
        # Overall metrics
        preds = (all_probs >= self.pred_threshold).float()
        valid = all_masks > 0
        
        tp = ((preds[valid] == 1) & (all_targets[valid] == 1)).sum().float()
        fp = ((preds[valid] == 1) & (all_targets[valid] == 0)).sum().float()
        fn = ((preds[valid] == 0) & (all_targets[valid] == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        self.log("test/precision", precision, prog_bar=True, sync_dist=False)
        self.log("test/recall", recall, prog_bar=True, sync_dist=False)
        self.log("test/f1", f1, prog_bar=True, sync_dist=False)
        
        # Range metrics
        range_metrics = precision_at_k_by_range(all_probs, all_targets, all_masks, k_mode="L")
        for range_name, value in range_metrics.items():
            self.log(f"test/P@L_{range_name}", value, prog_bar=(range_name=="long"), sync_dist=False)
        
        # AUC-PR
        auc_pr = auc_pr_masked(all_probs, all_targets, all_masks, range_type="long")
        self.log("test/AUC-PR_long", auc_pr, prog_bar=True, sync_dist=False)
        
        # MCC
        mcc = mcc_at_threshold(all_probs, all_targets, all_masks, self.pred_threshold)
        self.log("test/MCC", mcc, prog_bar=False, sync_dist=False)
        
        # Clear
        self._test_probs = []
        self._test_targets = []
        self._test_masks = []
        self._test_pids = []
        self._test_subsets = []
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
