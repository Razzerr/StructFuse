from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from src.models.components.esm_backbone import ESM2Backbone
from src.models.components.contact_model import ContactModel
from src.models.components.pair2d_head import relpos_buckets
from src.models.utils.faiss import FaissIndex
from src.models.utils.loss import masked_bce_balanced, masked_focal_tversky, masked_ce_distogram
from src.models.utils.metrics import (
    precision_at_k_masked,
    precision_at_k_by_range,
    _create_range_mask,
)
from src.models.utils.visualize import (
    plot_contact_map_comparison,
    plot_precision_recall_curve,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

_INIT_THRESHOLD = 0.5

class ContactLitModule(LightningModule):
    def __init__(
        self,
        index_dir: str,
        topk: int = 4,
        only_positive_transfer: bool = False,
        min_seq_sep: int = 6,
        esm_model: str = "esm2_t33_650M_UR50D",
        finetune_esm: bool = False,
        use_blosum: bool = True,
        fusion_strategy: str = "standard",  # standard or trufor
        fusion_num_heads: int = 8,
        fusion_reduction: int = 1,
        head_type: str = "cnn",  # cnn, dilated or axial
        head_num_heads: int = 8,  # for axial attention head
        head_num_kv_heads: int = None,  # GQA: KV heads (None = same as head_num_heads)
        alternating_axial: bool = False,  # Alternate row/col attention per block
        d_esm: int = 1280,
        d_pair: int = 128,
        width: int = 128,
        depth: int = 8,
        rank: int = 32,
        lr: float = 2e-4,
        wd: float = 1e-2,
        pos_weight_scale: float = 1.0,
        label_smoothing: float = 0.1,
        # Tversky loss parameters
        use_tversky: bool = False,
        tversky_weight: float = 0.3,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 1.0,
        max_tpl_cache: int = 1000,
        n_dist_bins: int = 0,
        n_ss_feat: int = 0,
        # Distogram output: predict distance bins instead of binary contacts
        distogram: bool = False,
        cb_beta: float = 0.0,  # Effective-number class balancing for distogram (0=off, 0.999=strong)
        # Scheduler parameters
        warmup_steps: int = 1000,
        total_steps: int = 0,  # 0 = auto-calculate from trainer
        min_lr_ratio: float = 0.01,  # min_lr = lr * min_lr_ratio
        cosine_restarts: bool = False,  # Enable SGDR warm restarts
        restart_period: int = 1000,     # T_0: first cycle length in steps
        restart_mult: float = 2.0,      # T_mult: cycle length multiplier (>=1)
        restart_decay: float = 1.0,     # Peak LR decay per cycle (0.75 = 75% of prev peak)
        # Ablation parameters for retrieval
        min_template_similarity: float = 0.0,  # Filter templates below this similarity
        random_retrieval: bool = False,  # Use random templates instead of FAISS
        # Compilation
        compile_model: bool = False,  # torch.compile (use dynamic=True for variable-length inputs)
        # Gradient checkpointing
        use_checkpoint: bool = False,  # Activation checkpointing for axial attention blocks
        # Test visualizations
        save_test_viz: bool = False,  # Save contact map PNGs during test
    ):
        super().__init__()
        self.save_hyperparameters()

        self.esm = ESM2Backbone(esm_model, finetune=finetune_esm)
        self.esm_alphabet = self.esm.alphabet

        # Distogram: predict N distance bins instead of binary contact
        from src.data.components.dataset import N_DIST_CLASSES, CONTACT_BIN_THRESHOLD
        self.distogram = bool(distogram)
        self.cb_beta = float(cb_beta)
        self.n_dist_classes = N_DIST_CLASSES if self.distogram else 1
        self.contact_bin_threshold = CONTACT_BIN_THRESHOLD  # bins 0..K-1 = contact

        self.net = ContactModel(
            d_esm=d_esm,
            d_pair=d_pair,
            width=width,
            depth=depth,
            rank=rank,
            fusion_strategy=fusion_strategy,
            fusion_num_heads=fusion_num_heads,
            fusion_reduction=fusion_reduction,
            n_dist_bins=n_dist_bins,
            n_ss_feat=n_ss_feat,
            n_out=self.n_dist_classes,
            head_type=head_type,
            head_num_heads=head_num_heads,
            head_num_kv_heads=head_num_kv_heads,
            alternating_axial=alternating_axial,
            use_checkpoint=use_checkpoint,
        )
        if compile_model:
            self.net = torch.compile(self.net, dynamic=True)

        self._index_dir = index_dir
        self._index: Optional["FaissIndex"] = None  # lazy-loaded
        self.topk = int(topk)
        self.min_seq_sep = int(min_seq_sep)
        self.only_positive_transfer = bool(only_positive_transfer)
        self.use_blosum = bool(use_blosum)

        self.lr = float(lr)
        self.wd = float(wd)

        self.pos_weight_scale = float(pos_weight_scale)
        self.label_smoothing = float(label_smoothing)
        self.pred_threshold = _INIT_THRESHOLD

        # Tversky loss parameters
        self.use_tversky = bool(use_tversky)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_gamma = float(tversky_gamma)

        # Scheduler parameters
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.cosine_restarts = bool(cosine_restarts)
        self.restart_period = int(restart_period)
        self.restart_mult = float(restart_mult)
        self.restart_decay = float(restart_decay)

        # Ablation parameters
        self.min_template_similarity = float(min_template_similarity)
        self.random_retrieval = bool(random_retrieval)

        # For visualization logging
        self._val_viz_logged = False  # Log one viz per validation epoch
        self._viz_protein_id = None  # Track same protein across epochs

        # ── Streaming validation metrics ──
        # 20 thresholds for F1/precision/recall/MCC search
        self._val_thresholds = torch.linspace(0.05, 0.99, 20)
        # Per-range TP/FP/FN/TN: keys = "short", "medium", "long"
        self._val_range_tp: Dict[str, torch.Tensor] = {}   # each (20,)
        self._val_range_fp: Dict[str, torch.Tensor] = {}
        self._val_range_fn: Dict[str, torch.Tensor] = {}
        self._val_range_tn: Dict[str, torch.Tensor] = {}
        # Per-batch P@L by range (accumulated for averaging)
        self._val_pL_range: Dict[str, List[float]] = {}
        # Subsampled (prob, target) pairs for AUC-PR per range (capped total)
        self._val_auc_probs: Dict[str, List[np.ndarray]] = {}
        self._val_auc_targets: Dict[str, List[np.ndarray]] = {}
        self._val_auc_n_pairs: Dict[str, int] = {}
        self._auc_max_pairs = 500_000  # cap total pairs for AUC-PR per range
        self._range_defs = {"short": (6, 12), "medium": (12, 24), "long": (24, None)}

        # ── Streaming test metrics (mirrors validation, avoids pad+concat) ──
        # Accumulators are reset in on_test_epoch_start(); declared here
        # only for type-annotation reference.
        self._test_range_tp: Dict[str, torch.Tensor] = {}
        self._test_range_fp: Dict[str, torch.Tensor] = {}
        self._test_range_fn: Dict[str, torch.Tensor] = {}
        self._test_range_tn: Dict[str, torch.Tensor] = {}
        self._test_pL_range: Dict[str, List[float]] = {}
        self._test_auc_probs: Dict[str, List[np.ndarray]] = {}
        self._test_auc_targets: Dict[str, List[np.ndarray]] = {}
        self._test_auc_n_pairs: Dict[str, int] = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0
        self._test_subset_tp: Dict[str, float] = {}
        self._test_subset_fp: Dict[str, float] = {}
        self._test_subset_fn: Dict[str, float] = {}
        self._test_subset_pL: Dict[str, List[float]] = {}
        self._test_subset_auc_probs: Dict[str, List[np.ndarray]] = {}
        self._test_subset_auc_targets: Dict[str, List[np.ndarray]] = {}
        self._test_subset_count: Dict[str, int] = {}
        self._test_per_sample: List[Dict] = []
        self._test_pids: List[str] = []
        self._test_subsets: List[str] = []
        self._n_non_casp_viz_saved = 0

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)

        # Calculate total steps if not provided
        if self.total_steps <= 0:
            # Try to get from trainer
            if self.trainer is not None and self.trainer.estimated_stepping_batches is not None:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback: assume 10000 steps
                total_steps = 10000
        else:
            total_steps = self.total_steps

        # Linear warmup + cosine annealing scheduler (optionally with warm restarts)
        _warmup = self.warmup_steps
        _min_ratio = self.min_lr_ratio
        _restarts = self.cosine_restarts
        _T0 = self.restart_period
        _Tmult = self.restart_mult
        _decay = self.restart_decay

        def lr_lambda(current_step: int) -> float:
            if current_step < _warmup:
                return float(current_step) / float(max(1, _warmup))

            step = current_step - _warmup

            if _restarts:
                # SGDR: cosine annealing with warm restarts + amplitude decay
                if _Tmult == 1.0:
                    cycle_idx = step // _T0
                    cycle_pos = (step % _T0) / float(_T0)
                else:
                    t_cur = _T0
                    cumul = 0
                    cycle_idx = 0
                    while cumul + t_cur <= step:
                        cumul += t_cur
                        t_cur = int(t_cur * _Tmult)
                        cycle_idx += 1
                    cycle_pos = (step - cumul) / float(max(1, t_cur))
                # Decay peak amplitude: peak_k = decay^k
                amplitude = _decay ** cycle_idx
            else:
                # Standard single cosine decay over remaining steps
                cycle_pos = step / float(max(1, total_steps - _warmup))
                amplitude = 1.0

            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * min(cycle_pos, 1.0)))
            return max(_min_ratio, amplitude * cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def state_dict(self):
        """Save state including the optimal threshold"""
        state = super().state_dict()
        # Add the current optimal threshold to the state
        state['pred_threshold'] = self.pred_threshold
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state including the optimal threshold"""
        # Extract and set the optimal threshold if it exists
        if 'pred_threshold' in state_dict:
            self.pred_threshold = state_dict.pop('pred_threshold')
            log.info(f"Loaded pred_threshold={self.pred_threshold:.4f} from state_dict")
        
        # Load the rest of the state
        return super().load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint):
        """Fallback: restore pred_threshold from checkpoint top-level or state_dict."""
        sd = checkpoint.get("state_dict", {})
        if "pred_threshold" in sd:
            self.pred_threshold = sd["pred_threshold"]
            log.info(f"on_load_checkpoint: restored pred_threshold={self.pred_threshold:.4f}")
        elif "pred_threshold" in checkpoint:
            self.pred_threshold = checkpoint["pred_threshold"]
            log.info(f"on_load_checkpoint: restored pred_threshold={self.pred_threshold:.4f} from top-level")

    def _get_embedding(
        self, pids: List[str], seqs: List[str], crop_bounds: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ESM embeddings and contacts for a batch of cropped regions.

        Passes all cropped subsequences to ESM2 in a single batched forward
        call — avoids B separate forward passes.

        Returns:
            Tuple of (h, esm_contacts) where:
                h: (B, L_max, D) padded embeddings
                esm_contacts: (B, 1, L_max, L_max) padded contacts
        """
        seq_list = [
            (pid, seq[b0:b1])
            for pid, seq, (b0, b1) in zip(pids, seqs, crop_bounds)
        ]
        reps, contacts = self.esm(seq_list, self.device)  # (B, L_max, D), (B, 1, L_max, L_max)
        return reps, contacts

    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        return_visualization: bool = False,
    ) -> Dict[str, float]:
        """
        batch from DataModule collate_padded:
          - pid: list[str]
          - seq: list[str]
          - contact: (B, Lmax, Lmax) float
          - pair_mask: (B, Lmax, Lmax) float
          - long_mask: (B, Lmax, Lmax) float
          - prior: (B, 1, Lmax, Lmax) float  (built in DataLoader workers)
          - count: (B, 1, Lmax, Lmax) float  (built in DataLoader workers)
        """
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]  # (B, 2)
        contact = batch["contact"].to(self.device)  # (B, Lmax, Lmax)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)

        # Template priors (already built in DataLoader workers on CPU)
        prior = batch["prior"].to(self.device)   # (B, 1, Lmax, Lmax)
        count = batch["count"].to(self.device)   # (B, 1, Lmax, Lmax)
        dist_bins = batch.get("dist_bins")       # (B, N, Lmax, Lmax) or None
        if dist_bins is not None:
            dist_bins = dist_bins.to(self.device)
        ss_feat = batch.get("ss_feat")           # (B, S, Lmax, Lmax) or None
        if ss_feat is not None:
            ss_feat = ss_feat.to(self.device)

        # Batched ESM2 forward — single call for all samples
        # If precomputed embeddings are in the batch, skip ESM2 entirely
        if "h_esm" in batch and "esm_contacts" in batch:
            h = batch["h_esm"].to(self.device)
            esm_contacts = batch["esm_contacts"].to(self.device)
        else:
            h, esm_contacts = self._get_embedding(pids, seqs, crop_bounds)

        Lmax = h.shape[1]

        rel = relpos_buckets(Lmax, self.device)  # (R,L,L)
        # broadcast to batch and mask (no big expands needed)
        valid = (pair_mask * long_mask).unsqueeze(1)  # (B,1,L,L)
        rel = rel.unsqueeze(0) * valid  # (B,R,L,L) via broadcast

        logits = self.net(h, prior, count, rel, esm_contacts, pair_mask=pair_mask.unsqueeze(1), dist_bins=dist_bins, ss_feat=ss_feat)
        # logits: (B, n_out, Lmax, Lmax) — n_out=1 binary, n_out=N distogram

        valid_mask = valid.squeeze(1)  # (B, L, L) — reuse already-computed product

        if self.distogram:
            # ── Distogram mode: CE loss over distance bins ──
            dist_target = batch["dist_target"].to(self.device)  # (B, Lmax, Lmax) int64
            loss = masked_ce_distogram(
                logits, dist_target, valid_mask,
                label_smoothing=self.label_smoothing,
                cb_beta=self.cb_beta,
            )
        else:
            # ── Binary mode: BCE loss ──
            loss_bce = masked_bce_balanced(
                logits,
                contact,
                valid_mask,
                pos_weight_scale=self.pos_weight_scale,
                label_smoothing=self.label_smoothing,
            )

            # Optional Tversky loss
            if self.use_tversky:
                loss_tversky = masked_focal_tversky(
                    logits,
                    contact,
                    valid_mask,
                    alpha=self.tversky_alpha,
                    beta=self.tversky_beta,
                    gamma=self.tversky_gamma,
                )
                loss = loss_bce + self.tversky_weight * loss_tversky
            else:
                loss = loss_bce

        if stage != "train" or self.trainer.global_step % 10 == 0:
            with torch.no_grad():
                if self.distogram:
                    # Derive contact probability: sum softmax probs for bins < 8Å
                    dist_probs = torch.softmax(logits, dim=1)  # (B, N, L, L)
                    prob = dist_probs[:, :self.contact_bin_threshold].sum(dim=1, keepdim=True)  # (B, 1, L, L)
                else:
                    prob = torch.sigmoid(logits)
                pL = precision_at_k_masked(prob, contact, valid_mask, k_mode="L")
                pL2 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/2")
                pL5 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/5")

            prog_bar = stage == "val"
            on_step = stage == "train"
            on_epoch = True
            self.log(
                f"{stage}/loss",
                loss,
                prog_bar=True,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L",
                pL,
                prog_bar=prog_bar,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L2",
                pL2,
                prog_bar=False,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L5",
                pL5,
                prog_bar=False,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
            if self.use_tversky and not self.distogram:
                self.log(
                    f"{stage}/loss_tversky",
                    loss_tversky,
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/loss_bce",
                    loss_bce,
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    sync_dist=True,
                )

        viz_cache = None
        if return_visualization:
            with torch.no_grad():
                if self.distogram:
                    dist_probs = torch.softmax(logits, dim=1)
                    prob = dist_probs[:, :self.contact_bin_threshold].sum(dim=1, keepdim=True)
                else:
                    prob = torch.sigmoid(logits)
            
            # Get subset info from batch (if available)
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

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, _ = self._step(batch, stage="train")

        # ── NaN / Inf guard ──────────────────────────────────────────
        loss_val = loss.item()
        if not (loss_val == loss_val) or loss_val == float("inf"):  # fast NaN check
            pids = batch["pid"]
            seqs_len = [len(s) for s in batch["seq"]]
            log.error(
                f"NaN/Inf loss detected at step {self.trainer.global_step}, "
                f"batch_idx={batch_idx}, pids={pids}, seq_lens={seqs_len}"
            )
            raise ValueError(
                f"Training stopped: loss is {'NaN' if loss_val != loss_val else 'Inf'} "
                f"at global_step={self.trainer.global_step} "
                f"(pids={pids}, seq_lens={seqs_len})"
            )

        # ── Console heartbeat (every 50 steps) ────────────────────────
        if batch_idx % 50 == 0:
            seqs_len = [len(s) for s in batch["seq"]]
            log.info(
                f"  [step {self.trainer.global_step:>5d} | batch {batch_idx:>3d}] "
                f"loss={loss_val:.4f}  seq_lens={seqs_len}"
            )

        # Log learning rate
        if self.trainer.global_step % 10 == 0:
            opt = self.optimizers()
            current_lr = opt.param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient statistics before optimizer step for debugging training stability.
        
        Uses vectorized operations with only 3 GPU→CPU syncs total (instead of
        ~3000 per-param syncs in the naive loop).
        """
        if self.trainer.global_step % 50 == 0:
            # Incremental norm — avoids allocating one huge flat tensor for all grads
            total_norm_sq = 0.0
            has_nan = False
            has_inf = False
            for p in self.net.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                total_norm_sq += g.norm(2).item() ** 2
                if not has_nan:
                    has_nan = bool(torch.isnan(g).any().item())
                if not has_inf:
                    has_inf = bool(torch.isinf(g).any().item())
            total_norm = total_norm_sq ** 0.5
            
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False)
            
            if has_nan or has_inf:
                self.log("train/grad_has_nan", float(has_nan), on_step=True, on_epoch=False)
                self.log("train/grad_has_inf", float(has_inf), on_step=True, on_epoch=False)

    def _find_optimal_threshold(
        self, prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Find optimal threshold that maximizes F1 score.

        Args:
            prob: (B, L, L) prediction probabilities
            target: (B, L, L) ground truth contacts
            mask: (B, L, L) valid pair mask

        Returns:
            Tuple of (best_threshold, best_f1)
        """
        # Flatten and filter by mask
        p = prob[mask > 0].flatten().cpu()
        t = target[mask > 0].flatten().cpu()

        # Try thresholds from 0.05 to 0.99
        thresholds = torch.linspace(0.05, 0.99, 20)
        best_f1 = 0.0
        best_threshold = 0.5

        for tau in thresholds:
            pred = (p >= tau).float()
            tp = (pred * t).sum().item()
            fp = (pred * (1 - t)).sum().item()
            fn = ((1 - pred) * t).sum().item()

            # Compute F1
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(tau)

        return best_threshold, best_f1

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Log visualization for first batch of each validation epoch
        if batch_idx == 0 and not self._val_viz_logged:
            loss, viz_cache = self._step(batch, stage="val", return_visualization=True)
            self._log_visualization(viz_cache)
            self._val_viz_logged = True
        else:
            loss, viz_cache = self._step(batch, stage="val", return_visualization=True)

        # ── Streaming metric accumulation (scalars only, no tensor storage) ──
        prob = viz_cache["prob"].detach()       # (B, 1, L, L) or (B, L, L)
        contact = viz_cache["contact"].detach() # (B, L, L)
        mask = viz_cache["valid_mask"].detach() # (B, L, L)

        if prob.dim() == 4:
            prob = prob.squeeze(1)  # (B, L, L)

        B, L, _ = prob.shape
        thresholds = self._val_thresholds.to(prob.device)  # (20,)

        # Per-range accumulation of TP/FP/FN/TN and AUC-PR pairs
        for rname, (min_sep, max_sep) in self._range_defs.items():
            range_mask = _create_range_mask(L, min_sep, max_sep, prob.device).float()
            combined = mask * range_mask.unsqueeze(0)  # (B, L, L)

            p_flat = prob[combined > 0].flatten()     # (N_range,)
            t_flat = contact[combined > 0].flatten()   # (N_range,)

            if p_flat.numel() == 0:
                continue

            # TP/FP/FN/TN at each threshold
            preds = (p_flat.unsqueeze(0) >= thresholds.unsqueeze(1)).float()  # (20, N)
            t_exp = t_flat.unsqueeze(0).expand_as(preds)
            tp = (preds * t_exp).sum(dim=1)              # (20,)
            fp = (preds * (1 - t_exp)).sum(dim=1)
            fn = ((1 - preds) * t_exp).sum(dim=1)
            tn = ((1 - preds) * (1 - t_exp)).sum(dim=1)

            if rname not in self._val_range_tp:
                self._val_range_tp[rname] = tp
                self._val_range_fp[rname] = fp
                self._val_range_fn[rname] = fn
                self._val_range_tn[rname] = tn
            else:
                self._val_range_tp[rname] += tp
                self._val_range_fp[rname] += fp
                self._val_range_fn[rname] += fn
                self._val_range_tn[rname] += tn

            # Subsample pairs for AUC-PR per range
            n_so_far = self._val_auc_n_pairs.get(rname, 0)
            if n_so_far < self._auc_max_pairs:
                p_cpu = p_flat.cpu().float().numpy()
                t_cpu = t_flat.cpu().float().numpy()
                budget = self._auc_max_pairs - n_so_far
                n = len(p_cpu)
                if n > budget:
                    idx = np.random.choice(n, int(budget), replace=False)
                    p_cpu = p_cpu[idx]
                    t_cpu = t_cpu[idx]
                self._val_auc_probs.setdefault(rname, []).append(p_cpu)
                self._val_auc_targets.setdefault(rname, []).append(t_cpu)
                self._val_auc_n_pairs[rname] = n_so_far + len(p_cpu)

        # P@L by range (per-protein average, lightweight)
        range_metrics = precision_at_k_by_range(prob, contact, mask, k_mode="L")
        for rname, val in range_metrics.items():
            self._val_pL_range.setdefault(rname, []).append(val)

        return loss

    def on_validation_epoch_start(self):
        """Reset streaming accumulators at start of each validation epoch."""
        self._val_viz_logged = False
        self._val_range_tp = {}
        self._val_range_fp = {}
        self._val_range_fn = {}
        self._val_range_tn = {}
        self._val_pL_range = {}
        self._val_auc_probs = {}
        self._val_auc_targets = {}
        self._val_auc_n_pairs = {}

    def on_validation_epoch_end(self):
        """Aggregate streaming stats into final metrics — zero large tensor allocations."""
        if not self._val_range_tp:
            return

        # Skip during sanity check
        if self.trainer.sanity_checking:
            return

        # For each range: compute F1, precision, recall, MCC, AUC-PR
        for rname in ("short", "medium", "long"):
            if rname not in self._val_range_tp:
                continue

            tp = self._val_range_tp[rname].cpu()
            fp = self._val_range_fp[rname].cpu()
            fn = self._val_range_fn[rname].cpu()
            tn = self._val_range_tn[rname].cpu()

            # F1 / precision / recall at best-F1 threshold
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)  # (20,)
            best_idx = int(f1.argmax())
            best_f1 = f1[best_idx].item()
            best_thresh = self._val_thresholds[best_idx].item()
            prec = (tp[best_idx] / (tp[best_idx] + fp[best_idx] + 1e-8)).item()
            rec = (tp[best_idx] / (tp[best_idx] + fn[best_idx] + 1e-8)).item()

            is_long = rname == "long"
            self.log(f"val/f1_{rname}", best_f1, prog_bar=is_long, sync_dist=False)
            self.log(f"val/precision_{rname}", prec, prog_bar=False, sync_dist=False)
            self.log(f"val/recall_{rname}", rec, prog_bar=False, sync_dist=False)
            self.log(f"val/threshold_{rname}", best_thresh, prog_bar=False, sync_dist=False)

            # MCC at best-F1 threshold & best-MCC threshold
            mcc_num = tp * tn - fp * fn
            mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            mcc_all = mcc_num / mcc_den  # (20,)
            mcc_at_f1 = mcc_all[best_idx].item()
            mcc_best_idx = int(mcc_all.argmax())
            self.log(f"val/MCC_{rname}", mcc_at_f1, prog_bar=False, sync_dist=False)
            self.log(f"val/MCC_optimal_{rname}", mcc_all[mcc_best_idx].item(), prog_bar=False, sync_dist=False)

            # AUC-PR from subsampled pairs
            if rname in self._val_auc_probs and self._val_auc_probs[rname]:
                from sklearn.metrics import average_precision_score
                all_p = np.concatenate(self._val_auc_probs[rname])
                all_t = np.concatenate(self._val_auc_targets[rname])
                if all_t.sum() > 0 and all_t.sum() < len(all_t):
                    auc = float(average_precision_score(all_t, all_p))
                else:
                    auc = 0.0
                self.log(f"val/AUC-PR_{rname}", auc, prog_bar=is_long, sync_dist=False)

        # P@L by range (average of per-batch averages)
        for rname, vals in self._val_pL_range.items():
            avg = float(np.mean(vals)) if vals else 0.0
            self.log(f"val/P@L_{rname}", avg, prog_bar=(rname == "long"), sync_dist=False)

        # Set pred_threshold from long-range (headline metric)
        if "long" in self._val_range_tp:
            tp_l = self._val_range_tp["long"].cpu()
            fp_l = self._val_range_fp["long"].cpu()
            fn_l = self._val_range_fn["long"].cpu()
            f1_l = 2 * tp_l / (2 * tp_l + fp_l + fn_l + 1e-8)
            self.pred_threshold = self._val_thresholds[int(f1_l.argmax())].item()
        self.log("val/optimal_threshold", self.pred_threshold, prog_bar=True, sync_dist=False)

    def on_test_epoch_end(self):
        """Aggregate streaming test stats into final metrics — no pad+concat needed."""
        if not self._test_range_tp:
            log.warning("No test predictions accumulated for metrics computation.")
            return

        # ── Global precision / recall / F1 at pred_threshold ──
        tp = self._test_tp
        fp = self._test_fp
        fn = self._test_fn
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        self.log("test/precision", precision, prog_bar=True, sync_dist=False)
        self.log("test/recall", recall, prog_bar=True, sync_dist=False)
        self.log("test/f1", f1, prog_bar=True, sync_dist=False)
        self.log("test/threshold_used", self.pred_threshold, prog_bar=False, sync_dist=False)

        # ── Per-range metrics (same aggregation as validation) ──
        for rname in ("short", "medium", "long"):
            if rname not in self._test_range_tp:
                continue
            tp_r = self._test_range_tp[rname].cpu()
            fp_r = self._test_range_fp[rname].cpu()
            fn_r = self._test_range_fn[rname].cpu()
            tn_r = self._test_range_tn[rname].cpu()

            f1_r = 2 * tp_r / (2 * tp_r + fp_r + fn_r + 1e-8)
            best_idx = int(f1_r.argmax())
            best_f1 = f1_r[best_idx].item()
            prec_r = (tp_r[best_idx] / (tp_r[best_idx] + fp_r[best_idx] + 1e-8)).item()
            rec_r = (tp_r[best_idx] / (tp_r[best_idx] + fn_r[best_idx] + 1e-8)).item()

            is_long = rname == "long"
            self.log(f"test/f1_{rname}", best_f1, prog_bar=is_long, sync_dist=False)
            self.log(f"test/precision_{rname}", prec_r, prog_bar=False, sync_dist=False)
            self.log(f"test/recall_{rname}", rec_r, prog_bar=False, sync_dist=False)

            # MCC
            mcc_num = tp_r * tn_r - fp_r * fn_r
            mcc_den = torch.sqrt((tp_r + fp_r) * (tp_r + fn_r) * (tn_r + fp_r) * (tn_r + fn_r) + 1e-8)
            mcc_all = mcc_num / mcc_den
            self.log(f"test/MCC_{rname}", mcc_all[best_idx].item(), prog_bar=False, sync_dist=False)

            # AUC-PR from subsampled pairs
            if rname in self._test_auc_probs and self._test_auc_probs[rname]:
                from sklearn.metrics import average_precision_score
                all_p = np.concatenate(self._test_auc_probs[rname])
                all_t = np.concatenate(self._test_auc_targets[rname])
                if all_t.sum() > 0 and all_t.sum() < len(all_t):
                    auc = float(average_precision_score(all_t, all_p))
                else:
                    auc = 0.0
                self.log(f"test/AUC-PR_{rname}", auc, prog_bar=is_long, sync_dist=False)

        # P@L by range
        for rname, vals in self._test_pL_range.items():
            avg = float(np.mean(vals)) if vals else 0.0
            self.log(f"test/P@L_{rname}", avg, prog_bar=(rname == "long"), sync_dist=False)

        # ── Per-subset evaluation (streaming) ──
        self._log_per_subset_metrics()

        # ── Export per-sample metrics to TSV ──
        self._export_per_sample_metrics()

        # Clear accumulators
        self._test_range_tp = {}
        self._test_range_fp = {}
        self._test_range_fn = {}
        self._test_range_tn = {}
        self._test_pL_range = {}
        self._test_auc_probs = {}
        self._test_auc_targets = {}
        self._test_auc_n_pairs = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0
        self._test_subset_tp = {}
        self._test_subset_fp = {}
        self._test_subset_fn = {}
        self._test_subset_pL = {}
        self._test_subset_auc_probs = {}
        self._test_subset_auc_targets = {}
        self._test_subset_count = {}
        self._test_per_sample = []
        self._test_pids = []
        self._test_subsets = []
        self._n_non_casp_viz_saved = 0
    
    def _log_per_subset_metrics(self):
        """Log metrics for each test subset from streaming accumulators."""
        from src.data.components.dataset import SUBSET_GOLD, SUBSET_CASP16, SUBSET_CLUSTER_PROMOTED

        for subset_name in [SUBSET_GOLD, SUBSET_CASP16, SUBSET_CLUSTER_PROMOTED]:
            n = self._test_subset_count.get(subset_name, 0)
            if n == 0:
                log.info(f"No samples in subset '{subset_name}', skipping.")
                continue

            s_tp = self._test_subset_tp.get(subset_name, 0.0)
            s_fp = self._test_subset_fp.get(subset_name, 0.0)
            s_fn = self._test_subset_fn.get(subset_name, 0.0)

            prec = s_tp / max(1, s_tp + s_fp)
            rec = s_tp / max(1, s_tp + s_fn)
            f1_val = 2 * prec * rec / max(1e-8, prec + rec)

            pl_long = float(np.mean(self._test_subset_pL.get(subset_name, [0.0])))

            auc_pr = 0.0
            if subset_name in self._test_subset_auc_probs and self._test_subset_auc_probs[subset_name]:
                from sklearn.metrics import average_precision_score
                all_p = np.concatenate(self._test_subset_auc_probs[subset_name])
                all_t = np.concatenate(self._test_subset_auc_targets[subset_name])
                if all_t.sum() > 0 and all_t.sum() < len(all_t):
                    auc_pr = float(average_precision_score(all_t, all_p))

            prefix = f"test/{subset_name}"
            self.log(f"{prefix}/n_samples", float(n), prog_bar=False, sync_dist=False)
            self.log(f"{prefix}/precision", prec, prog_bar=False, sync_dist=False)
            self.log(f"{prefix}/recall", rec, prog_bar=False, sync_dist=False)
            self.log(f"{prefix}/f1", f1_val, prog_bar=False, sync_dist=False)
            self.log(f"{prefix}/P@L_long", pl_long, prog_bar=True, sync_dist=False)
            self.log(f"{prefix}/AUC-PR_long", auc_pr, prog_bar=False, sync_dist=False)

            log.info(f"Subset '{subset_name}': n={n}, P@L_long={pl_long:.4f}, F1={f1_val:.4f}, AUC-PR={auc_pr:.4f}")

    def _export_per_sample_metrics(self):
        """Export per-sample metrics to TSV for qualitative analysis."""
        from pathlib import Path
        from hydra.core.hydra_config import HydraConfig

        try:
            hydra_cfg = HydraConfig.get()
            log_dir = Path(hydra_cfg.runtime.output_dir)
        except Exception:
            if self.trainer.log_dir is not None and not str(self.trainer.log_dir).startswith('.neptune'):
                log_dir = Path(self.trainer.log_dir)
            else:
                log_dir = Path("results")

        output_path = log_dir / "per_sample_metrics.tsv"

        if self._test_per_sample:
            with open(output_path, "w") as f:
                headers = list(self._test_per_sample[0].keys())
                f.write("\t".join(headers) + "\n")
                for row in self._test_per_sample:
                    f.write("\t".join(str(row[h]) for h in headers) + "\n")
            log.info(f"Per-sample metrics saved to {output_path}")

    def _log_visualization(self, viz_cache: Dict):
        """Log a contact map visualization for the first sample in batch"""
        # Extract relevant data from cache
        idx = 0  # First sample

        seq = viz_cache["seq"][idx]
        crop_bounds = viz_cache["crop_bounds"][idx]
        pid = viz_cache["pid"][idx]

        # Get cropped sequence
        seq_crop = seq[crop_bounds[0] : crop_bounds[1]]
        L = len(seq_crop)

        # Crop tensors to actual length (remove padding)
        prob = viz_cache["prob"][idx, 0, :L, :L]  # (L, L)
        contact = viz_cache["contact"][idx, :L, :L]  # (L, L)
        valid_mask = viz_cache["valid_mask"][idx, :L, :L]  # (L, L)

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
                        "val/contact_map",
                        fig,
                        global_step=self.current_epoch,
                    )
                    self.logger.experiment.add_figure(
                        "val/pr_curve",
                        prc_fig,
                        global_step=self.current_epoch,
                    )
                elif logger_name == "NeptuneLogger":
                    # Neptune - log as series with step for slider
                    import neptune.types as neptune_types

                    self.logger.experiment["val/contact_map"].append(
                        neptune_types.File.as_image(img), step=self.current_epoch
                    )
                    self.logger.experiment["val/pr_curve"].append(
                        neptune_types.File.as_image(prc_img), step=self.current_epoch
                    )
                else:
                    # Fallback: try generic log_image if available
                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(
                            key="val/contact_map",
                            images=[img],
                            step=self.current_epoch,
                        )
                        self.logger.log_image(
                            key="val/pr_curve",
                            images=[prc_img],
                            step=self.current_epoch,
                        )
            except Exception as e:
                log.warning(f"Failed to log visualization with {logger_name}: {e}")

            buf.close()
            prc_buf.close()

        plt.close(fig)
        plt.close(prc_fig)

    def on_test_epoch_start(self):
        """Reset streaming test accumulators."""
        self._test_range_tp = {}
        self._test_range_fp = {}
        self._test_range_fn = {}
        self._test_range_tn = {}
        self._test_pL_range = {}
        self._test_auc_probs = {}
        self._test_auc_targets = {}
        self._test_auc_n_pairs = {}
        self._test_tp = 0.0
        self._test_fp = 0.0
        self._test_fn = 0.0
        self._test_subset_tp = {}
        self._test_subset_fp = {}
        self._test_subset_fn = {}
        self._test_subset_pL = {}
        self._test_subset_auc_probs = {}
        self._test_subset_auc_targets = {}
        self._test_subset_count = {}
        self._test_per_sample = []
        self._test_pids = []
        self._test_subsets = []
        self._n_non_casp_viz_saved = 0

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="test", return_visualization=True)

        prob = viz_cache["prob"].detach()       # (B, 1, L, L) or (B, L, L)
        contact = viz_cache["contact"].detach() # (B, L, L)
        mask = viz_cache["valid_mask"].detach() # (B, L, L)
        pids = viz_cache["pid"]
        subsets = viz_cache.get("subset", ["all"] * len(pids))

        if prob.dim() == 4:
            prob = prob.squeeze(1)

        self._test_pids.extend(pids)
        self._test_subsets.extend(subsets)

        B, L, _ = prob.shape
        thresholds = self._val_thresholds.to(prob.device)

        # ── Global threshold-level TP/FP/FN (at self.pred_threshold) ──
        preds_bin = (prob >= self.pred_threshold).float()
        valid_preds = preds_bin[mask > 0]
        valid_targets = contact[mask > 0]
        self._test_tp += ((valid_preds == 1) & (valid_targets == 1)).sum().item()
        self._test_fp += ((valid_preds == 1) & (valid_targets == 0)).sum().item()
        self._test_fn += ((valid_preds == 0) & (valid_targets == 1)).sum().item()

        # ── Per-range streaming (same pattern as validation_step) ──
        for rname, (min_sep, max_sep) in self._range_defs.items():
            range_mask = _create_range_mask(L, min_sep, max_sep, prob.device).float()
            combined = mask * range_mask.unsqueeze(0)
            p_flat = prob[combined > 0].flatten()
            t_flat = contact[combined > 0].flatten()
            if p_flat.numel() == 0:
                continue

            preds_r = (p_flat.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
            t_exp = t_flat.unsqueeze(0).expand_as(preds_r)
            tp = (preds_r * t_exp).sum(dim=1)
            fp = (preds_r * (1 - t_exp)).sum(dim=1)
            fn = ((1 - preds_r) * t_exp).sum(dim=1)
            tn = ((1 - preds_r) * (1 - t_exp)).sum(dim=1)

            if rname not in self._test_range_tp:
                self._test_range_tp[rname] = tp
                self._test_range_fp[rname] = fp
                self._test_range_fn[rname] = fn
                self._test_range_tn[rname] = tn
            else:
                self._test_range_tp[rname] += tp
                self._test_range_fp[rname] += fp
                self._test_range_fn[rname] += fn
                self._test_range_tn[rname] += tn

            # Subsample for AUC-PR
            n_so_far = self._test_auc_n_pairs.get(rname, 0)
            if n_so_far < self._auc_max_pairs:
                p_cpu = p_flat.cpu().float().numpy()
                t_cpu = t_flat.cpu().float().numpy()
                budget = self._auc_max_pairs - n_so_far
                n = len(p_cpu)
                if n > budget:
                    idx = np.random.choice(n, int(budget), replace=False)
                    p_cpu, t_cpu = p_cpu[idx], t_cpu[idx]
                self._test_auc_probs.setdefault(rname, []).append(p_cpu)
                self._test_auc_targets.setdefault(rname, []).append(t_cpu)
                self._test_auc_n_pairs[rname] = n_so_far + len(p_cpu)

        # P@L by range
        range_metrics = precision_at_k_by_range(prob, contact, mask, k_mode="L")
        for rname, val in range_metrics.items():
            self._test_pL_range.setdefault(rname, []).append(val)

        # ── Per-subset streaming ──
        for b_idx in range(B):
            subset = subsets[b_idx]
            prob_b = prob[b_idx]
            contact_b = contact[b_idx]
            mask_b = mask[b_idx]
            pred_b = (prob_b >= self.pred_threshold).float()
            vp = pred_b[mask_b > 0]
            vt = contact_b[mask_b > 0]
            if vp.numel() == 0:
                continue
            s_tp = ((vp == 1) & (vt == 1)).sum().item()
            s_fp = ((vp == 1) & (vt == 0)).sum().item()
            s_fn = ((vp == 0) & (vt == 1)).sum().item()
            self._test_subset_tp[subset] = self._test_subset_tp.get(subset, 0.0) + s_tp
            self._test_subset_fp[subset] = self._test_subset_fp.get(subset, 0.0) + s_fp
            self._test_subset_fn[subset] = self._test_subset_fn.get(subset, 0.0) + s_fn
            self._test_subset_count[subset] = self._test_subset_count.get(subset, 0) + 1

            # Per-subset P@L long
            prob_3d = prob_b.unsqueeze(0)
            contact_3d = contact_b.unsqueeze(0)
            mask_3d = mask_b.unsqueeze(0)
            rm = precision_at_k_by_range(prob_3d, contact_3d, mask_3d, k_mode="L")
            pl_long = rm.get("long", 0.0)
            if isinstance(pl_long, torch.Tensor):
                pl_long = pl_long.item()
            self._test_subset_pL.setdefault(subset, []).append(pl_long)

            # Per-subset AUC-PR subsampling (long-range)
            range_mask_long = _create_range_mask(L, 24, None, prob_b.device).float()
            comb = mask_b * range_mask_long
            p_s = prob_b[comb > 0].cpu().float().numpy()
            t_s = contact_b[comb > 0].cpu().float().numpy()
            if len(p_s) > 0:
                self._test_subset_auc_probs.setdefault(subset, []).append(p_s)
                self._test_subset_auc_targets.setdefault(subset, []).append(t_s)

            # Per-sample metrics (lightweight dict, no tensor storage)
            pid = pids[b_idx]
            seq_len = int(mask_b.any(dim=0).sum().item())
            prec_s = s_tp / max(1, s_tp + s_fp)
            rec_s = s_tp / max(1, s_tp + s_fn)
            f1_s = 2 * prec_s * rec_s / max(1e-8, prec_s + rec_s)
            pL_all = precision_at_k_masked(prob_3d, contact_3d, mask_3d, k_mode="L")
            self._test_per_sample.append({
                "sample_id": pid,
                "subset": subset,
                "seq_len": seq_len,
                "P@L": round(pL_all, 4),
                "P@L_long": round(pl_long, 4),
                "precision": round(prec_s, 4),
                "recall": round(rec_s, 4),
                "f1": round(f1_s, 4),
                "n_contacts_true": int(vt.sum().item()),
                "n_contacts_pred": int(vp.sum().item()),
            })

        # Save visualizations: all CASP16 + limited non-CASP16
        if self.hparams.get("save_test_viz", False):
            self._save_test_batch_visualizations(viz_cache)

        return loss

    def _save_test_batch_visualizations(
        self,
        viz_cache: Dict,
        max_non_casp_samples: int = 10,
    ):
        """Save contact map visualizations for CASP16 samples (always) and
        a limited number of non-CASP16 samples."""
        from pathlib import Path
        from hydra.core.hydra_config import HydraConfig
        from src.data.components.dataset import SUBSET_CASP16

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

        subsets = viz_cache.get("subset", ["all"] * len(viz_cache["pid"]))

        # Track how many non-CASP samples we've saved (persists across batches)
        if not hasattr(self, "_n_non_casp_viz_saved"):
            self._n_non_casp_viz_saved = 0

        # Process each sample in the batch
        for idx in range(len(viz_cache["pid"])):
            subset = subsets[idx]
            is_casp = (subset == SUBSET_CASP16)

            # Skip non-CASP16 samples after we've saved enough
            if not is_casp and self._n_non_casp_viz_saved >= max_non_casp_samples:
                continue

            prob = viz_cache["prob"][idx]
            contact = viz_cache["contact"][idx]
            valid_mask = viz_cache["valid_mask"][idx]
            seq = viz_cache["seq"][idx]
            crop_bounds = viz_cache["crop_bounds"][idx]
            pid = viz_cache["pid"][idx]

            # Get cropped sequence and tensors
            seq_crop = seq[crop_bounds[0] : crop_bounds[1]]
            L = len(seq_crop)
            prob_crop = prob[0, :L, :L]
            contact_crop = contact[:L, :L]
            valid_crop = valid_mask[:L, :L]

            # Use subfolder for CASP16
            if is_casp:
                sample_dir = save_dir / "casp16"
            else:
                sample_dir = save_dir
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save contact map visualization
            save_path = sample_dir / f"test_{pid}.png"
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
            prc_save_path = sample_dir / f"test_{pid}_PRC.png"
            prc_fig = plot_precision_recall_curve(
                pred_prob=prob_crop,
                target=contact_crop,
                mask=valid_crop,
                pid=pid,
            )
            prc_fig.savefig(prc_save_path, dpi=100, bbox_inches="tight")
            plt.close(prc_fig)

            if not is_casp:
                self._n_non_casp_viz_saved += 1

    def predict_binary_contacts(
        self,
        seq: str,
        pid: str = "query",
        threshold: Optional[float] = None,
        return_probs: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict binary contact map for a single sequence.

        Args:
            seq: Protein sequence (one-letter codes)
            pid: Protein ID (for caching)
            threshold: Probability threshold for binary prediction.
                      If None, uses self.pred_threshold (default 0.5)
            return_probs: If True, also return probability matrix

        Returns:
            dict with keys:
                - "binary": (L, L) binary contact map (0/1)
                - "probs": (L, L) probabilities (only if return_probs=True)
                - "mask": (L, L) valid region mask (long-range pairs)
        """
        from src.data.components.dataset import PriorBuilder

        if threshold is None:
            threshold = self.pred_threshold

        L = len(seq)
        crop_bounds = (0, L)

        # Get embedding and contacts
        h, esm_contacts = self._get_embedding(
            [pid], [seq], [crop_bounds]
        )  # (1, L, D), (1, 1, L, L)

        # Build prior via PriorBuilder (CPU, single sample)
        # Cache the builder to avoid reloading FAISS index + embeddings per call
        if not hasattr(self, '_prior_builder') or self._prior_builder is None:
            self._prior_builder = PriorBuilder(
                index_dir=self.hparams.index_dir,
                topk=self.hparams.get("topk", 4),
                use_blosum=self.hparams.get("use_blosum", True),
                only_positive_transfer=self.hparams.get("only_positive_transfer", False),
                min_seq_sep=self.min_seq_sep,
                min_template_similarity=self.hparams.get("min_template_similarity", 0.0),
                random_retrieval=self.hparams.get("random_retrieval", False),
                n_dist_bins=self.hparams.get("n_dist_bins", 0),
                use_ss_feat=self.hparams.get("n_ss_feat", 0) > 0,
            )
        pb = self._prior_builder
        p_np, c_np, d_np, s_np = pb.build_one(pid, seq, 0, L)
        prior = torch.from_numpy(p_np).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,L,L)
        count = torch.from_numpy(c_np).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,L,L)
        dist_bins = None
        if d_np.shape[0] > 0:
            dist_bins = torch.from_numpy(d_np).float().unsqueeze(0).to(self.device)  # (1,N,L,L)
        ss_feat = None
        if s_np.shape[0] > 0:
            ss_feat = torch.from_numpy(s_np).float().unsqueeze(0).to(self.device)  # (1,S,L,L)

        # Build masks
        rel = relpos_buckets(L, self.device)  # (R, L, L)
        residue_mask = torch.ones(L, device=self.device)
        pair_mask = torch.ones(L, L, device=self.device)

        # Build long-range mask (|i-j| >= min_seq_sep)
        ii, jj = torch.meshgrid(
            torch.arange(L, device=self.device),
            torch.arange(L, device=self.device),
            indexing="ij",
        )
        sep_ok = (torch.abs(ii - jj) >= self.min_seq_sep).float()
        long_mask = pair_mask * sep_ok

        valid = (pair_mask * long_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        rel = rel.unsqueeze(0) * valid  # (1, R, L, L)
        # Note: prior and count are NOT masked here — match training behaviour
        # where prior/count arrive from DataLoader unmasked.

        # Forward pass
        with torch.no_grad():
            logits = self.net(h, prior, count, rel, esm_contacts,
                              pair_mask=pair_mask.unsqueeze(0).unsqueeze(0),
                              dist_bins=dist_bins, ss_feat=ss_feat)  # (1, n_out, L, L)
            if self.distogram:
                dist_probs = torch.softmax(logits, dim=1)  # (1, N, L, L)
                probs = dist_probs[0, :self.contact_bin_threshold].sum(dim=0)  # (L, L)
            else:
                probs = torch.sigmoid(logits[0, 0])  # (L, L)
            binary = (probs >= threshold).float()  # (L, L)

        result = {
            "binary": binary.cpu().to(torch.float32).numpy().astype(np.uint8),
            "mask": (pair_mask * long_mask).cpu().to(torch.float32).numpy().astype(np.uint8),
        }

        if return_probs:
            result["probs"] = probs.cpu().to(torch.float32).numpy().astype(np.float32)

        return result
