from typing import List, Tuple, Dict, Optional

from collections import OrderedDict
import numpy as np
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from src.models.components.esm_backbone import ESM2Backbone
from src.models.components.contact_model import ContactModel
from src.models.components.pair2d_head import relpos_buckets
from src.models.utils.faiss import FaissIndex
from src.models.utils.loss import masked_bce_balanced, masked_focal_tversky
from src.models.utils.metrics import precision_at_k_masked
from src.models.utils.visualize import (
    plot_contact_map_comparison,
    plot_precision_recall_curve,
)
from src.data.utils.align import project_prior

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
        d_esm: int = 1280,
        d_pair: int = 128,
        width: int = 128,
        depth: int = 8,
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.esm = ESM2Backbone(esm_model, finetune=finetune_esm)
        self.esm_alphabet = self.esm.alphabet

        self.net = ContactModel(
            d_esm=d_esm,
            d_pair=d_pair,
            width=width,
            depth=depth,
            fusion_strategy=fusion_strategy,
            fusion_num_heads=fusion_num_heads,
            fusion_reduction=fusion_reduction,
            head_type=head_type,
            head_num_heads=head_num_heads,
        )

        self.index = FaissIndex(index_dir)
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

        self._id_to_npz = {m["id"]: m["npz"] for m in self.index.meta}
        self._faiss_cache: Dict[Tuple[str, int, int], List[Tuple[str, float]]] = (
            {}
        )  # (pid, topk, min_sep)
        self._tpl_npz_cache: Dict[str, Dict[str, np.ndarray]] = (
            {}
        )  # tpl_id -> {"seq":..., "contact":...}
        self._prior_cache: Dict[
            Tuple[str, str, int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}  # (qpid,tpl_id,Lq,min_sep) -> (Pk_pos,cnt)
        self._embedding_cache = OrderedDict()

        self._max_tpl_cache = max_tpl_cache

        # For visualization logging
        self._val_viz_logged = False  # Log one viz per validation epoch
        self._viz_protein_id = None  # Track same protein across epochs

        # For threshold optimization
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []

        self._test_probs = []
        self._test_targets = []
        self._test_masks = []

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        return opt
    
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
        
        # Load the rest of the state
        return super().load_state_dict(state_dict, strict=strict)

    def _get_hits(self, pid: str, seq: str, debug: bool = False):
        key = (pid, self.topk, self.min_seq_sep)
        if key in self._faiss_cache:
            return self._faiss_cache[key]
        hits = self.index.topk(
            self.esm.model,
            self.esm_alphabet,
            pid,
            seq,
            self.topk,
            device=str(self.device),
            debug=debug,  # Enable leakage checking
        )
        self._faiss_cache[key] = hits
        return hits

    def _get_tpl(self, tpl_id: str):
        if tpl_id in self._tpl_npz_cache:
            return self._tpl_npz_cache[tpl_id]

        # Evict oldest if cache too large
        if len(self._tpl_npz_cache) >= self._max_tpl_cache:
            # Simple FIFO eviction
            oldest_key = next(iter(self._tpl_npz_cache))
            del self._tpl_npz_cache[oldest_key]

        npz_path = self._id_to_npz.get(tpl_id)
        td = np.load(npz_path, allow_pickle=True)
        tseq_arr = td["seq"]
        tseq = (
            str(tseq_arr.item())
            if isinstance(tseq_arr, np.ndarray) and tseq_arr.shape == ()
            else str(tseq_arr)
        )
        tC = td["contact"].astype(np.uint8).copy()
        td.close()
        self._tpl_npz_cache[tpl_id] = {"seq": tseq, "contact": tC}
        return self._tpl_npz_cache[tpl_id]

    def _get_embedding(
        self, pid: str, seq: str, bounds: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ESM embedding and contacts with LRU cache

        Returns:
            Tuple of (embedding, contacts) where:
                embedding: (L_crop, D)
                contacts: (1, L_crop, L_crop)
        """
        cache_key = (pid, seq)

        if cache_key in self._embedding_cache:
            emb, cont = self._embedding_cache[cache_key]
            emb = emb.to(self.device)
            cont = cont.to(self.device)
            return (
                emb[bounds[0] : bounds[1]],
                cont[:, bounds[0] : bounds[1], bounds[0] : bounds[1]],
            )

        # Cache miss - compute
        h, esm_cont = self.esm(
            [(pid, seq)], self.device
        )  # (1, L_full, D), (1, 1, L_full, L_full)

        # Cache on CPU (store full-length results)
        self._embedding_cache[cache_key] = (h[0].cpu(), esm_cont[0].cpu())

        return (
            h[0, bounds[0] : bounds[1]],
            esm_cont[0, :, bounds[0] : bounds[1], bounds[0] : bounds[1]],
        )

    def _build_priors(
        self,
        pids: List[str],
        seqs: List[str],
        crop_bounds: List[Tuple[int, int]],
        Lmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(seqs)
        device = self.device
        prior = torch.zeros((B, 1, Lmax, Lmax), dtype=torch.float32, device=device)
        count = torch.zeros((B, 1, Lmax, Lmax), dtype=torch.float32, device=device)

        total_templates = 0
        cache_hits = 0
        cache_misses = 0

        for b, (pid, qseq, crop_bound) in enumerate(zip(pids, seqs, crop_bounds)):
            hits = self._get_hits(pid, qseq)
            if len(hits) == 0:
                continue

            sims = np.array([h[1] for h in hits], dtype=np.float32)
            w = np.exp(sims - sims.max())
            w = w / (w.sum() + 1e-8)

            Lq = len(qseq)
            for (tpl_id, sim), wk in zip(hits, w):
                total_templates += 1
                # Use the full sequence string as key (before cropping)
                cache_key = (pid, tpl_id, self.min_seq_sep)

                if cache_key in self._prior_cache:
                    cache_hits += 1
                    Pk_pos_cached, cnt_cached = self._prior_cache[cache_key]

                    Pk_crop = Pk_pos_cached[
                        crop_bound[0] : crop_bound[1], crop_bound[0] : crop_bound[1]
                    ]
                    cnt_crop = cnt_cached[
                        crop_bound[0] : crop_bound[1], crop_bound[0] : crop_bound[1]
                    ]

                    # Cached prior is for full query sequence, crop to current Lq
                    L_use = min(Pk_crop.shape[0], Lmax)
                    prior[b, 0, :L_use, :L_use] += wk * Pk_crop[
                        :L_use, :L_use
                    ].float().to(device)
                    count[b, 0, :L_use, :L_use] += (
                        cnt_crop[:L_use, :L_use].float().to(device)
                    )
                else:
                    cache_misses += 1
                    # Compute prior
                    tpl = self._get_tpl(tpl_id)
                    tseq, tC = tpl["seq"], tpl["contact"]
                    Pk = project_prior(
                        qseq,
                        tseq,
                        tC,
                        min_seq_sep=0,
                        symmetrize=True,
                        use_blosum=self.use_blosum,
                    )

                    if self.only_positive_transfer:
                        known = Pk == 1
                        Pk_pos_np = known.astype(np.float32)
                        cnt_np = known.astype(np.float32)
                    else:
                        known = Pk != -1
                        Pk_pos_np = np.clip(Pk, 0, 1).astype(np.float32)
                        if self.use_blosum:
                            # With BLOSUM, prior is only non-zero where there are contacts
                            cnt_np = (Pk_pos_np != 0).astype(np.float32)
                        else:
                            cnt_np = known.astype(np.float32)

                    if self.min_seq_sep > 0:
                        ii, jj = np.indices((Lq, Lq))
                        close = np.abs(ii - jj) < self.min_seq_sep
                        Pk_pos_np[close] = 0.0
                        cnt_np[close] = 0.0

                    Pk_pos = torch.from_numpy(Pk_pos_np).float()
                    cnt = torch.from_numpy(cnt_np).float()

                    # Cache on CPU (store full-length result)
                    self._prior_cache[cache_key] = (
                        Pk_pos.cpu().half(),
                        cnt.cpu().half(),
                    )

                    Pk_crop = Pk_pos[
                        crop_bound[0] : crop_bound[1], crop_bound[0] : crop_bound[1]
                    ]
                    cnt_crop = cnt[
                        crop_bound[0] : crop_bound[1], crop_bound[0] : crop_bound[1]
                    ]

                    L_use = min(Pk_crop.shape[0], Lmax)
                    prior[b, 0, :L_use, :L_use] += wk * Pk_crop[:L_use, :L_use].to(
                        device
                    )
                    count[b, 0, :L_use, :L_use] += cnt_crop[:L_use, :L_use].to(device)

        return prior, count

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
        """
        pids = batch["pid"]
        seqs = batch["seq"]
        crop_bounds = batch["crop_bounds"]  # (B, 2)
        contact = batch["contact"].to(self.device)  # (B, Lmax, Lmax)
        long_mask = batch["long_mask"].to(self.device)
        pair_mask = batch["pair_mask"].to(self.device)

        h_list = []
        esm_contacts_list = []
        for pid, seq, bounds in zip(pids, seqs, crop_bounds):
            h_seq, esm_cont_seq = self._get_embedding(pid, seq, bounds)
            h_list.append(h_seq)
            esm_contacts_list.append(esm_cont_seq)
        h = torch.nn.utils.rnn.pad_sequence(h_list, batch_first=True)  # (B, L_max, D)

        # Pad ESM contacts to match batch dimensions
        # Contacts are already cropped inside _get_embedding, just need to pad to batch Lmax
        Lmax = h.shape[1]
        esm_contacts = torch.zeros(
            (len(esm_contacts_list), 1, Lmax, Lmax), device=self.device
        )
        for i, cont in enumerate(esm_contacts_list):
            L = cont.shape[1]  # cont is (1, L, L)
            esm_contacts[i, :, :L, :L] = cont

        prior, count = self._build_priors(
            pids, seqs, crop_bounds, Lmax
        )  # (B, 1, Lmax, Lmax)

        rel = relpos_buckets(Lmax, self.device)  # (R,L,L)
        # broadcast to batch and mask (no big expands needed)
        valid = (pair_mask * long_mask).unsqueeze(1)  # (B,1,L,L)
        rel = rel.unsqueeze(0) * valid  # (B,R,L,L) via broadcast

        logits = self.net(h, prior, count, rel, esm_contacts)  # (B, 1, Lmax, Lmax)
        valid_mask = long_mask * pair_mask

        # Primary loss: BCE
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

        if stage == "val" or self.trainer.global_step % 10 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pL = precision_at_k_masked(prob, contact, valid_mask, k_mode="L")
                pL2 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/2")
                pL5 = precision_at_k_masked(prob, contact, valid_mask, k_mode="L/5")

            prog_bar = stage == "val"
            self.log(
                f"{stage}/loss",
                loss,
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L",
                pL,
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L2",
                pL2,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}/P@L5",
                pL5,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if self.use_tversky:
                self.log(
                    f"{stage}/loss_tversky",
                    loss_tversky,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/loss_bce",
                    loss_bce,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        viz_cache = None
        if return_visualization:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
            
            viz_cache = {
                "prob": prob,
                "contact": contact,
                "valid_mask": valid_mask,
                "seq": seqs,
                "crop_bounds": crop_bounds,
                "pid": pids,
            }

        return loss, viz_cache

    def on_train_start(self):
        """Pre-populate caches to speed up training"""
        # Get a few samples from train dataloader
        train_loader = self.trainer.train_dataloader
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Warm up with 10 batches
                break

            pids = batch["pid"]
            seqs = batch["seq"]

            # Populate FAISS cache
            for pid, seq in zip(pids, seqs):
                self._get_hits(pid, seq)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._step(batch, stage="train")[0]

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

        # Try thresholds from 0.05 to 0.95
        thresholds = torch.linspace(0.05, 0.95, 19)
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

        # Accumulate predictions for threshold optimization
        self._val_probs.append(viz_cache["prob"].detach().cpu())
        self._val_targets.append(viz_cache["contact"].detach().cpu())
        self._val_masks.append(viz_cache["valid_mask"].detach().cpu())

        return loss

    def on_validation_epoch_start(self):
        """Reset visualization flag and accumulation at start of each validation epoch"""
        self._val_viz_logged = False

    def on_validation_epoch_end(self):
        """Find optimal threshold on accumulated validation predictions"""
        if len(self._val_probs) == 0:
            raise ValueError(
                "No validation predictions accumulated for threshold optimization."
            )
        
        # Skip threshold optimization during sanity check
        if self.trainer.sanity_checking:
            self._val_probs = []
            self._val_targets = []
            self._val_masks = []
            return

        # Find max length across all batches
        max_L = max(batch.shape[-1] for batch in self._val_probs)
        
        # Pad all batches to max_L
        padded_probs = []
        padded_targets = []
        padded_masks = []
        
        for prob_batch, target_batch, mask_batch in zip(self._val_probs, self._val_targets, self._val_masks):
            # prob_batch is (B, 1, L, L) or (B, L, L)
            if prob_batch.dim() == 4:
                prob_batch = prob_batch.squeeze(1)  # (B, L, L)
            
            B, L, _ = prob_batch.shape
            if L < max_L:
                # Pad to max_L
                pad_size = max_L - L
                prob_batch = torch.nn.functional.pad(prob_batch, (0, pad_size, 0, pad_size), value=0)
                target_batch = torch.nn.functional.pad(target_batch, (0, pad_size, 0, pad_size), value=0)
                mask_batch = torch.nn.functional.pad(mask_batch, (0, pad_size, 0, pad_size), value=0)
            
            padded_probs.append(prob_batch)
            padded_targets.append(target_batch)
            padded_masks.append(mask_batch)
        
        # Now concatenate - all have same shape (B, max_L, max_L)
        all_probs = torch.cat(padded_probs, dim=0)  # (N, max_L, max_L)
        all_targets = torch.cat(padded_targets, dim=0)  # (N, max_L, max_L)
        all_masks = torch.cat(padded_masks, dim=0)  # (N, max_L, max_L)

        # Find optimal threshold
        optimal_threshold, optimal_f1 = self._find_optimal_threshold(
            all_probs, all_targets, all_masks
        )

        # Update model's threshold with the optimal value
        self.pred_threshold = optimal_threshold

        # Compute precision and recall at optimal threshold
        all_preds = (all_probs >= optimal_threshold).float()
        valid_preds = all_preds[all_masks > 0]
        valid_targets = all_targets[all_masks > 0]

        tp = ((valid_preds == 1) & (valid_targets == 1)).sum().float()
        fp = ((valid_preds == 1) & (valid_targets == 0)).sum().float()
        fn = ((valid_preds == 0) & (valid_targets == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Log the optimal threshold and metrics
        self.log(
            "val/optimal_threshold", optimal_threshold, prog_bar=True, sync_dist=False
        )
        self.log("val/precision", precision, prog_bar=True, sync_dist=False)
        self.log("val/recall", recall, prog_bar=True, sync_dist=False)
        self.log("val/f1", optimal_f1, prog_bar=True, sync_dist=False)

        # Clear accumulated data
        self._val_probs = []
        self._val_targets = []
        self._val_masks = []

    def on_test_epoch_end(self):
        """Compute precision, recall, F1 at optimal threshold on test set"""
        if len(self._test_probs) == 0:
            print("No test predictions accumulated for metrics computation.")
            return

        # Find max length across all batches
        max_L = max(batch.shape[-1] for batch in self._test_probs)
        
        # Pad all batches to max_L
        padded_probs = []
        padded_targets = []
        padded_masks = []
        
        for prob_batch, target_batch, mask_batch in zip(self._test_probs, self._test_targets, self._test_masks):
            # prob_batch is (B, 1, L, L) or (B, L, L)
            if prob_batch.dim() == 4:
                prob_batch = prob_batch.squeeze(1)  # (B, L, L)
            
            B, L, _ = prob_batch.shape
            if L < max_L:
                # Pad to max_L
                pad_size = max_L - L
                prob_batch = torch.nn.functional.pad(prob_batch, (0, pad_size, 0, pad_size), value=0)
                target_batch = torch.nn.functional.pad(target_batch, (0, pad_size, 0, pad_size), value=0)
                mask_batch = torch.nn.functional.pad(mask_batch, (0, pad_size, 0, pad_size), value=0)
            
            padded_probs.append(prob_batch)
            padded_targets.append(target_batch)
            padded_masks.append(mask_batch)
        
        # Now concatenate - all have same shape (B, max_L, max_L)
        all_probs = torch.cat(padded_probs, dim=0)  # (N, max_L, max_L)
        all_targets = torch.cat(padded_targets, dim=0)  # (N, max_L, max_L)
        all_masks = torch.cat(padded_masks, dim=0)  # (N, max_L, max_L)

        # Apply threshold to get binary predictions
        all_preds = (all_probs >= self.pred_threshold).float()

        # Compute metrics (masked)
        valid_preds = all_preds[all_masks > 0]
        valid_targets = all_targets[all_masks > 0]

        tp = ((valid_preds == 1) & (valid_targets == 1)).sum().float()
        fp = ((valid_preds == 1) & (valid_targets == 0)).sum().float()
        fn = ((valid_preds == 0) & (valid_targets == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Log metrics
        self.log("test/precision", precision, prog_bar=True, sync_dist=False)
        self.log("test/recall", recall, prog_bar=True, sync_dist=False)
        self.log("test/f1", f1, prog_bar=True, sync_dist=False)
        self.log(
            "test/threshold_used", self.pred_threshold, prog_bar=False, sync_dist=False
        )

        # Clear accumulated data
        self._test_probs = []
        self._test_targets = []
        self._test_masks = []

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

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, viz_cache = self._step(batch, stage="test", return_visualization=True)

        # Accumulate predictions for metrics at optimal threshold
        self._test_probs.append(viz_cache["prob"].detach().cpu())
        self._test_targets.append(viz_cache["contact"].detach().cpu())
        self._test_masks.append(viz_cache["valid_mask"].detach().cpu())

        self._save_test_batch_visualizations(viz_cache)

        return loss

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
            seq_crop = seq[crop_bounds[0] : crop_bounds[1]]
            L = len(seq_crop)
            prob_crop = prob[0, :L, :L]
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
        if threshold is None:
            threshold = self.pred_threshold

        L = len(seq)
        crop_bounds = (0, L)

        # Get embedding and contacts
        h, esm_contacts = self._get_embedding(
            pid, seq, crop_bounds
        )  # (L, D), (1, L, L)
        h = h.unsqueeze(0)  # (1, L, D)
        esm_contacts = esm_contacts.unsqueeze(0)  # (1, 1, L, L)

        # Get priors
        prior, count = self._build_priors([pid], [seq], [crop_bounds], L)

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
        prior *= valid
        count *= valid

        # Forward pass
        with torch.no_grad():
            logits = self.net(h, prior, count, rel, esm_contacts)  # (1, 1, L, L)
            probs = torch.sigmoid(logits[0, 0])  # (L, L)
            binary = (probs >= threshold).float()  # (L, L)

        result = {
            "binary": binary.cpu().numpy().astype(np.uint8),
            "mask": (pair_mask * long_mask).cpu().numpy().astype(np.uint8),
        }

        if return_probs:
            result["probs"] = probs.cpu().numpy().astype(np.float32)

        return result
