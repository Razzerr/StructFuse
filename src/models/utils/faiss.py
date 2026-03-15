import os
import json
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
import faiss

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _get_protein_id(chain_id: str) -> str:
    """Extract protein-level ID from a chain-level ID.

    Handles both PDB chains (``1Z5R_A`` → ``1z5r``) and AlphaFold
    entries (``AF_AFA0A075B5L4F1_1`` → ``af_afa0a075b5l4f1``).

    Using ``rsplit("_", 1)[0]`` instead of ``split("_")[0]`` is
    critical: the naïve split gives ``af`` for *every* AlphaFold
    entry, collapsing thousands of distinct proteins into one ID.
    """
    return chain_id.rsplit("_", 1)[0].lower()


class FaissIndex:
    """
    Minimal runtime helper for FAISS retrieval.
    Built by scripts/build_index.py:
      - <index_dir>/faiss.index
      - <index_dir>/ids.json   (includes cluster_id per entry)

    Cluster-based filtering ensures that templates from the same
    30 % sequence-identity cluster as the query are never returned.
    This mirrors the test-set construction (cluster promotion) and
    prevents the model from learning to copy near-identical templates.
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "ids.json")) as f:
            self.meta = json.load(f)   # list of dicts: id, seq_len, npz, cluster_id
        # quick map from row -> npz path
        self.id2npz = [m["npz"] for m in self.meta]
        self.row2id = [m["id"] for m in self.meta]
        self.d = self.index.d

        # Cluster-based filtering
        self.row2cluster: List[int] = [m.get("cluster_id", -1) for m in self.meta]
        # protein_id -> cluster_id  (for looking up query cluster at runtime)
        self.prot2cluster: Dict[str, int] = {}
        for m in self.meta:
            pid = _get_protein_id(m["id"])
            cid = m.get("cluster_id", -1)
            if cid != -1:
                self.prot2cluster[pid] = cid

        # Precomputed embeddings: chain_id → row index for O(1) lookup
        # Eliminates per-query ESM2 forward passes during training
        emb_path = os.path.join(index_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self._embeddings = np.load(emb_path).astype(np.float32)  # (N, D)
            self._id2row: Dict[str, int] = {
                cid: i for i, cid in enumerate(self.row2id)
            }
            log.info(f"[FaissIndex] Loaded {len(self._id2row)} precomputed embeddings from {emb_path}")
        else:
            self._embeddings = None
            self._id2row = {}
            log.warning(f"[FaissIndex] No embeddings.npy found at {emb_path}, will use ESM2 forward")
        self._emb_hit = 0
        self._emb_miss = 0

    @staticmethod
    def _mean_pool_esm(model, alphabet, seqs: List[Tuple[str, str]], device: str) -> np.ndarray:
        batch_converter = alphabet.get_batch_converter()
        labels, strs, tokens = batch_converter(seqs)
        if device.startswith("cuda") and torch.cuda.is_available():
            tokens = tokens.cuda(non_blocking=True)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers], need_head_weights=False)
            rep = out["representations"][model.num_layers][:, 1:-1, :]  # strip BOS/EOS
            mask = (tokens != alphabet.padding_idx).float()[:, 1:-1]
            masked = rep * mask.unsqueeze(-1)
            sums = masked.sum(dim=1)
            lens = mask.sum(dim=1).clamp_min(1.0)
            x = (sums / lens.unsqueeze(-1)).float()  # (B, D)
        x = x.cpu().to(torch.float32).numpy().astype(np.float32)
        # L2-normalize for cosine/IP search
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x

    def topk(
        self,
        esm_model,
        esm_alphabet,
        query_name: str,
        seq: str,
        k: int,
        device: str,
        debug: bool = False,
        min_similarity: float = 0.0,
        random_retrieval: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k template candidates for a query sequence.

        Filtering order (all applied in a single pass):
          1. Same protein (same protein-level ID) → always blocked
          2. Same 30 % seq-id cluster → blocked (prevents homolog leakage)
          3. min_similarity floor → blocked

        Args:
            esm_model: ESM2 model for embedding
            esm_alphabet: ESM2 alphabet
            query_name: Query protein ID (e.g., "1Z5R_A")
            seq: Query amino acid sequence
            k: Number of templates to retrieve
            device: Torch device
            debug: Enable debug logging
            min_similarity: Minimum similarity threshold (0.0 = no filter)
            random_retrieval: If True, return random templates instead of
                             FAISS neighbors (ablation mode)

        Returns:
            List of (template_id, similarity) tuples
        """
        query_prot_id = _get_protein_id(query_name)
        query_cluster = self.prot2cluster.get(query_prot_id, -1)

        if random_retrieval:
            return self._random_topk(query_prot_id, query_cluster, k)

        # Use precomputed embedding if available (skips ESM2 forward entirely)
        if self._embeddings is not None and query_name in self._id2row:
            row = self._id2row[query_name]
            x = self._embeddings[row : row + 1]  # (1, D)
            self._emb_hit += 1
        else:
            x = self._mean_pool_esm(
                esm_model, esm_alphabet, [(query_name, seq)], device=device
            )  # (1, D)
            self._emb_miss += 1
            if self._emb_miss <= 5:
                log.info(f"[FaissIndex] MISS for '{query_name}' (not in precomputed, total misses={self._emb_miss})")
        
        if (self._emb_hit + self._emb_miss) == 100:
            log.info(f"[FaissIndex] After 100 queries: {self._emb_hit} hits, {self._emb_miss} misses")

        # Large buffer: cluster filtering can remove hundreds of neighbors
        search_k = max(k * 3, k + 500)
        sims, idxs = self.index.search(x.astype(np.float32), search_k)
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Tuple[str, float]] = []
        filtered_same_prot = []
        filtered_cluster = []
        filtered_low_sim = []

        for sim, row in zip(sims, idxs):
            if row < 0:
                continue
            tpl_id = self.row2id[row]
            tpl_prot_id = _get_protein_id(tpl_id)

            # 1. Skip same protein (same PDB + chain group, or same AF protein)
            if tpl_prot_id == query_prot_id:
                filtered_same_prot.append((tpl_id, float(sim)))
                continue

            # 2. Skip same 30 % seq-id cluster (prevents homolog leakage)
            if query_cluster != -1:
                tpl_cluster = self.row2cluster[row]
                if tpl_cluster == query_cluster:
                    filtered_cluster.append((tpl_id, float(sim)))
                    continue

            # 3. min_similarity floor
            if sim < min_similarity:
                filtered_low_sim.append((tpl_id, float(sim)))
                continue

            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break

        if debug:
            log.info(f"\n[RETRIEVAL] Query: {query_name} (prot={query_prot_id}, cluster={query_cluster})")
            log.info(f"  Filtered {len(filtered_same_prot)} same-protein templates")
            for tid, s in filtered_same_prot[:3]:
                log.info(f"    SAME_PROT: {tid} (sim={s:.4f})")
            log.info(f"  Filtered {len(filtered_cluster)} same-cluster templates")
            for tid, s in filtered_cluster[:3]:
                log.info(f"    CLUSTER:   {tid} (sim={s:.4f})")
            if filtered_low_sim:
                log.info(f"  Filtered {len(filtered_low_sim)} below min_similarity={min_similarity}")
            log.info(f"  Retrieved {len(out)} templates:")
            for tid, s in out:
                log.info(f"    OK: {tid} (sim={s:.4f})")

        return out

    def topk_precomputed(
        self,
        query_name: str,
        k: int,
        min_similarity: float = 0.0,
        random_retrieval: bool = False,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k templates using only precomputed embeddings (no ESM model).

        Designed for use in DataLoader workers where no GPU/ESM model is
        available.  Falls back to an empty list if the query is not found in
        the precomputed embedding table.

        Returns:
            List of (template_id, similarity) tuples (may be shorter than *k*).
        """
        query_prot_id = _get_protein_id(query_name)
        query_cluster = self.prot2cluster.get(query_prot_id, -1)

        if random_retrieval:
            return self._random_topk(query_prot_id, query_cluster, k)

        if self._embeddings is None or query_name not in self._id2row:
            return []  # graceful fallback: no prior for this query

        row = self._id2row[query_name]
        x = self._embeddings[row : row + 1]  # (1, D)

        search_k = max(k * 3, k + 500)
        sims, idxs = self.index.search(x.astype(np.float32), search_k)
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Tuple[str, float]] = []
        for sim, row_idx in zip(sims, idxs):
            if row_idx < 0:
                continue
            tpl_id = self.row2id[row_idx]
            tpl_prot_id = _get_protein_id(tpl_id)
            if tpl_prot_id == query_prot_id:
                continue
            if query_cluster != -1:
                tpl_cluster = self.row2cluster[row_idx]
                if tpl_cluster == query_cluster:
                    continue
            if sim < min_similarity:
                continue
            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break
        return out

    def _random_topk(
        self, query_prot_id: str, query_cluster: int, k: int
    ) -> List[Tuple[str, float]]:
        """Return *k* random templates (excluding same protein & cluster).

        Used for ablation to test whether retrieval quality matters.
        Returns templates with similarity = 0.5 (neutral weight).
        """
        valid_indices = []
        for i, tpl_id in enumerate(self.row2id):
            if _get_protein_id(tpl_id) == query_prot_id:
                continue
            if query_cluster != -1 and self.row2cluster[i] == query_cluster:
                continue
            valid_indices.append(i)

        if len(valid_indices) < k:
            k = len(valid_indices)
        if k == 0:
            return []

        selected = random.sample(valid_indices, k)
        return [(self.row2id[i], 0.5) for i in selected]