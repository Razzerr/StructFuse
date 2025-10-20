import os
import json
from typing import List, Tuple

import numpy as np
import torch
import faiss


class FaissIndex:
    """
    Minimal runtime helper for FAISS retrieval.
    Built by scripts/build_index.py:
      - <index_dir>/faiss.index
      - <index_dir>/ids.json
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "ids.json")) as f:
            self.meta = json.load(f)   # list of dicts: id, seq_len, npz
        # quick map from row -> npz path
        self.id2npz = [m["npz"] for m in self.meta]
        self.row2id = [m["id"] for m in self.meta]
        self.d = self.index.d

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
        x = x.cpu().numpy().astype(np.float32)
        # L2-normalize for cosine/IP search
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x

    def topk(self, esm_model, esm_alphabet, query_name: str, seq: str, k: int, device: str, debug: bool = False) -> List[Tuple[str, float]]:
        x = self._mean_pool_esm(esm_model, esm_alphabet, [(query_name, seq)], device=device)  # (1, D)
        # FAISS expects float32 contiguous
        # Request extra candidates to account for same-PDB filtering
        # Dataset has up to 24 chains per PDB, but typically 1-4
        sims, idxs = self.index.search(x.astype(np.float32), k + 24)
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()
        
        # Extract PDB ID from query (e.g., "1Z5R_A" -> "1Z5R")
        query_pdb_id = query_name.split('_')[0]
        
        out = []
        filtered_same_pdb = []  # Track what we filtered for debugging
        
        for sim, row in zip(sims, idxs):
            if row < 0:
                continue
            tpl_id = self.row2id[row]
            tpl_pdb_id = tpl_id.split('_')[0]
            
            # CRITICAL: Skip same PDB ID (blocks self + other chains from same structure)
            if tpl_pdb_id == query_pdb_id:
                filtered_same_pdb.append((tpl_id, float(sim)))
                continue
                
            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break
        
        # Debug logging to detect leakage
        if debug:
            print(f"\n[LEAKAGE CHECK] Query: {query_name} (PDB: {query_pdb_id})")
            print(f"  Filtered {len(filtered_same_pdb)} same-PDB templates:")
            for tpl_id, sim in filtered_same_pdb[:5]:
                print(f"    BLOCKED: {tpl_id} (sim={sim:.4f})")
            print(f"  Retrieved {len(out)} templates from different PDBs:")
            for tpl_id, sim in out:
                tpl_pdb = tpl_id.split('_')[0]
                # ASSERTION: Should never match query PDB
                assert tpl_pdb != query_pdb_id, f"LEAKAGE DETECTED: {tpl_id} shares PDB {query_pdb_id}!"
                print(f"    OK: {tpl_id} (PDB={tpl_pdb}, sim={sim:.4f})")
        
        return out