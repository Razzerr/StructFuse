import os
import json
import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple
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
    def __init__(
        self,
        index_dir: str,
        holdout_ids: Optional[Iterable[str]] = None,
        random_seed: int = 0,
    ):
        self.index_dir = index_dir
        # Stable seed for the random_retrieval ablation. Per-query RNG is seeded
        # by stable_hash(random_seed, query_id) so each chain always draws the
        # same admissible templates regardless of worker/batch scheduling.
        self.random_seed = int(random_seed)
        # Protein-level holdout set: any retrieved template whose protein-ID
        # matches one of these is dropped when `filter_holdout=True` is passed
        # to topk*. Used at training time to prevent train queries from
        # retrieving val/test structures (while keeping those chains in the
        # index so val/test queries themselves can retrieve meaningful
        # neighbours at inference).
        self.holdout_prot_ids: Set[str] = {
            _get_protein_id(x) for x in (holdout_ids or [])
        }
        if self.holdout_prot_ids:
            log.info(
                f"[FaissIndex] Loaded {len(self.holdout_prot_ids)} holdout protein IDs "
                f"(filter_holdout=True queries will skip them)"
            )
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "ids.json")) as f:
            self.meta = json.load(f)   # list of dicts: id, seq_len, npz, cluster_id
        # quick map from row -> npz path
        self.id2npz = [m["npz"] for m in self.meta]
        self.row2id = [m["id"] for m in self.meta]
        self.d = self.index.d

        # Cluster-based filtering
        self.row2cluster: List[int] = [int(m.get("cluster_id", -1)) for m in self.meta]
        # chain_id -> cluster_id. This is the authoritative lookup for query
        # chains. A single PDB entry can contain multiple polymer entities that
        # belong to different sequence clusters, so a protein-level lookup would
        # assign the wrong cluster to some chains and leak near-identical
        # templates through the same-cluster filter.
        self.chain2cluster: Dict[str, int] = {
            m["id"]: int(m.get("cluster_id", -1)) for m in self.meta
        }
        # protein_id -> set(cluster_id). A single PDB entry can contain
        # multiple polymer entities from different sequence clusters. The
        # filtering path uses the union as a conservative fallback whenever an
        # exact chain/entity cluster is unavailable (-1).
        self.prot2clusters: Dict[str, Set[int]] = {}
        # Legacy/debug fallback for callers that still inspect this attribute.
        self.prot2cluster: Dict[str, int] = {}
        for m in self.meta:
            pid = _get_protein_id(m["id"])
            cid = int(m.get("cluster_id", -1))
            if cid != -1:
                self.prot2clusters.setdefault(pid, set()).add(cid)
                self.prot2cluster.setdefault(pid, cid)
        # cluster_id -> #chains in that cluster. Used to expand search_k when a
        # query sits in a mega-cluster (e.g. ribosomal proteins, ~63k chains):
        # the top-500 FAISS neighbours would all be same-cluster and get
        # filtered, leaving zero hits. We need search_k ≥ cluster_size to see
        # the first out-of-cluster neighbour.
        self.cluster2size: Dict[int, int] = {}
        for cid in self.row2cluster:
            if cid != -1:
                self.cluster2size[cid] = self.cluster2size.get(cid, 0) + 1

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

        # Lazily-built candidate pools for the random_retrieval ablation — built
        # once per process (O(N)), then O(K) rejection sampling per query instead
        # of the old O(N)-per-query full scan.
        self._row2prot = None
        self._row2cluster_arr = None
        self._pool_all = None
        self._pool_nonholdout = None

    def _clusters_for_chain(self, chain_id: str, prot_id: Optional[str] = None) -> Set[int]:
        """Return the sequence-cluster ID(s) to filter a query/template against.

        The chain-level ID from strict `ids.json` metadata is authoritative and
        is used alone whenever it is known (99.6 % of chains). The PDB-level
        union of sibling entities is a fallback used ONLY when the chain-level
        ID is missing (-1) — using it unconditionally would block legitimate
        remote homologs of unrelated chains that merely share a crystal, and
        would inflate the `search_k` over-fetch below.

        An EMPTY return means "cluster genuinely unknown", not "no restriction":
        `_same_cluster` blocks every template for such a query.
        """
        exact = int(self.chain2cluster.get(chain_id, -1))
        if exact != -1:
            return {exact}
        pid = prot_id or _get_protein_id(chain_id)
        return set(self.prot2clusters.get(pid, set()))

    def _same_cluster(
        self,
        query_clusters: Set[int],
        tpl_id: str,
        tpl_cluster: int,
    ) -> bool:
        """Same-cluster predicate shared by all retrieval paths. True ⇒ blocked.

        An unknown cluster on either side is never treated as evidence that the
        pair is admissible — it BLOCKS. Concretely:

        * empty ``query_clusters`` — the query's own chain cluster is -1 and its
          PDB has no clustered sibling either, so no template can be *shown* to
          be out-of-cluster. Nothing is admissible.
        * template cluster -1 — fall back to the template's PDB-level union; if
          that union is empty too (the entry is absent from ``clusters_30.txt``
          entirely), block.

        The empty-union case used to fall through to ``set().intersection(...)``
        → falsy → admitted, which is the exact opposite of the stated policy. It
        was the mechanism behind 3.29 % of best-retrieved templates carrying
        cluster -1 at a median sequence identity of 1.000 (2026-08-05 audit):
        RCSB omits some entries/entities from the 30 % cluster file, and those
        omissions were being read as "different cluster, therefore fine".
        """
        if not query_clusters:
            return True
        tpl_cluster = int(tpl_cluster)
        if tpl_cluster != -1:
            return tpl_cluster in query_clusters
        tpl_clusters = self.prot2clusters.get(_get_protein_id(tpl_id), set())
        if not tpl_clusters:
            return True
        return bool(query_clusters.intersection(tpl_clusters))

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
        filter_holdout: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k template candidates for a query sequence.

        Filtering order (all applied in a single pass):
          1. Same protein (same protein-level ID) → always blocked
          2. Holdout protein IDs (val/test) → blocked when ``filter_holdout``
          3. Same 30 % seq-id cluster → blocked (prevents homolog leakage)
          4. min_similarity floor → blocked

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
        query_clusters = self._clusters_for_chain(query_name, query_prot_id)

        if random_retrieval:
            return self._random_topk(
                query_name, query_prot_id, query_clusters, k, filter_holdout=filter_holdout
            )

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

        # Adaptive over-fetch: the query may sit in a mega-cluster (e.g. cluster=1
        # ≈ 63k ribosomal chains) where the top-500 FAISS neighbours are all
        # same-cluster and would be filtered out.
        extra_holdout = len(self.holdout_prot_ids) if filter_holdout else 0
        extra_cluster = sum(self.cluster2size.get(cid, 0) for cid in query_clusters)
        search_k = max(k * 3, k + 500 + extra_holdout + extra_cluster)
        search_k = min(search_k, self.index.ntotal)
        sims, idxs = self.index.search(x.astype(np.float32), search_k)
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Tuple[str, float]] = []
        filtered_same_prot = []
        filtered_holdout = []
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

            # 2. Skip val/test holdout proteins (train-time leak prevention)
            if filter_holdout and tpl_prot_id in self.holdout_prot_ids:
                filtered_holdout.append((tpl_id, float(sim)))
                continue

            # 3. Skip same 30 % seq-id cluster (prevents homolog leakage)
            tpl_cluster = self.row2cluster[row]
            if self._same_cluster(query_clusters, tpl_id, tpl_cluster):
                filtered_cluster.append((tpl_id, float(sim)))
                continue

            # 4. min_similarity floor
            if sim < min_similarity:
                filtered_low_sim.append((tpl_id, float(sim)))
                continue

            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break

        if debug:
            log.info(
                f"\n[RETRIEVAL] Query: {query_name} "
                f"(prot={query_prot_id}, clusters={sorted(query_clusters)})"
            )
            log.info(f"  Filtered {len(filtered_same_prot)} same-protein templates")
            for tid, s in filtered_same_prot[:3]:
                log.info(f"    SAME_PROT: {tid} (sim={s:.4f})")
            if filtered_holdout:
                log.info(f"  Filtered {len(filtered_holdout)} holdout (val/test) templates")
                for tid, s in filtered_holdout[:3]:
                    log.info(f"    HOLDOUT:   {tid} (sim={s:.4f})")
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
        filter_holdout: bool = True,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k templates using only precomputed embeddings (no ESM model).

        Designed for use in DataLoader workers where no GPU/ESM model is
        available.  Falls back to an empty list if the query is not found in
        the precomputed embedding table.

        ``filter_holdout=True`` (default) drops any candidate whose protein-ID
        is in ``self.holdout_prot_ids``.  Training loaders should pass True;
        val/test loaders should pass False so inference can see the full index.

        Returns:
            List of (template_id, similarity) tuples (may be shorter than *k*).
        """
        query_prot_id = _get_protein_id(query_name)
        query_clusters = self._clusters_for_chain(query_name, query_prot_id)

        if random_retrieval:
            return self._random_topk(
                query_name, query_prot_id, query_clusters, k, filter_holdout=filter_holdout
            )

        if self._embeddings is None or query_name not in self._id2row:
            return []  # graceful fallback: no prior for this query

        row = self._id2row[query_name]
        x = self._embeddings[row : row + 1]  # (1, D)

        # Adaptive over-fetch: the query may sit in a mega-cluster (e.g. cluster=1
        # ≈ 63k ribosomal chains) where the top-500 FAISS neighbours are all
        # same-cluster and would be filtered out. search_k must at least cover
        # the query's cluster so the first out-of-cluster hit is returned.
        extra_holdout = len(self.holdout_prot_ids) if filter_holdout else 0
        extra_cluster = sum(self.cluster2size.get(cid, 0) for cid in query_clusters)
        search_k = max(k * 3, k + 500 + extra_holdout + extra_cluster)
        search_k = min(search_k, self.index.ntotal)
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
            if filter_holdout and tpl_prot_id in self.holdout_prot_ids:
                continue
            tpl_cluster = self.row2cluster[row_idx]
            if self._same_cluster(query_clusters, tpl_id, tpl_cluster):
                continue
            if sim < min_similarity:
                continue
            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break
        return out

    def _ensure_random_pools(self):
        """Build (once per process) the candidate-row pools + lookup arrays used
        by `_random_topk`. O(N) one-off, shared by all subsequent queries."""
        if self._pool_all is not None:
            return
        self._row2prot = np.array([_get_protein_id(x) for x in self.row2id])
        self._row2cluster_arr = np.asarray(self.row2cluster)
        self._pool_all = np.arange(len(self.row2id), dtype=np.int64)
        if self.holdout_prot_ids:
            keep = np.fromiter(
                (p not in self.holdout_prot_ids for p in self._row2prot),
                dtype=bool, count=len(self._row2prot),
            )
            self._pool_nonholdout = self._pool_all[keep]
        else:
            self._pool_nonholdout = self._pool_all

    def _stable_seed(self, query_name: str) -> int:
        """Deterministic per-query seed from (random_seed, query_id) using a
        STABLE hash (blake2b) — NOT builtin hash() (randomized per process)."""
        digest = hashlib.blake2b(
            f"{self.random_seed}:{query_name}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little")

    def _random_topk(
        self,
        query_name: str,
        query_prot_id: str,
        query_clusters: Set[int],
        k: int,
        filter_holdout: bool = True,
    ) -> List[Tuple[str, float]]:
        """Return up to *k* random admissible templates (excluding same protein,
        same cluster, and — when filter_holdout — holdout proteins). Similarity is
        a neutral 0.5. Stateless + reproducible: seeded by stable_hash(random_seed,
        query_name), so a chain always draws the same templates regardless of
        worker scheduling. O(K) expected via rejection sampling; a deterministic
        cyclic scan is the worst-case fallback. If fewer than k admissible
        candidates exist, returns ALL admissible + logs a warning (never raises).
        """
        self._ensure_random_pools()
        pool = self._pool_nonholdout if filter_holdout else self._pool_all
        if len(pool) == 0 or k <= 0:
            return []

        rng = np.random.default_rng(self._stable_seed(query_name))

        def _admissible(row: int) -> bool:
            if self._row2prot[row] == query_prot_id:
                return False
            tpl_id = self.row2id[row]
            tpl_cluster = int(self._row2cluster_arr[row])
            if self._same_cluster(query_clusters, tpl_id, tpl_cluster):
                return False
            return True

        selected: List[int] = []
        seen: Set[int] = set()
        n_pool = len(pool)
        max_attempts = max(64, k * 32)
        attempts = 0
        chunk = max(k * 4, 16)
        while len(selected) < k and attempts < max_attempts:
            draws = rng.integers(0, n_pool, size=chunk)
            for d in draws:
                attempts += 1
                row = int(pool[int(d)])
                if row in seen or not _admissible(row):
                    continue
                seen.add(row)
                selected.append(row)
                if len(selected) >= k:
                    break

        # Deterministic cyclic fallback — guarantees k IF ≥k admissible exist.
        if len(selected) < k:
            start = int(rng.integers(0, n_pool))
            for off in range(n_pool):
                row = int(pool[(start + off) % n_pool])
                if row in seen or not _admissible(row):
                    continue
                seen.add(row)
                selected.append(row)
                if len(selected) >= k:
                    break

        if len(selected) < k:
            log.warning(
                f"[FaissIndex] random_retrieval: only {len(selected)} admissible "
                f"templates for '{query_name}' (< k={k}); returning all admissible."
            )
        return [(self.row2id[r], 0.5) for r in selected]
