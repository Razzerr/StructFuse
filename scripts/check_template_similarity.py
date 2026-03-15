"""
Diagnostic: check distribution of cosine similarities for retrieved templates.
Samples chains directly from the FAISS index (which contains chain-level IDs).
"""
import json
import os
import sys
import numpy as np
import torch

ROOT = "/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse"
sys.path.insert(0, ROOT)

from src.models.utils.faiss import FaissIndex

INDEX_DIR = os.path.join(ROOT, "data/index_t6")

# Load ESM model
print("Loading ESM2...")
from src.models.components.esm import pretrained
model, alphabet = pretrained.esm2_t6_8M_UR50D()
model = model.eval().cuda()

# Load index
print("Loading FAISS index...")
idx = FaissIndex(INDEX_DIR)

# Sample 200 random chains from the index
np.random.seed(42)
n_sample = 200
sample_indices = np.random.choice(len(idx.meta), size=n_sample, replace=False)

print(f"Sampling {n_sample} chains from index ({len(idx.meta)} total)...")

K_VALUES = [4, 10, 25, 50, 100, 200, 500]
max_k = max(K_VALUES)

all_hits_by_rank = {}  # rank -> list of sims
skipped = 0

for i, row_idx in enumerate(sample_indices):
    m = idx.meta[row_idx]
    pid = m["id"]
    npz_path = os.path.join(ROOT, m["npz"])

    if not os.path.exists(npz_path):
        skipped += 1
        continue

    td = np.load(npz_path, allow_pickle=True)
    seq_arr = td["seq"]
    seq = str(seq_arr.item()) if isinstance(seq_arr, np.ndarray) and seq_arr.shape == () else str(seq_arr)
    td.close()

    hits = idx.topk(model, alphabet, pid, seq, k=max_k, device="cuda")
    if not hits:
        continue

    for rank, (tpl_id, sim) in enumerate(hits):
        all_hits_by_rank.setdefault(rank, []).append(sim)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{n_sample} done...")

print(f"\n{'='*60}")
print(f"Results: {n_sample} queries, {skipped} skipped")

# Per-rank statistics
print(f"\nSimilarity by rank position:")
print(f"  {'Rank':>5s}  {'Mean':>7s}  {'Median':>7s}  {'Min':>7s}  {'Max':>7s}  {'<0.95':>6s}  {'<0.90':>6s}  {'<0.80':>6s}  {'<0.70':>6s}  {'N':>5s}")
for rank in sorted(all_hits_by_rank.keys()):
    sims = np.array(all_hits_by_rank[rank])
    n = len(sims)
    below_95 = (sims < 0.95).sum()
    below_90 = (sims < 0.90).sum()
    below_80 = (sims < 0.80).sum()
    below_70 = (sims < 0.70).sum()
    print(f"  {rank+1:>5d}  {sims.mean():>7.4f}  {np.median(sims):>7.4f}  "
          f"{sims.min():>7.4f}  {sims.max():>7.4f}  "
          f"{below_95:>5d}  {below_90:>5d}  {below_80:>5d}  {below_70:>5d}  {n:>5d}")

# Aggregate stats per topk setting
for k in K_VALUES:
    sims_k = []
    for rank in range(k):
        if rank in all_hits_by_rank:
            sims_k.extend(all_hits_by_rank[rank])
    sims_k = np.array(sims_k)
    print(f"\n--- topk={k}: {len(sims_k)} hits ---")
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    counts, _ = np.histogram(sims_k, bins=bins)
    for j in range(len(bins) - 1):
        pct = 100 * counts[j] / max(len(sims_k), 1)
        bar = "#" * int(pct / 2)
        print(f"  [{bins[j]:.2f}, {bins[j+1]:.2f}): {counts[j]:5d} ({pct:5.1f}%) {bar}")

# === SIMULATION: similarity thresholds ===
# For each query, we already have top-500 hits. Simulate picking top-4
# from those hits after filtering by a similarity threshold.
print(f"\n{'='*60}")
print(f"SIMULATION: 'fetch 500, keep top-4 below threshold'")
print(f"  How many queries get 0, 1, 2, 3, 4 templates at each threshold?\n")

# Collect all hits per query
per_query_hits = {}  # query_idx -> list of (rank, sim)
for rank in sorted(all_hits_by_rank.keys()):
    for q_idx, sim in enumerate(all_hits_by_rank[rank]):
        per_query_hits.setdefault(q_idx, []).append(sim)

n_queries = len(per_query_hits)
want_k = 4

thresholds = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80]
print(f"  {'Thresh':>7s}  {'avg_k':>6s}  {'med_k':>6s}  {'0 tpl':>6s}  {'1 tpl':>6s}  "
      f"{'2 tpl':>6s}  {'3 tpl':>6s}  {'4 tpl':>6s}  {'avg_sim':>8s}  {'med_sim':>8s}")

for thresh in thresholds:
    counts_per_query = []
    sims_selected = []
    for q_idx in range(n_queries):
        hits = per_query_hits.get(q_idx, [])
        # Filter: keep only below threshold, take top want_k (highest sim first)
        filtered = sorted([s for s in hits if s < thresh], reverse=True)[:want_k]
        counts_per_query.append(len(filtered))
        sims_selected.extend(filtered)
    
    counts_arr = np.array(counts_per_query)
    hist = [int((counts_arr == i).sum()) for i in range(want_k + 1)]
    avg_sim = np.mean(sims_selected) if sims_selected else 0.0
    med_sim = np.median(sims_selected) if sims_selected else 0.0
    
    print(f"  {thresh:>7.2f}  {counts_arr.mean():>6.2f}  {np.median(counts_arr):>6.1f}  "
          f"{hist[0]:>6d}  {hist[1]:>6d}  {hist[2]:>6d}  {hist[3]:>6d}  {hist[4]:>6d}  "
          f"{avg_sim:>8.4f}  {med_sim:>8.4f}")
