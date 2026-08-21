#!/usr/bin/env python3
"""Pre-compute ESM2 embeddings for all proteins in the dataset.

Runs ESM2 once on each **full-length** sequence and saves per-residue
representations and contact maps so that training can skip the frozen
ESM2 forward pass entirely (~5-10x speedup).

Usage:
    python scripts/precompute_esm2_embeddings.py \
        --data_root data/processed_2026 \
        --output_dir data/esm2_embeddings \
        --model_name esm2_t33_650M_UR50D \
        --batch_size 4 \
        --device cuda

Outputs one NPZ per protein: {output_dir}/{stem}.npz with keys:
    rep:      (L, 1280) float16  — per-residue representations
    contacts: (L, L)    float16  — contact probabilities from attention
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.…` imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm


def length_aware_batches(items_with_paths, batch_size, max_pair_elems, max_len):
    """Group length-sorted items into batches bounded by B * L^2, not by B.

    ESM2 materialises one attention map per layer-head over the full L x L pair
    grid, so peak memory scales with `batch * n_maps * L^2`. A fixed sequence
    count is the wrong unit when lengths span 50x: a batch of 32 is trivial at
    L=100 and 16 GB at L=1022 on the 8M model (more on 650M, which has 660 maps
    to the 8M's 120). Because the caller sorts ascending by length, the newest
    item is always the longest in the batch, so the bound is exact.

    A single item that exceeds the budget on its own still gets its own batch —
    truncation to max_len is the only thing standing between us and OOM there.
    """
    batch = []
    for entry in items_with_paths:
        length = min(len(entry[0][1]), max_len)
        if batch and (
            len(batch) + 1 > batch_size
            or (max_pair_elems and (len(batch) + 1) * length * length > max_pair_elems)
        ):
            yield batch
            batch = []
        batch.append(entry)
    if batch:
        yield batch


def load_skip_stems(skip_files) -> set:
    """Union of chain stems listed in the given files, one stem per line.

    A missing file is an error rather than an empty set: silently embedding the
    unclustered chains would put them back into a pipeline that excludes them
    everywhere else.
    """
    stems: set = set()
    for raw in skip_files or []:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"skip-ids file not found: {path}. Run "
                f"scripts/resolve_chain_clusters.py first, or pass "
                f"--skip_ids_files with no values to disable filtering."
            )
        before = len(stems)
        stems.update(
            line.strip() for line in path.read_text().splitlines() if line.strip()
        )
        print(f"  skip list {path}: {len(stems) - before} new stems")
    return stems


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ESM2 embeddings")
    parser.add_argument("--data_root", type=str, default="data/processed_2026",
                        help="Directory with per-chain NPZ files")
    parser.add_argument("--output_dir", type=str, default="data/esm2_embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--model_name", type=str, default="esm2_t6_8M_UR50D",
                        help="ESM2 model variant")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for ESM2 forward (adjust to VRAM)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run ESM2 on")
    parser.add_argument("--max_len", type=int, default=1022,
                        help="Max sequence length (ESM2 limit is 1022 tokens)")
    parser.add_argument("--max_pair_elems", type=int, default=4_000_000,
                        help="Cap on batch*L^2 per forward. Peak memory is set by "
                             "the L x L attention maps, so this bounds it directly "
                             "where --batch_size cannot: at L=1022 it yields "
                             "batches of 3, at L=384 batches of 27, and short "
                             "chains stay limited by --batch_size. Set 0 to "
                             "disable and batch purely by sequence count.")
    parser.add_argument("--skip_ids_files", type=str, nargs="*",
                        default=["data/corrupt_ids.txt",
                                 "data/output_splits_2026/no_cluster_ids.txt"],
                        help="Files listing chain stems to exclude, one per line. "
                             "Chains with no RCSB cluster assignment are excluded "
                             "from the whole pipeline (splits, index, training), so "
                             "they must not receive embeddings either. Pass with no "
                             "values to disable filtering.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all NPZ files
    npz_files = sorted(data_root.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files in {data_root}")

    # Drop chains excluded from the pipeline (corrupt, or no RCSB cluster)
    skip_stems = load_skip_stems(args.skip_ids_files)
    if skip_stems:
        n_before = len(npz_files)
        npz_files = [p for p in npz_files if p.stem not in skip_stems]
        print(f"Excluded {n_before - len(npz_files)} chains via skip lists "
              f"({len(skip_stems)} stems listed), {len(npz_files)} remaining")

    # Skip already processed
    todo = []
    for npz_path in tqdm(npz_files, desc="Checking existing embeddings"):
        out_path = output_dir / npz_path.name
        if not out_path.exists():
            todo.append(npz_path)
    print(f"Skipping {len(npz_files) - len(todo)} already processed, {len(todo)} remaining")

    if not todo:
        print("Nothing to do.")
        return

    # Load ESM2
    from src.models.components.esm import pretrained
    model, alphabet = getattr(pretrained, args.model_name)()
    model = model.to(args.device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    # Load sequences from NPZ files
    def load_seq(npz_path: Path) -> tuple:
        with np.load(npz_path, allow_pickle=True) as data:
            seq_arr = data["seq"]
            seq = (
                str(seq_arr.item())
                if isinstance(seq_arr, np.ndarray) and seq_arr.shape == ()
                else str(seq_arr)
            )
        stem = npz_path.stem
        return stem, seq

    # Process in batches
    items = [load_seq(p) for p in tqdm(todo, desc="Loading sequences")]

    # Sort by length for efficient batching (less padding waste)
    items_with_paths = list(zip(items, todo))
    items_with_paths.sort(key=lambda x: len(x[0][1]))

    n_processed = 0
    n_truncated = 0

    batches = list(length_aware_batches(
        items_with_paths, args.batch_size, args.max_pair_elems, args.max_len
    ))
    sizes = [len(b) for b in batches]
    print(f"Batching: {len(batches)} batches, size min={min(sizes)} max={max(sizes)} "
          f"(batch_size={args.batch_size}, max_pair_elems={args.max_pair_elems})")

    for batch_items in tqdm(batches, desc="Computing embeddings"):

        seq_list = []
        for (stem, seq), npz_path in batch_items:
            if len(seq) > args.max_len:
                seq = seq[:args.max_len]
                n_truncated += 1
            seq_list.append((stem, seq))

        _, _, tokens = batch_converter(seq_list)
        tokens = tokens.to(args.device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[num_layers], return_contacts=True)
            reps = out["representations"][num_layers][:, 1:-1, :]  # (B, L, D)
            contacts = out["contacts"]  # (B, L, L) — already stripped BOS/EOS

        # Save each sample individually
        for i, ((stem, seq), npz_path) in enumerate(batch_items):
            L = min(len(seq), args.max_len)
            rep_np = reps[i, :L, :].cpu().half().numpy()       # (L, 1280) float16
            cont_np = contacts[i, :L, :L].cpu().half().numpy()  # (L, L) float16

            out_path = output_dir / f"{stem}.npz"
            np.savez_compressed(out_path, rep=rep_np, contacts=cont_np)
            n_processed += 1

    print(f"Done. Processed {n_processed} proteins, {n_truncated} truncated to {args.max_len}.")


if __name__ == "__main__":
    main()
