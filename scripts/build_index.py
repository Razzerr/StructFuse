"""
build_index.py

Build FAISS retrieval index for template-based contact prediction.

Processes NPZ files from build_contacts.py to create:
- FAISS index over L2-normalized ESM2 embeddings (for cosine similarity search)
- Metadata mapping: chain IDs to sequence lengths and NPZ paths
- Optional: saved embeddings matrix

Usage:
  python scripts/build_index.py \
      --processed_dir data/processed \
      --out_dir data/index_t6 \
      --esm_model esm2_t6_8M_UR50D \
      --batch_size 8 \
      --device cuda \
      --exclude_ids data/splits/all_test_ids.txt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Set

import rootutils
import numpy as np
from tqdm import tqdm
import torch
import faiss

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.esm import pretrained

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_seq_from_npz(npz_path: Path) -> Tuple[str, int]:
    """
    Load sequence from NPZ file.

    Args:
        npz_path (Path): Path to NPZ file containing 'seq' field

    Returns:
        Tuple[str, int]: (sequence, length)
    """
    d = np.load(npz_path, allow_pickle=True)
    seq = d["seq"]
    if isinstance(seq, np.ndarray) and seq.shape == ():
        seq = str(seq.item())
    else:
        seq = str(seq)
    return seq, len(seq)


def _mean_pool_representations(reps: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations, excluding padding."""
    masked = reps * masks.unsqueeze(-1)
    sums = masked.sum(dim=1)
    lens = masks.sum(dim=1).clamp_min(1)
    return (sums / lens.unsqueeze(-1)).float()


def build_faiss_index(
    npz_files: List[Path],
    out_dir: Path,
    esm_model_name: str,
    batch_size: int,
    device: str,
    max_len: int,
    exclude_ids: Set[str] = None,
) -> None:
    """
    Build FAISS index from ESM2 embeddings of protein sequences.

    Args:
        npz_files (List[Path]): List of NPZ file paths
        out_dir (Path): Output directory for index artifacts
        esm_model_name (str): ESM2 model name (e.g., 'esm2_t6_8M_UR50D')
        batch_size (int): Batch size for embedding
        device (str): 'cuda' or 'cpu'
        max_len (int): Maximum sequence length per chunk (excluding BOS/EOS)
        exclude_ids (Set[str], optional): Set of PDB IDs to exclude (e.g., test set). Defaults to None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_ids = exclude_ids or set()

    # Scan NPZ files and filter by exclusion list
    ids_meta = []
    seqs = []
    skipped = 0

    for npz_path in tqdm(npz_files, desc="Scanning NPZ files"):
        try:
            pdb_id = npz_path.stem.split("_")[0]
            if pdb_id in exclude_ids:
                skipped += 1
                continue

            seq, L = load_seq_from_npz(npz_path)
            if L == 0:
                continue

            ids_meta.append({"id": npz_path.stem, "seq_len": L, "npz": str(npz_path)})
            seqs.append((npz_path.stem, seq))
        except Exception as e:
            logger.warning(f"Skipping {npz_path.name}: {e}")

    if skipped > 0:
        logger.info(f"Excluded {skipped} chains from {len(exclude_ids)} PDB IDs")

    if len(seqs) == 0:
        raise RuntimeError("No sequences found to embed")

    # Load ESM2 model
    logger.info(f"Loading ESM2 model: {esm_model_name}")
    model, alphabet = getattr(pretrained, esm_model_name)()
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    # Embed sequences in batches
    logger.info(f"Embedding {len(seqs)} sequences with ESM2 (batch_size={batch_size})")
    embed_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="Embedding"):
            batch = seqs[i : i + batch_size]

            # Check if any sequence in batch exceeds max_len
            has_long_seq = any(len(seq) > max_len for _, seq in batch)

            if has_long_seq:
                # Sequences > max_len (default 1022 for ESM2): chunk embeddings
                pooled = []
                for name, seq in batch:
                    chunks = []
                    for start in range(0, len(seq), max_len):
                        s_chunk = seq[start : start + max_len]
                        _, _, tokens = batch_converter([(name, s_chunk)])
                        if device.startswith("cuda") and torch.cuda.is_available():
                            tokens = tokens.cuda(non_blocking=True)
                        out = model(
                            tokens,
                            repr_layers=[model.num_layers],
                            need_head_weights=False,
                        )
                        # strip BOS/EOS
                        rep = out["representations"][model.num_layers][:, 1:-1, :]
                        mask = torch.ones(
                            rep.shape[:2], dtype=torch.float32, device=rep.device
                        )
                        chunks.append(_mean_pool_representations(rep, mask))
                    pooled.append(torch.stack(chunks, dim=0).mean(dim=0))
                X = torch.cat(pooled, dim=0)
            else:
                # Normal batch processing
                _, _, tokens = batch_converter(batch)
                if device.startswith("cuda") and torch.cuda.is_available():
                    tokens = tokens.cuda(non_blocking=True)
                out = model(
                    tokens, repr_layers=[model.num_layers], need_head_weights=False
                )
                rep = out["representations"][model.num_layers][:, 1:-1, :]  # (B, L, D)
                mask = (tokens != alphabet.padding_idx).float()[:, 1:-1]  # (B, L)
                X = _mean_pool_representations(rep, mask)  # (B, D)
            embed_list.append(X.cpu().numpy())

    embeddings = np.concatenate(embed_list, axis=0).astype(np.float32)  # (N, D)

    # L2-normalize for cosine similarity
    logger.info("Normalizing embeddings (L2 norm)")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings /= norms

    # Build FAISS index (IndexFlatIP for inner product = cosine similarity on normalized vectors)
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={dim}, n={len(ids_meta)})")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save artifacts
    logger.info(f"Saving index artifacts to {out_dir}")
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "ids.json", "w") as f:
        json.dump(ids_meta, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"FAISS index built successfully:")
    logger.info(f"  Entries: {len(ids_meta)}")
    logger.info(f"  Dimension: {dim}")
    logger.info(f"  Output: {out_dir / 'faiss.index'}")
    logger.info(f"  Metadata: {out_dir / 'ids.json'}")
    logger.info(f"  Embeddings: {out_dir / 'embeddings.npy'}")
    logger.info("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from ESM2 embeddings.")
    ap.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory with NPZ files from build_contacts.py",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/index",
        help="Output directory for index artifacts",
    )
    ap.add_argument(
        "--esm_model",
        type=str,
        default="esm2_t6_8M_UR50D",
        help="ESM2 model name",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device: 'cuda' or 'cpu'",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=1022,
        help="Max tokens per chunk (excluding BOS/EOS)",
    )
    ap.add_argument(
        "--exclude_ids",
        type=str,
        default="data/splits/all_test_ids.txt",
        help="Path to text file with PDB IDs to exclude (e.g., test set)",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    # Load exclusion list
    exclude_ids = set()
    if args.exclude_ids:
        exclude_path = Path(args.exclude_ids)
        with open(exclude_path) as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(exclude_ids)} PDB IDs to exclude from {exclude_path}")

    # Find NPZ files
    npz_files = sorted(processed_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise RuntimeError(f"No NPZ files found in {processed_dir}")

    logger.info(f"Found {len(npz_files)} NPZ files in {processed_dir}")

    # Build FAISS index
    build_faiss_index(
        npz_files=npz_files,
        out_dir=out_dir,
        esm_model_name=args.esm_model,
        batch_size=args.batch_size,
        device=args.device,
        max_len=args.max_len,
        exclude_ids=exclude_ids,
    )


if __name__ == "__main__":
    main()
