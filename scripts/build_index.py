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
      --exclude_ids data/output_splits/all_test_ids.txt
"""

import argparse
import gzip
import json
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from tqdm import tqdm
import torch

try:
    import rootutils
except ModuleNotFoundError:
    rootutils = None

if rootutils is not None:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
else:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
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


def _norm_chain_id(chain_id: str) -> str:
    """Normalize an index/cluster member ID for metadata lookups."""
    return str(chain_id).strip().lower()


def load_cluster_map(cluster_file: str) -> Dict[str, int]:
    """Parse clusters_30.txt into an entity/chain ID → cluster_id mapping.

    Each line in the file is one cluster whose members are space-separated
    entity- or chain-level IDs (e.g. ``12E8_2 15C8_2 ...``). We keep these IDs
    at entity/chain granularity. Collapsing to PDB/protein ID is incorrect for
    multi-entity structures because different entities from one PDB can belong
    to different sequence clusters.

    Returns:
        Dict mapping normalized member ID → cluster_id (int ≥ 0).
    """
    member2cluster: Dict[str, int] = {}
    n_clusters = 0
    n_duplicate_members = 0
    with open(cluster_file) as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            cid = n_clusters
            for tok in toks:
                member_id = _norm_chain_id(tok)
                if member_id in member2cluster and member2cluster[member_id] != cid:
                    n_duplicate_members += 1
                    continue
                member2cluster[member_id] = cid
            n_clusters += 1
    logger.info(
        f"Loaded {n_clusters} clusters covering {len(member2cluster)} entity/chain IDs"
    )
    if n_duplicate_members:
        logger.warning(
            f"Cluster file contains {n_duplicate_members} duplicate member assignments; "
            "kept first assignment for exact duplicate member IDs"
        )
    return member2cluster


def _split_strand_ids(value: str) -> List[str]:
    if value is None:
        return []
    raw = str(value).strip().strip("'\"")
    if raw in ("", ".", "?"):
        return []
    return [tok.strip().strip("'\"") for tok in raw.replace(";", ",").split(",") if tok.strip()]


def _read_cif_block_header(path: Path, max_lines: int = 120000):
    import gemmi

    def _read_full():
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="latin-1", errors="replace") as handle:
                return gemmi.cif.read_string(handle.read()).sole_block()
        return gemmi.cif.read(str(path)).sole_block()

    try:
        open_func = gzip.open if str(path).endswith(".gz") else open
        lines = []
        with open_func(path, "rt", encoding="latin-1", errors="replace") as handle:
            for i, line in enumerate(handle):
                lines.append(line)
                if i + 1 >= max_lines:
                    break
        return gemmi.cif.read_string("".join(lines)).sole_block()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not parse mmCIF header {path}: {exc}; trying full file")
        try:
            return _read_full()
        except Exception as full_exc:  # noqa: BLE001
            logger.warning(f"Could not parse mmCIF file {path}: {full_exc}")
            return None


def _entity_chain_map_from_block(pdb_id: str, block) -> Dict[str, str]:
    """Return normalized chain stem -> normalized entity stem for one mmCIF."""
    out: Dict[str, str] = {}
    pdb = _norm_chain_id(pdb_id)

    # Main PDBx source: entity_id with the comma-separated author strand IDs.
    loop = block.find(["_entity_poly.entity_id", "_entity_poly.pdbx_strand_id"])
    if loop:
        for row in loop:
            entity = str(row[0]).strip().strip("'\"")
            entity_stem = _norm_chain_id(f"{pdb}_{entity}")
            for chain in _split_strand_ids(row[1]):
                out[_norm_chain_id(f"{pdb}_{chain}")] = entity_stem

    # Fallback/augmentation: label asym IDs.
    loop = block.find(["_struct_asym.id", "_struct_asym.entity_id"])
    if loop:
        for row in loop:
            chain = str(row[0]).strip().strip("'\"")
            entity = str(row[1]).strip().strip("'\"")
            if chain not in ("", ".", "?") and entity not in ("", ".", "?"):
                out.setdefault(_norm_chain_id(f"{pdb}_{chain}"), _norm_chain_id(f"{pdb}_{entity}"))

    # Fallback/augmentation: auth/PDB strand IDs from sequence scheme.
    loop = block.find(
        [
            "_pdbx_poly_seq_scheme.entity_id",
            "_pdbx_poly_seq_scheme.asym_id",
            "_pdbx_poly_seq_scheme.pdb_strand_id",
        ]
    )
    if loop:
        for row in loop:
            entity = str(row[0]).strip().strip("'\"")
            entity_stem = _norm_chain_id(f"{pdb}_{entity}")
            for chain in (row[1], row[2]):
                chain = str(chain).strip().strip("'\"")
                if chain not in ("", ".", "?"):
                    out.setdefault(_norm_chain_id(f"{pdb}_{chain}"), entity_stem)
    return out


def build_chain_to_entity_map(mmcif_dir: Optional[Path], stems: Set[str]) -> Dict[str, str]:
    """Build chain stem -> entity stem mapping for the requested index stems."""
    if mmcif_dir is None:
        return {}
    if not mmcif_dir.exists():
        logger.warning(f"mmCIF directory not found: {mmcif_dir}; exact cluster IDs only")
        return {}

    wanted_pdbs = {_norm_chain_id(stem).split("_")[0] for stem in stems}
    paths: Dict[str, Path] = {}
    for pattern in ("**/*.cif.gz", "**/*.cif"):
        for path in mmcif_dir.glob(pattern):
            name = path.name
            if name.endswith(".cif.gz"):
                pdb = name[:-7].lower()
            elif name.endswith(".cif"):
                pdb = name[:-4].lower()
            else:
                continue
            if pdb in wanted_pdbs and pdb not in paths:
                paths[pdb] = path

    missing_paths = wanted_pdbs.difference(paths)
    if missing_paths:
        logger.warning(
            f"Missing mmCIF files for {len(missing_paths)} PDB IDs; "
            "their chain-level cluster IDs may remain unknown"
        )

    chain2entity: Dict[str, str] = {}
    for pdb, path in tqdm(paths.items(), desc="Parsing mmCIF entity-chain maps"):
        block = _read_cif_block_header(path)
        if block is None:
            continue
        chain2entity.update(_entity_chain_map_from_block(pdb, block))
    logger.info(f"Built chain→entity map for {len(chain2entity)} chain IDs")
    return chain2entity


def _allows_exact_cluster_lookup(norm_stem: str) -> bool:
    """Avoid treating numeric PDB chain IDs as polymer entity IDs.

    Cluster members such as ``6xmx_1`` usually denote polymer entities, while a
    processed chain stem with the same suffix can denote author chain ``1``.
    AlphaFold-style IDs are not PDB chain IDs and keep exact lookup.
    """
    if norm_stem.startswith("af_"):
        return True
    suffix = norm_stem.rsplit("_", 1)[-1] if "_" in norm_stem else ""
    return not suffix.isdigit()


def resolve_cluster_id(stem: str, member2cluster: Dict[str, int], chain2entity: Dict[str, str]) -> int:
    """Resolve a processed chain stem to the correct sequence-cluster ID."""
    norm = _norm_chain_id(stem)
    entity = chain2entity.get(norm)
    if entity is not None:
        return member2cluster.get(entity, -1)
    if _allows_exact_cluster_lookup(norm) and norm in member2cluster:
        return member2cluster[norm]
    return -1


def _load_precomputed_rep(
    pc_path: Path,
) -> Optional[Tuple[str, int, np.ndarray]]:
    """Load one precomputed ESM2 embedding NPZ, mean-pool across residues.

    Returns ``(stem, seq_len, emb_1280)`` or ``None`` if the file is
    unreadable/corrupted (caller skips). ``rep`` is already BOS/EOS-stripped
    by ``scripts/precompute_esm2_embeddings.py``.
    """
    try:
        with np.load(pc_path) as d:
            rep = d["rep"]  # (L, D) float16
            L = int(rep.shape[0])
            emb = rep.astype(np.float32).mean(axis=0)  # (D,)
        return pc_path.stem, L, emb
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        logger.warning(f"Skipping corrupted precomputed NPZ {pc_path.name}: {e}")
        return None


def build_faiss_index_from_precomputed(
    precomputed_dir: Path,
    processed_dir: Path,
    out_dir: Path,
    exclude_ids: Set[str] = None,
    id2cluster: Dict[str, int] = None,
    num_threads: int = 16,
) -> None:
    """Fast path: build FAISS index directly from precomputed ESM2 embeddings.

    Skips the ESM2 forward entirely (~4h → ~3min on 40k chains) by reading
    the per-residue ``rep`` tensors already cached at
    ``data/precomputed/esm_t33_650M/*.npz`` and mean-pooling them.

    Long sequences: ``precompute_esm2_embeddings.py`` truncates at
    ``max_len=1022`` whereas the ESM-forward path chunk-averages. For
    FAISS retrieval this is acceptable — and in fact MORE consistent, since
    at training time FAISS queries use the same precomputed embeddings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_ids = exclude_ids or set()
    id2cluster = id2cluster or {}

    processed_map = {p.stem: p for p in processed_dir.glob("*.npz")}
    logger.info(f"Found {len(processed_map)} NPZs in {processed_dir}")

    pc_files = sorted(precomputed_dir.glob("*.npz"))
    logger.info(f"Found {len(pc_files)} precomputed embeddings in {precomputed_dir}")

    # Filter out excluded PDB prefixes and stems without a processed NPZ.
    todo: List[Path] = []
    skipped_excluded = 0
    skipped_no_npz = 0
    for pc_path in pc_files:
        stem = pc_path.stem
        pdb_id = stem.split("_")[0]
        if pdb_id in exclude_ids:
            skipped_excluded += 1
            continue
        if stem not in processed_map:
            skipped_no_npz += 1
            continue
        todo.append(pc_path)
    logger.info(
        f"To embed: {len(todo)} (excluded={skipped_excluded}, no-npz={skipped_no_npz})"
    )

    ids_meta: List[Dict] = []
    emb_rows: List[np.ndarray] = []
    n_corrupt = 0

    # Parallel I/O: np.load is the bottleneck (disk-bound), threads are enough.
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for result in tqdm(
            pool.map(_load_precomputed_rep, todo),
            total=len(todo),
            desc="Mean-pooling precomputed embeddings",
        ):
            if result is None:
                n_corrupt += 1
                continue
            stem, L, emb = result
            if L == 0:
                continue
            ids_meta.append({
                "id": stem,
                "seq_len": L,
                "npz": str(processed_map[stem]),
                "cluster_id": id2cluster.get(_norm_chain_id(stem), -1),
            })
            emb_rows.append(emb)

    if n_corrupt > 0:
        logger.warning(f"Skipped {n_corrupt} corrupted precomputed NPZs (logged above)")

    if not emb_rows:
        raise RuntimeError("No embeddings collected — check precomputed/processed dirs")

    embeddings = np.stack(emb_rows).astype(np.float32)  # (N, D)

    logger.info("L2-normalising embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings /= norms

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={dim}, n={len(ids_meta)})")
    import faiss

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(f"Saving index artifacts to {out_dir}")
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "ids.json", "w") as f:
        json.dump(ids_meta, f, indent=2)

    logger.info("=" * 80)
    logger.info("FAISS index built via precomputed-embedding fast path:")
    logger.info(f"  Entries:    {len(ids_meta)}")
    logger.info(f"  Dimension:  {dim}")
    logger.info(f"  Output:     {out_dir / 'faiss.index'}")
    logger.info(f"  Metadata:   {out_dir / 'ids.json'}")
    logger.info(f"  Embeddings: {out_dir / 'embeddings.npy'}")
    logger.info("=" * 80)


def build_faiss_index(
    npz_files: List[Path],
    out_dir: Path,
    esm_model_name: str,
    batch_size: int,
    device: str,
    max_len: int,
    exclude_ids: Set[str] = None,
    id2cluster: Dict[str, int] = None,
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
        id2cluster (Dict[str, int], optional): normalized chain/entity ID → cluster_id mapping
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_ids = exclude_ids or set()
    id2cluster = id2cluster or {}

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

            cluster_id = id2cluster.get(_norm_chain_id(npz_path.stem), -1)
            ids_meta.append({
                "id": npz_path.stem,
                "seq_len": L,
                "npz": str(npz_path),
                "cluster_id": cluster_id,
            })
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
    import faiss

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
        default="data/output_splits/all_test_ids.txt",
        help="Path to text file with PDB IDs to exclude (e.g., test set)",
    )
    ap.add_argument(
        "--cluster_file",
        type=str,
        default="data/clusters_30.txt",
        help="Path to cluster file (one cluster per line, space-separated chain IDs). "
             "Used to embed cluster_id in index metadata for homolog filtering.",
    )
    ap.add_argument(
        "--mmcif_dir",
        type=str,
        default="data/mmcif",
        help="Directory with mmCIF files. Used to map processed chain IDs "
             "(e.g. 6xmx_H) to clustered polymer entity IDs (e.g. 6xmx_2).",
    )
    ap.add_argument(
        "--precomputed_embeddings_dir",
        type=str,
        default=None,
        help="If set, skip the ESM2 forward and build the index by mean-pooling "
             "per-residue embeddings from this directory (one NPZ per chain with "
             "key 'rep', as produced by scripts/precompute_esm2_embeddings.py). "
             "Dramatically faster (~4h → ~3min).",
    )
    ap.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Threads for parallel NPZ loading in the precomputed fast path.",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    # Load exclusion list. Empty string → build FULL index (no exclusions).
    # Use-case: we now keep val/test chains in the index and filter them at
    # query-time in FaissIndex.topk*; see PriorBuilder(holdout_id_files=...).
    exclude_ids: Set[str] = set()
    if args.exclude_ids:
        exclude_path = Path(args.exclude_ids)
        with open(exclude_path) as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(exclude_ids)} PDB IDs to exclude from {exclude_path}")
    else:
        logger.info("No exclusion list — building FULL index (val/test included)")

    # Load chain/entity-level cluster mapping.
    member2cluster: Dict[str, int] = {}
    chain2entity: Dict[str, str] = {}
    id2cluster: Dict[str, int] = {}
    if args.cluster_file:
        cluster_path = Path(args.cluster_file)
        if cluster_path.exists():
            member2cluster = load_cluster_map(str(cluster_path))
        else:
            logger.warning(f"Cluster file not found: {cluster_path} — building without cluster IDs")

    # Fast path: precomputed embeddings (skip ESM2 entirely)
    if args.precomputed_embeddings_dir:
        precomputed_dir = Path(args.precomputed_embeddings_dir)
        if not precomputed_dir.exists():
            raise RuntimeError(f"Precomputed dir not found: {precomputed_dir}")
        stems = {p.stem for p in precomputed_dir.glob("*.npz")}
        chain2entity = build_chain_to_entity_map(Path(args.mmcif_dir), stems)
        id2cluster = {
            _norm_chain_id(stem): resolve_cluster_id(stem, member2cluster, chain2entity)
            for stem in stems
        }
        build_faiss_index_from_precomputed(
            precomputed_dir=precomputed_dir,
            processed_dir=processed_dir,
            out_dir=out_dir,
            exclude_ids=exclude_ids,
            id2cluster=id2cluster,
            num_threads=args.num_threads,
        )
        return

    # Find NPZ files
    npz_files = sorted(processed_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise RuntimeError(f"No NPZ files found in {processed_dir}")

    logger.info(f"Found {len(npz_files)} NPZ files in {processed_dir}")
    stems = {p.stem for p in npz_files}
    chain2entity = build_chain_to_entity_map(Path(args.mmcif_dir), stems)
    id2cluster = {
        _norm_chain_id(stem): resolve_cluster_id(stem, member2cluster, chain2entity)
        for stem in stems
    }

    # Build FAISS index (legacy path: run ESM2 forward)
    build_faiss_index(
        npz_files=npz_files,
        out_dir=out_dir,
        esm_model_name=args.esm_model,
        batch_size=args.batch_size,
        device=args.device,
        max_len=args.max_len,
        exclude_ids=exclude_ids,
        id2cluster=id2cluster,
    )


if __name__ == "__main__":
    main()
