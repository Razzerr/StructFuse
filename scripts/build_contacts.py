"""
build_contacts.py

Parse PDB files into per-chain caches containing:
- 1-letter sequence
- C-alpha coordinates
- binary contact map (C_alpha-C_alpha < cutoff Å, optional min sequence separation)
- residue mask (1 if C_alpha present)
- metadata (pdb_id, chain_id, path, cutoff, min_sep)

Output: one NPZ per chain: {pdbid}_{chain}.npz

Usage:
  python scripts/build_contacts.py \
    --pdb_dir data/pdb \
    --out_dir data/processed \
    --cutoff 8.0 \
    --min_sep 6 \
    --min_len 20 \
    --num_workers 8
"""

import argparse
import traceback
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import rootutils
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.utils.pdb_utils import parse_chain, contact_map_from_coords

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_pdb_file(
    path: Path,
    cutoff: float,
    min_sep: int,
    min_len: int,
) -> List[dict]:
    """
    Parse a PDB file and extract per-chain contact maps.

    Args:
        path: Path to PDB file
        cutoff: C_alpha distance threshold in Angstroms
        min_sep: Minimum sequence separation |i-j|
        min_len: Minimum chain length (residues)

    Returns:
        List[dict]: One dict per valid chain with keys:
            - seq: 1-letter sequence
            - coords: C_alpha coordinates
            - mask: residue validity mask
            - contact: binary contact map
            - L: sequence length
            - pdb_id: PDB identifier
            - chain_id: chain identifier
            - source_path: original file path
            - cutoff, min_sep: preprocessing parameters

    Raises:
        RuntimeError: If file cannot be parsed
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}") from e

    results = []
    model = next(structure.get_models())

    pdb_id = path.stem
    for chain in model:
        seq, coords, mask = parse_chain(chain)
        L = len(seq)
        if L < min_len:
            continue

        # Skip chains with no valid CA atoms
        if mask.sum() == 0:
            continue

        contact = contact_map_from_coords(coords, mask, cutoff=cutoff, min_sep=min_sep)

        results.append(
            dict(
                seq=np.array(seq, dtype=object),  # as object to keep exact string
                coords=coords.astype(np.float32),
                mask=mask.astype(np.uint8),
                contact=contact.astype(np.uint8),
                L=np.int32(L),
                pdb_id=pdb_id,
                chain_id=str(chain.id),
                source_path=str(path),
                cutoff=np.float32(cutoff),
                min_sep=np.int32(min_sep),
            )
        )
    return results


def _worker(
    args: Tuple[Path, Path, float, int, int, bool],
) -> Tuple[bool, Path, List[Path], Optional[str], int, int]:
    """
    Worker function for parallel PDB processing.

    Args:
        args: Tuple of (pdb_path, out_dir, cutoff, min_sep, min_len, overwrite)

    Returns:
        Tuple[bool, Path, List[Path], Optional[str], int, int]:
            - success: True if processed successfully
            - pdb_path: Input PDB file path
            - output_paths: List of saved NPZ file paths
            - traceback: Error traceback string if failed, else None
            - skipped: Number of chains skipped (already exist)
            - total: Total number of chains found
    """
    path, out_dir, cutoff, min_sep, min_len, overwrite = args

    try:
        items = process_pdb_file(
            path=path,
            cutoff=cutoff,
            min_sep=min_sep,
            min_len=min_len,
        )
        outputs = []
        skipped = 0
        for item in items:
            out_name = f"{item['pdb_id']}_{item['chain_id']}.npz"
            out_path = out_dir / out_name
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            np.savez_compressed(
                out_path,
                seq=item["seq"],
                coords=item["coords"],
                mask=item["mask"],
                contact=item["contact"],
                L=item["L"],
                pdb_id=item["pdb_id"],
                chain_id=item["chain_id"],
                source_path=item["source_path"],
                cutoff=item["cutoff"],
                min_sep=item["min_sep"],
            )
            outputs.append(out_path)
        return True, path, outputs, None, skipped
    except Exception:
        tb = traceback.format_exc()
        return False, path, [], tb, 0


def main():
    ap = argparse.ArgumentParser(
        description="Build per-chain contact caches from PDB files."
    )
    ap.add_argument(
        "--pdb_dir",
        type=str,
        default="data/pdb",
        help="Directory with PDB files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Directory to write NPZ files.",
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="CA distance cutoff in Å.",
    )
    ap.add_argument(
        "--min_sep",
        type=int,
        default=6,
        help="Min |i-j| to count as contact.",
    )
    ap.add_argument(
        "--min_len", type=int, default=20, help="Skip chains shorter than this length."
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel workers (0 = single process).",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing NPZ files."
    )
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting contact map generation with args:")
    logger.info(f"  PDB directory: {pdb_dir}")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  Contact cutoff: {args.cutoff} Å")
    logger.info(f"  Min sequence separation: {args.min_sep}")
    logger.info(f"  Min chain length: {args.min_len}")
    logger.info(f"  Num workers: {args.num_workers}")
    logger.info(f"  Overwrite existing: {args.overwrite}")
    logger.info("=" * 80)

    # Collect input files
    files = sorted([p for p in pdb_dir.rglob("*.pdb")])
    if len(files) == 0:
        logger.warning(f"No PDB files found under: {pdb_dir}")
        return
    logger.info(f"Found {len(files)} PDB files")

    n_failed = 0
    n_skipped = 0
    n_chains = 0

    out_dir = out_dir.resolve()
    worker_args = [
        (p, out_dir, args.cutoff, args.min_sep, args.min_len, args.overwrite)
        for p in files
    ]

    if args.num_workers and args.num_workers > 0:
        logger.info(f"Using {args.num_workers} workers")
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_worker, arg) for arg in worker_args]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing",
                unit="file",
            ):
                ok, path, outs, tb, skipped = fut.result()
                if ok:
                    n_skipped += skipped
                    n_chains += len(outs)
                    if skipped > 0:
                        logger.debug(
                            f"  {path.name}: ({skipped} chains skipped - already exist)"
                        )
                else:
                    n_failed += 1
                    logger.error(f"✗ {path.name} failed:\n{tb}")
    else:
        logger.info("Using single process")
        for arg in tqdm(worker_args, desc="Processing", unit="file"):
            ok, path, outs, tb, skipped = _worker(arg)
            if ok:
                n_skipped += skipped
                n_chains += len(outs)
                if len(outs) > 0:
                    logger.debug(f"✓ {path.name}: {len(outs)} chains saved")
            else:
                n_failed += 1
                logger.error(f"✗ {path.name} failed:\n{tb}")

    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info(f"  Total PDB files:       {len(files)}")
    logger.info(f"  Failed:                {n_failed}")
    logger.info(f"  Chains saved:          {n_chains}")
    logger.info(f"  Chains skipped:        {n_skipped} (already exist)")
    logger.info(f"  Output directory:      {out_dir}")
    logger.info("=" * 80)
    logger.info("Done!")


if __name__ == "__main__":
    main()
