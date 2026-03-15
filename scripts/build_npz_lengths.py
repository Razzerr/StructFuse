#!/usr/bin/env python
"""
Build npz_lengths.json — a mapping of NPZ stem -> sequence length L.

This index is required by ContactDataset so that training starts instantly
instead of opening every NPZ file to read L.

Usage:
    python scripts/build_npz_lengths.py --processed_dir data/processed

The output file is written to <processed_dir>/npz_lengths.json.
Run again whenever you add / remove NPZ files.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build npz_lengths.json index")
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing *.npz files (default: data/processed)",
    )
    args = parser.parse_args()

    root = args.processed_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    all_npz = sorted(root.glob("*.npz"))
    log.info(f"Found {len(all_npz)} NPZ files in {root}")

    lengths = {}
    failed = 0
    for npz_path in tqdm(all_npz, desc="Reading lengths"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            lengths[npz_path.stem] = int(data["L"])
        except Exception as e:
            log.warning(f"Failed to read {npz_path.name}: {e}")
            failed += 1

    out_path = root / "npz_lengths.json"
    with open(out_path, "w") as f:
        json.dump(lengths, f)

    log.info(f"Done: {len(lengths)} entries written to {out_path}")
    if failed:
        log.warning(f"{failed} files failed to read")


if __name__ == "__main__":
    main()
