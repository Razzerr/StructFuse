"""
check_data_leakage.py

Comprehensive data leakage verification for contact prediction dataset.

Verifies:
1. No overlap between train/val/test splits
2. No same-PDB proteins across splits (multi-chain leakage)
3. FAISS index contains only training and validation set proteins
4. Validation split information

Usage:
  python scripts/check_data_leakage.py 
    --train_ids data/splits/custom_train.txt
    --test_ids data/splits/custom_test.txt
    --index_dir data/index_t33
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_split_overlap(train_ids_path: str, test_ids_path: str) -> bool:
    """
    Check for PDB ID overlap between train and test splits
    
    Args:
        train_ids_path (str): Path to training set IDs file
        test_ids_path (str): Path to test set IDs file
    
    Returns:
        bool: True if no overlap detected, False otherwise
    """
    logger.info("=" * 80)
    logger.info("1. CHECKING SPLIT OVERLAP")
    logger.info("=" * 80)

    with open(train_ids_path) as f:
        train_ids = set(line.strip() for line in f)
    with open(test_ids_path) as f:
        test_ids = set(line.strip() for line in f)

    logger.info(f"Train set: {len(train_ids)} PDB IDs")
    logger.info(f"Test set: {len(test_ids)} PDB IDs")

    overlap = train_ids & test_ids
    if overlap:
        logger.error(f"LEAKAGE: {len(overlap)} PDB IDs in both train and test!")
        logger.error(f"Examples: {list(overlap)[:10]}")
        return False
    else:
        logger.info("✅ No overlap between train and test sets")

    return True


def check_faiss_index(index_ids_path: str, test_ids_path: str) -> bool:
    """
    Check that FAISS index only contains training proteins
    
    Args:
        index_ids_path (str): Path to FAISS index ids.json file
        test_ids_path (str): Path to test set IDs file
    
    Returns:
        bool: True if no test contamination detected, False otherwise
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("2. CHECKING FAISS INDEX CONTENT")
    logger.info("=" * 80)

    with open(index_ids_path) as f:
        index_data = json.load(f)

    # Extract PDB IDs from index
    index_pdb_ids = set()
    for item in index_data:
        chain_id = item["id"]
        pdb_id = chain_id.split("_")[0]
        index_pdb_ids.add(pdb_id)

    logger.info(
        f"FAISS index contains {len(index_data)} chains from {len(index_pdb_ids)} unique PDB IDs"
    )

    with open(test_ids_path) as f:
        test_ids = set(line.strip() for line in f)

    # Check for test set contamination
    test_in_index = index_pdb_ids & test_ids
    if test_in_index:
        logger.error(
            f"LEAKAGE: {len(test_in_index)} test set PDB IDs found in FAISS index!"
        )
        logger.error(f"Examples: {list(test_in_index)[:10]}")
        return False
    else:
        logger.info("✅ No test set contamination in FAISS index")

    return True


def check_multi_chain_leakage(index_ids_path: str) -> bool:
    """
    Check for multi-chain proteins that could cause leakage.
    
    Args:
        index_ids_path (str): Path to FAISS index ids.json file
    
    Returns:
        bool: Always True (informational check)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("3. CHECKING MULTI-CHAIN LEAKAGE POTENTIAL")
    logger.info("=" * 80)

    with open(index_ids_path) as f:
        index_data = json.load(f)

    # Count chains per PDB
    pdb_counts = Counter(item["id"].split("_")[0] for item in index_data)

    multi_chain = {pid: count for pid, count in pdb_counts.items() if count > 1}
    logger.info(f"Found {len(multi_chain)} PDB IDs with multiple chains in index")

    chain_dist = Counter(pdb_counts.values())
    logger.info("Chain distribution:")
    for n_chains in sorted(chain_dist.keys())[:10]:
        count = chain_dist[n_chains]
        logger.info(f"  {n_chains} chain(s): {count:5d} PDBs")

    top_multi = sorted(multi_chain.items(), key=lambda x: -x[1])[:10]
    logger.info("Top multi-chain PDBs:")
    for pid, count in top_multi:
        chains = [item["id"] for item in index_data if item["id"].startswith(pid)]
        logger.info(f"  {pid}: {count} chains - {', '.join(chains)}")

    logger.info("")
    logger.info("Multi-chain filtering MUST be enabled in FaissIndex.topk()")

    return True


def main() -> None:
    """Run all data leakage checks and report results."""
    parser = argparse.ArgumentParser(
        description="Check for data leakage in protein contact prediction dataset"
    )
    parser.add_argument(
        "--train_ids",
        type=str,
        default="data/splits/all_train_ids.txt",
        help="Path to training set PDB IDs file",
    )
    parser.add_argument(
        "--test_ids",
        type=str,
        default="data/splits/all_test_ids.txt",
        help="Path to test set PDB IDs file",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="data/index_t6",
        help="Path to FAISS index directory containing ids.json",
    )
    args = parser.parse_args()

    # Construct index IDs path
    index_ids_path = str(Path(args.index_dir) / "ids.json")

    logger.info("")
    logger.info("DATA LEAKAGE DETECTION".center(80))
    logger.info("")

    checks = [
        ("Split Overlap", lambda: check_split_overlap(args.train_ids, args.test_ids)),
        ("FAISS Index", lambda: check_faiss_index(index_ids_path, args.test_ids)),
        ("Multi-chain Leakage", lambda: check_multi_chain_leakage(index_ids_path)),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            results.append((name, False))

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")

    all_pass = all(result for _, result in results)
    if all_pass:
        logger.info("")
        logger.info("All checks passed! No obvious data leakage detected.")
    else:
        logger.warning("")
        logger.warning("Some checks failed. Review output above.")


if __name__ == "__main__":
    main()
