from collections import defaultdict
import csv
import gc
import argparse
from datetime import datetime
import json
from pathlib import Path
import gzip
import re
from tqdm import tqdm
import gemmi
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TemporalSplitter:
    def _parse_date(self, s: str):
        s = str(s)
        if len(s) >= 10:
            s = s[:10]
        if not s or s == "?" or s == ".":
            return None
        try:
            return datetime.strptime(str(s), "%Y-%m-%d").date()
        except ValueError:
            return None

    def _get_cif_header_only(self, path: str, max_lines: int = 90000):
        """
        Read only the first `max_lines` lines and try to parse them with gemmi.
        This avoids fragile 'stop at atom_site loop' heuristics that may cut too early.
        """
        try:
            open_func = gzip.open if path.endswith(".gz") else open
            lines = []
            with open_func(path, "rt", encoding="latin-1", errors="replace") as f:
                for i, line in enumerate(f):
                    lines.append(line)
                    if i + 1 >= max_lines:
                        break

            doc = gemmi.cif.read_string("".join(lines))
            return doc.sole_block()
        except Exception:
            return None

    def get_release_date(self, mmcif_path: str):
        doc = None
        block = self._get_cif_header_only(mmcif_path)
        if block is None:
            logger.warning(
                f"Failed to read header of mmCIF file: {mmcif_path}. Trying full read."
            )
            try:
                doc = gemmi.cif.read(mmcif_path)
                block = doc.sole_block()
            except Exception:
                logger.error(f"Failed to read full mmCIF file: {mmcif_path}.")
                return {
                    "release_date": None,
                    "release_source": "read_error",
                    "status_code": None,
                    "deposition_date": None,
                }

        status = str(block.find_value("_pdbx_database_status.status_code")).upper()
        deposition_date = self._parse_date(
            block.find_value("_pdbx_database_status.recvd_initial_deposition_date")
        )

        result = {
            "release_date": None,
            "release_source": "missing",
            "status_code": status,
            "deposition_date": deposition_date,
        }

        # Priority 1 - PDBx Standard
        audit_loop = block.find(
            [
                "_pdbx_audit_revision_history.ordinal",
                "_pdbx_audit_revision_history.revision_date",
            ]
        )

        candidates = []
        if audit_loop:
            for row in audit_loop:
                d = self._parse_date(row[1])
                if d:
                    try:
                        ord_val = int(row[0])
                    except:
                        ord_val = 999
                    candidates.append((ord_val, d))

        if candidates:
            # Sort by ordinal (lowest = first rev. = release)
            candidates.sort(key=lambda x: (x[0], x[1]))
            result["release_date"] = candidates[0][1]
            result["release_source"] = "audit_history"
            del block
            if doc is not None:
                del doc
            return result

        # Priority 2 - Legacy Date Original
        date_orig = self._parse_date(
            block.find_value("_database_PDB_rev.date_original")
        )
        if date_orig:
            result["release_date"] = date_orig
            result["release_source"] = "db_rev_date_original"
            del block
            if doc is not None:
                del doc
            return result

        # Priority 3 - Legacy Loop
        rev_loop = block.find(["_database_PDB_rev.num", "_database_PDB_rev.date"])
        legacy_candidates = []
        if rev_loop:
            for row in rev_loop:
                d = self._parse_date(row[1])
                if d:
                    try:
                        num = int(row[0])
                    except:
                        num = 999
                    legacy_candidates.append((num, d))

        if legacy_candidates:
            legacy_candidates.sort(key=lambda x: (x[0], x[1]))
            result["release_date"] = legacy_candidates[0][1]
            result["release_source"] = "db_rev_loop_legacy"
            del block
            if doc is not None:
                del doc
            return result

        # No release date. Return only status and deposition.
        if status == "OBS":
            result["release_source"] = "obsolete_entry"
        else:
            result["release_source"] = "missing_release_date"

        del block
        if doc is not None:
            del doc
        return result

    def get_temporal_splits(self, mmcif_files, cutoff_datetime):
        stats = defaultdict(int)
        counter = 0

        logger.info("Assigning temporal splits...")
        for info in tqdm(mmcif_files.values()):
            mmcif_path = info["path"]

            try:
                res = self.get_release_date(mmcif_path)
            except Exception:
                # nie ubijaj całego runa przez jeden uszkodzony plik
                res = {
                    "release_date": None,
                    "release_source": "exception",
                    "status_code": None,
                    "deposition_date": None,
                }

            if res["release_date"] is None:
                info["release_date"] = ""
                info["date_source"] = res["release_source"]
                info["primary_date_source"] = res["release_source"]
                info["status_code"] = res["status_code"]
                info["deposition_date"] = (
                    res["deposition_date"].strftime("%Y-%m-%d")
                    if res["deposition_date"]
                    else ""
                )

                dep = res.get("deposition_date")
                if dep is not None and dep > cutoff_datetime:
                    info["split"] = "test"
                    info["date_source"] = "deposition_fallback"
                    stats["test"] += 1
                else:
                    stats["discarded"] += 1
                    info["split"] = "discarded"

                    src = res["release_source"]
                    stats[f"discarded_{src}"] += 1

                    if res["status_code"] == "OBS":
                        stats["discarded_obsolete"] += 1
                    else:
                        stats["discarded_risk"] += 1
                stats[info["date_source"]] += 1
            else:
                info["release_date"] = res["release_date"].strftime("%Y-%m-%d")
                info["date_source"] = res["release_source"]
                info["primary_date_source"] = res["release_source"]
                info["status_code"] = res["status_code"]
                info["deposition_date"] = (
                    res["deposition_date"].strftime("%Y-%m-%d")
                    if res["deposition_date"]
                    else ""
                )
                stats[res["release_source"]] += 1

                if res["release_date"] <= cutoff_datetime:
                    info["split"] = "train"
                    stats["train"] += 1
                else:
                    info["split"] = "test"
                    stats["test"] += 1

            counter += 1
            if counter % 100 == 0:
                gc.collect()
        logger.info("Done assigning temporal splits. Stats:")
        logger.info(dict(stats))
        return stats


class ClusterSplitter:
    def __init__(self, clusters_file, casp_csv_path):
        logger.info(f"Reading clusters from: {clusters_file}")
        self.clusters = []
        with open(clusters_file, "r") as f:
            for line in f:
                toks = line.strip().split()
                pdb_ids = [t.split("_")[0].lower() for t in toks if t]
                self.clusters.append(list(dict.fromkeys(pdb_ids)))
        logger.info(f"Num clusters: {len(self.clusters)}")
        logger.info(f"Reading CASP16 PDB IDs from: {casp_csv_path}")
        self.casp16_ids = self.read_casp16_pdb_ids(casp_csv_path)
        logger.info(f"Num CASP16 PDB IDs: {len(self.casp16_ids)}")

    def read_casp16_pdb_ids(self, casp_csv_path):
        pdb_ids = set()
        with open(casp_csv_path, newline="", encoding="latin-1", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                desc = row.get("Description", "")
                # PDB ID = ostatni 4-znakowy token
                tokens = desc.strip().split()
                if tokens:
                    last = tokens[-1].lower()
                    if re.fullmatch(r"[0-9][a-z0-9]{3}", last):
                        pdb_ids.add(last)
        return sorted(pdb_ids)

    def set_casp16_test_set(self, mmcif_files):
        stats = defaultdict(int)
        missing = []

        logger.info("Setting CASP16 test set...")
        for info in tqdm(mmcif_files.values(), desc="Initializing CASP16 flags"):
            info.setdefault("casp16_test_set", False)

        logger.info("Promoting CASP16 entries to test set...")
        for pid in tqdm(self.casp16_ids, desc="Setting CASP16 test set"):
            pid_lower = pid.lower()
            if pid_lower in mmcif_files:
                prev_split = mmcif_files[pid_lower].get("split", "none")
                mmcif_files[pid_lower]["split"] = "test"
                mmcif_files[pid_lower]["casp16_test_set"] = True
                stats[f"casp16_promoted_{prev_split}_to_test"] += 1
            else:
                stats["casp16_id_not_found"] += 1
                missing.append(pid_lower)
        logger.info("Done setting CASP16 test set. Stats:")
        logger.info(dict(stats))
        return stats, missing

    def apply_cluster_split(self, mmcif_files):
        stats = defaultdict(int)

        logger.info("Applying cluster-based splitting...")
        for cl in tqdm(self.clusters, desc="Cluster split"):
            members = [pid for pid in cl if pid in mmcif_files]
            if len(members) < 2:
                continue

            splits = [mmcif_files[pid].get("split") for pid in members]

            has_test = any(s == "test" for s in splits)
            has_train = any(s == "train" for s in splits)

            if has_test:
                for pid in members:
                    if mmcif_files[pid].get("split") == "train":
                        mmcif_files[pid]["split"] = "test"
                        mmcif_files[pid]["cluster_promoted"] = True
                        stats["promoted_train_to_test"] += 1
                stats["clusters_with_test"] += 1
            elif has_train:
                stats["clusters_train_only"] += 1
            else:
                stats["clusters_no_train_test"] += 1

        logger.info("Done applying cluster-based splitting. Stats:")
        logger.info(dict(stats))
        return stats


def read_mmcif_files(input_dir: str, limit: int = 0):
    """
    Recursively finds .cif and .cif.gz under input_dir.
    Returns dict: entity_id -> {"path": "..."}
    If limit>0, stops after collecting 'limit' files (prevents huge RAM use).
    """
    logger.info(f"Reading mmCIF files from: {input_dir}")
    mmcif_files = {}
    root = Path(input_dir)

    patterns = ["**/*.cif.gz", "**/*.cif"]
    for pat in patterns:
        for p in root.glob(pat):
            name = p.name
            if name.endswith(".cif.gz"):
                entity_id = name[:-7]
            elif name.endswith(".cif"):
                entity_id = name[:-4]
            else:
                continue

            entity_id = entity_id.lower()
            mmcif_files[entity_id] = {"path": str(p)}

            if limit and len(mmcif_files) >= limit:
                return mmcif_files

    logger.info(f"Found {len(mmcif_files)} mmCIF files.")
    return mmcif_files


def drop_unclustered_entries(mmcif_files: dict, exclude_file: str) -> dict:
    """Remove PDB entries that have no sequence-cluster assignment at all.

    Cluster membership is what the split guarantee and the same-cluster retrieval
    filter are both built on, so an entry without one cannot be placed safely in
    either. Excluding it here — before the first split is drawn — keeps that
    decision in data curation instead of leaving a runtime special case.

    Partially-clustered entries are kept; their individual unclustered chains are
    filtered later via `data/no_cluster_ids.txt` (see `data.skip_ids_files`).
    """
    if not exclude_file:
        logger.info("No exclude_entries_file given — keeping every mmCIF entry")
        return mmcif_files
    path = Path(exclude_file)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with:\n"
            f"  python scripts/resolve_chain_clusters.py --cluster-file <clusters_30*.txt>\n"
            f"or pass --exclude_entries_file '' to build splits without the filter."
        )
    excluded = {ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()}
    kept = {pid: info for pid, info in mmcif_files.items() if pid.lower() not in excluded}
    logger.info(
        f"Dropped {len(mmcif_files) - len(kept)} entries with no clustered chain "
        f"({len(excluded)} listed in {path}); {len(kept)} entries remain"
    )
    return kept


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2025/mmCIF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output_splits",
    )
    parser.add_argument(
        "--clusters_file",
        type=str,
        default="data/clusters_30_05_08_2026.txt",
    )
    parser.add_argument(
        "--exclude_entries_file",
        type=str,
        default="data/output_splits/no_cluster_entries.txt",
        help="PDB entries with no clustered chain at all, from "
             "scripts/resolve_chain_clusters.py. Dropped before any split is drawn: "
             "without a cluster they can be placed in neither the split nor the "
             "same-cluster retrieval filter. Pass '' to disable.",
    )
    parser.add_argument(
        "--casp_csv_path",
        type=str,
        default="data/targetlist.csv",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default="2024-04-30",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=100,
        help="How many mmCIF files to load into dict (RAM guard). 0 = all.",
    )

    args = parser.parse_args()
    cutoff_datetime = datetime.strptime(args.cutoff_date, "%Y-%m-%d").date()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mmcif_files = read_mmcif_files(args.input_dir, limit=args.limit_files)
    mmcif_files = drop_unclustered_entries(mmcif_files, args.exclude_entries_file)

    # Temporal splitting
    splitter = TemporalSplitter()
    temporal_stats = splitter.get_temporal_splits(mmcif_files, cutoff_datetime)
    
    # Save temporal_split flag BEFORE any promotions (for clean subset analysis)
    for pdb_id, info in mmcif_files.items():
        info["temporal_split"] = info.get("split", "unknown")
    
    with open(Path(args.output_dir) / "temporal_split_stats.txt", "w") as f:
        json.dump(dict(temporal_stats), f, indent=4)
    with open(Path(args.output_dir) / "mmcif_temporal_splits.json", "w") as f:
        json.dump(mmcif_files, f, indent=4)

    # Clusters splitting
    cluster_splitter = ClusterSplitter(args.clusters_file, args.casp_csv_path)
    casp_16_stats, missing = cluster_splitter.set_casp16_test_set(mmcif_files)
    with open(Path(args.output_dir) / "casp16_split_stats.txt", "w") as f:
        json.dump(dict(casp_16_stats), f, indent=4)
    if missing:
        logger.warning(
            f"The following CASP16 PDB IDs were not found in the dataset: {missing}"
        )
        with open(Path(args.output_dir) / "casp16_missing_ids.txt", "w") as f:
            f.write("\n".join(missing) + "\n")
    cluster_stats = cluster_splitter.apply_cluster_split(mmcif_files)
    with open(Path(args.output_dir) / "cluster_split_stats.txt", "w") as f:
        json.dump(dict(cluster_stats), f, indent=4)
    with open(Path(args.output_dir) / "mmcif_final_splits.json", "w") as f:
        json.dump(mmcif_files, f, indent=4)

    # Print sample results
    logger.info("Sample mmCIF file info after splitting:")
    for entity_id, info in list(mmcif_files.items())[:2]:
        logger.info(f"{entity_id}: {info}")

    # Final export for StructFuse
    train_ids = []
    test_ids = []
    
    # Subset lists for detailed evaluation
    test_gold_ids = []      # temporal-only, no casp16, no cluster_promoted (cleanest)
    test_casp16_ids = []    # CASP16 targets
    test_cluster_promoted_ids = []  # Promoted via cluster homology
    test_temporal_only_ids = []     # temporal test without casp16 (may include cluster_promoted)

    for pdb_id, info in mmcif_files.items():
        split = info.get("split")
        if split == "train":
            train_ids.append(pdb_id)
        elif split == "test":
            test_ids.append(pdb_id)
            
            is_casp16 = info.get("casp16_test_set", False)
            is_cluster_promoted = info.get("cluster_promoted", False)
            temporal_split = info.get("temporal_split", "unknown")
            
            if is_casp16:
                test_casp16_ids.append(pdb_id)
            
            if is_cluster_promoted:
                test_cluster_promoted_ids.append(pdb_id)
            
            # Temporal-only: was test before any promotions, not CASP16
            if temporal_split == "test" and not is_casp16:
                test_temporal_only_ids.append(pdb_id)
            
            # Gold: temporal test, not CASP16, not cluster_promoted (cleanest generalization claim)
            if temporal_split == "test" and not is_casp16 and not is_cluster_promoted:
                test_gold_ids.append(pdb_id)

    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)
    test_gold_ids = sorted(test_gold_ids)
    test_casp16_ids = sorted(test_casp16_ids)
    test_cluster_promoted_ids = sorted(test_cluster_promoted_ids)
    test_temporal_only_ids = sorted(test_temporal_only_ids)

    with open(Path(args.output_dir) / "all_train_ids.txt", "w") as f:
        f.write("\n".join(train_ids) + "\n")

    with open(Path(args.output_dir) / "all_test_ids.txt", "w") as f:
        f.write("\n".join(test_ids) + "\n")
    
    with open(Path(args.output_dir) / "test_gold_ids.txt", "w") as f:
        f.write("\n".join(test_gold_ids) + "\n")
    
    with open(Path(args.output_dir) / "test_casp16_ids.txt", "w") as f:
        f.write("\n".join(test_casp16_ids) + "\n")
    
    with open(Path(args.output_dir) / "test_cluster_promoted_ids.txt", "w") as f:
        f.write("\n".join(test_cluster_promoted_ids) + "\n")
    
    with open(Path(args.output_dir) / "test_temporal_only_ids.txt", "w") as f:
        f.write("\n".join(test_temporal_only_ids) + "\n")

    logger.info(f"Final TRAIN size: {len(train_ids)}")
    logger.info(f"Final TEST size: {len(test_ids)}")
    logger.info(f"  - Gold (temporal, no casp16, no cluster_promoted): {len(test_gold_ids)}")
    logger.info(f"  - CASP16: {len(test_casp16_ids)}")
    logger.info(f"  - Cluster promoted: {len(test_cluster_promoted_ids)}")
    logger.info(f"  - Temporal-only (no casp16): {len(test_temporal_only_ids)}")


if __name__ == "__main__":
    __main__()
