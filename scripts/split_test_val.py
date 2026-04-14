"""Split the evaluated test set into val_holdout (~40%) and test (~60%).

Cluster-aware: entire clusters go to one split to avoid data leakage.
CASP16 proteins are forced into the test split.
Uses union-find on PDB codes that share a cluster to group them.

Usage:
    python scripts/split_test_val.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.4

ROOT = Path(__file__).resolve().parent.parent
SPLITS_JSON = ROOT / "data" / "output_splits" / "mmcif_final_splits.json"
TEST_IDS_FILE = ROOT / "data" / "output_splits" / "all_test_ids.txt"
CLUSTERS_FILE = ROOT / "data" / "clusters_30.txt"
OUT_DIR = ROOT / "data" / "splits"


# ── Union-Find ──────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def main():
    # 1) Load evaluated test proteins (gold + CASP16, no cluster_promoted)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    with open(TEST_IDS_FILE) as f:
        test_ids = {l.strip() for l in f if l.strip()}

    eval_test = set()
    casp_ids = set()
    for pid, info in splits.items():
        if pid in test_ids and not info.get("cluster_promoted", False):
            eval_test.add(pid)
            if info.get("casp16_test_set", False):
                casp_ids.add(pid)

    print(f"Evaluated test proteins: {len(eval_test)}")
    print(f"  CASP16: {len(casp_ids)}")
    print(f"  Gold (non-CASP): {len(eval_test - casp_ids)}")

    # 2) Load clusters and build PDB→cluster membership
    # Cluster IDs are PDB_chain (uppercase); test IDs are PDB (lowercase).
    eval_upper = {pid.upper() for pid in eval_test}
    casp_upper = {pid.upper() for pid in casp_ids}

    # For each cluster, collect which eval PDB codes appear (via chain prefix)
    uf = UnionFind()
    pdb_in_eval = set()

    with open(CLUSTERS_FILE) as f:
        for line in f:
            members = line.strip().split()
            if not members:
                continue
            # Extract PDB codes (part before _) that are in eval_test
            cluster_pdbs = set()
            for m in members:
                pdb = m.split("_")[0].upper()
                if pdb.lower() in eval_test:
                    cluster_pdbs.add(pdb.lower())

            if len(cluster_pdbs) >= 2:
                # Union all PDBs in this cluster
                pdbs_list = list(cluster_pdbs)
                for i in range(1, len(pdbs_list)):
                    uf.union(pdbs_list[0], pdbs_list[i])
            for p in cluster_pdbs:
                pdb_in_eval.add(p)

    # Ensure all eval proteins are in UF (including singletons)
    for pid in eval_test:
        uf.find(pid)

    # 3) Build connected components (groups)
    groups = defaultdict(set)
    for pid in eval_test:
        root = uf.find(pid)
        groups[root].add(pid)

    print(f"\nConnected components (groups): {len(groups)}")
    sizes = sorted([len(g) for g in groups.values()], reverse=True)
    print(f"  Largest 10 group sizes: {sizes[:10]}")
    print(f"  Singleton groups: {sum(1 for s in sizes if s == 1)}")

    # 4) Classify groups: CASP-containing → forced to test
    casp_groups = []
    non_casp_groups = []
    for root, members in groups.items():
        if members & casp_ids:
            casp_groups.append(members)
        else:
            non_casp_groups.append(members)

    casp_forced_count = sum(len(g) for g in casp_groups)
    print(f"\nCASP-containing groups: {len(casp_groups)} ({casp_forced_count} proteins)")
    print(f"Non-CASP groups: {len(non_casp_groups)} ({sum(len(g) for g in non_casp_groups)} proteins)")

    # 5) Shuffle and split non-CASP groups
    rng = random.Random(SEED)
    rng.shuffle(non_casp_groups)

    target_val = int(len(eval_test) * VAL_FRACTION)
    val_proteins = set()
    test_proteins = set()

    # Add to val until we reach target
    for g in non_casp_groups:
        if len(val_proteins) + len(g) <= target_val:
            val_proteins |= g
        else:
            test_proteins |= g

    # Add all CASP groups to test
    for g in casp_groups:
        test_proteins |= g

    print(f"\n── Split Results ──")
    print(f"Val holdout:  {len(val_proteins)} ({len(val_proteins)/len(eval_test)*100:.1f}%)")
    print(f"Test:         {len(test_proteins)} ({len(test_proteins)/len(eval_test)*100:.1f}%)")
    print(f"Total:        {len(val_proteins) + len(test_proteins)}")
    print(f"CASP16 in test: {len(test_proteins & casp_ids)}")
    print(f"CASP16 in val:  {len(val_proteins & casp_ids)}")

    # Verify no overlap
    assert not (val_proteins & test_proteins), "Overlap between val and test!"
    assert val_proteins | test_proteins == eval_test, "Missing proteins!"

    # 6) Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_file = OUT_DIR / "val_holdout_ids.txt"
    test_file = OUT_DIR / "test_ids.txt"

    with open(val_file, "w") as f:
        for pid in sorted(val_proteins):
            f.write(pid + "\n")

    with open(test_file, "w") as f:
        for pid in sorted(test_proteins):
            f.write(pid + "\n")

    print(f"\nWritten: {val_file}")
    print(f"Written: {test_file}")


if __name__ == "__main__":
    main()
