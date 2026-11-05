#!/usr/bin/env python3
"""Re-derive the splits after the entry exclusion list changed, without redrawing them.

Use case (2026-11-05): the chain-ID case fix shrank `no_cluster_entries.txt`, so 32
PDB entries that `prepare_data_splits.drop_unclustered_entries` had removed must
now take part in the split. A full re-run would re-parse every mmCIF for dates
(single process, hours) and `split_test_val.py` is not reproducible across
interpreter runs (it iterates a `set`, whose order depends on hash randomisation),
so it would redraw val/test for everything. Instead:

1. Load the archived pre-promotion temporal assignment (`mmcif_temporal_splits.json`).
   GATE A: re-running the production CASP16 + cluster-promotion pass on it must
   reproduce the archived `mmcif_final_splits.json` exactly.
2. Date the added entries with the production `TemporalSplitter`, drop removed
   ones, and run the same CASP16 + cluster pass over the whole set (the pass is
   in-memory and cheap; its result does not depend on dict order).
3. Evaluated entries (test, not cluster-promoted) keep their archived val/test
   assignment. A new evaluated entry joins the side of the archived entries it is
   cluster-connected to (union-find over the cluster file, as split_test_val does);
   GATE B refuses a group that would connect val and test. A group with no archived
   member goes to test if it holds a CASP16 entry, otherwise by a fixed hash of its
   sorted members (fraction --val-fraction to val), so the rule is reproducible.

Every change to an existing entry (e.g. a train entry promoted because an added
test entry shares its cluster) is listed in extend_report.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_data_splits as pds  # noqa: E402


def read_ids(path: Path) -> set:
    return {ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()}


def cluster_pass(temporal: dict, clusters_file: str, casp_csv: str) -> tuple[dict, dict, dict]:
    """The production CASP16 + cluster-promotion pass on a copy of `temporal`."""
    files = copy.deepcopy(temporal)
    splitter = pds.ClusterSplitter(clusters_file, casp_csv)
    casp_stats, _missing = splitter.set_casp16_test_set(files)
    cluster_stats = splitter.apply_cluster_split(files)
    return files, dict(casp_stats), dict(cluster_stats)


def evaluated(final: dict) -> set:
    return {p for p, i in final.items() if i.get("split") == "test" and not i.get("cluster_promoted", False)}


def eval_groups(eval_ids: set, clusters_file: str) -> dict:
    """Connected components of evaluated entries sharing a cluster (split_test_val's rule)."""
    parent = {p: p for p in eval_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with open(clusters_file) as fh:
        for line in fh:
            pdbs = sorted({m.split("_")[0].lower() for m in line.split()} & eval_ids)
            for other in pdbs[1:]:
                ra, rb = find(pdbs[0]), find(other)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(set)
    for p in eval_ids:
        groups[find(p)].add(p)
    return groups


def hash_fraction(members) -> float:
    digest = hashlib.sha256(",".join(sorted(members)).encode()).hexdigest()
    return int(digest, 16) / 16 ** len(digest)


def place_new_evaluated(groups: dict, val_old: set, test_old: set, casp: set, val_fraction: float):
    val_new, test_new, decisions, conflicts = set(), set(), [], []
    for members in groups.values():
        new = members - val_old - test_old
        if not new:
            continue
        in_val, in_test = members & val_old, members & test_old
        if in_val and in_test:
            conflicts.append(sorted(members))
            continue
        if in_val or in_test:
            side, reason = ("val" if in_val else "test"), "joins archived cluster-connected entries"
        elif members & casp:
            side, reason = "test", "new group with a CASP16 entry"
        else:
            side = "val" if hash_fraction(members) < val_fraction else "test"
            reason = "new group, fixed hash of sorted members"
        (val_new if side == "val" else test_new).update(new)
        decisions.append({"entries": sorted(new), "group_size": len(members), "side": side, "reason": reason})
    return val_new, test_new, decisions, conflicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dir", type=Path, default=Path("data/output_splits_2026"))
    ap.add_argument("--new-exclude", type=Path, required=True, help="corrected no_cluster_entries.txt")
    ap.add_argument("--chain-clusters-dir", type=Path, required=True,
                    help="dir with the corrected chain_clusters.tsv / no_cluster_ids.txt / no_cluster_entries.txt")
    ap.add_argument("--mmcif-dir", type=Path,
                    default=Path("/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2026/mmCIF"))
    ap.add_argument("--clusters-file", default="data/clusters_30_2026.txt")
    ap.add_argument("--casp-csv", default="data/targetlist.csv")
    ap.add_argument("--cutoff-date", default="2024-04-30")
    ap.add_argument("--val-fraction", type=float, default=0.4)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"{args.out_dir} exists and is not empty; refusing to overwrite")
    old_dir = args.old_dir
    temporal_old = json.loads((old_dir / "mmcif_temporal_splits.json").read_text())
    final_old = json.loads((old_dir / "mmcif_final_splits.json").read_text())

    # GATE A: the archived final split is reproduced from the archived temporal one.
    rep, _, _ = cluster_pass(temporal_old, args.clusters_file, args.casp_csv)
    diff = [p for p in final_old if rep.get(p) != final_old[p]]
    if set(rep) != set(final_old) or diff:
        raise SystemExit(f"GATE A failed: {len(diff)} entries differ from the archived final split, e.g. {diff[:5]}")

    excl_old, excl_new = read_ids(old_dir / "no_cluster_entries.txt"), read_ids(args.new_exclude)
    added, removed = sorted(excl_old - excl_new), sorted(excl_new - excl_old)
    if set(added) & set(temporal_old):
        raise SystemExit("an added entry is already in the archived split")

    new_entries = {}
    for pid in added:
        path = args.mmcif_dir / pid[1:3] / f"{pid}.cif.gz"
        if not path.exists():
            raise SystemExit(f"mmCIF for added entry {pid} not found at {path}")
        new_entries[pid] = {"path": str(path)}
    cutoff = datetime.strptime(args.cutoff_date, "%Y-%m-%d").date()
    pds.TemporalSplitter().get_temporal_splits(new_entries, cutoff)
    for info in new_entries.values():
        info["temporal_split"] = info.get("split", "unknown")
    temporal_new = {p: i for p, i in temporal_old.items() if p not in set(removed)}
    temporal_new.update(new_entries)

    final_new, casp_stats, cluster_stats = cluster_pass(temporal_new, args.clusters_file, args.casp_csv)

    keys = ("split", "cluster_promoted", "casp16_test_set")
    changed = [{"entry": p, "before": {k: final_old[p].get(k) for k in keys},
                "after": {k: final_new[p].get(k) for k in keys}}
               for p in final_old if p in final_new and any(final_old[p].get(k) != final_new[p].get(k) for k in keys)]

    # Val/test: archived assignment kept; GATE B against bridging val and test.
    val_old, test_old = read_ids(old_dir / "val_holdout_ids.txt"), read_ids(old_dir / "test_ids.txt")
    eval_old, eval_new = evaluated(final_old), evaluated(final_new)
    if eval_old != val_old | test_old:
        raise SystemExit("archived val/test lists do not match the archived evaluated set")
    lost = sorted(eval_old - eval_new)
    if lost:
        raise SystemExit(f"GATE B failed: archived evaluated entries no longer evaluated: {lost[:10]}")
    casp = {p for p, i in final_new.items() if i.get("casp16_test_set")}
    groups = eval_groups(eval_new, args.clusters_file)
    val_add, test_add, decisions, conflicts = place_new_evaluated(groups, val_old, test_old, casp, args.val_fraction)
    if conflicts:
        raise SystemExit(f"GATE B failed: {len(conflicts)} groups would connect val and test, e.g. {conflicts[:2]}")
    for members in groups.values():  # archived groups must not have merged across sides
        if members & val_old and members & test_old:
            raise SystemExit(f"GATE B failed: archived val and test entries share a group: {sorted(members)[:10]}")
    val_new, test_new = val_old | val_add, test_old | test_add

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "mmcif_temporal_splits.json").write_text(json.dumps(temporal_new, indent=4))
    (out / "mmcif_final_splits.json").write_text(json.dumps(final_new, indent=4))
    (out / "casp16_split_stats.txt").write_text(json.dumps(casp_stats, indent=4))
    (out / "cluster_split_stats.txt").write_text(json.dumps(cluster_stats, indent=4))
    temporal_counts = defaultdict(int)
    for info in temporal_new.values():
        temporal_counts[info.get("temporal_split", "unknown")] += 1
    (out / "temporal_split_stats.txt").write_text(json.dumps(dict(temporal_counts), indent=4))
    pds.export_id_lists(final_new, out)
    (out / "val_holdout_ids.txt").write_text("".join(f"{p}\n" for p in sorted(val_new)))
    (out / "test_ids.txt").write_text("".join(f"{p}\n" for p in sorted(test_new)))
    for name in ("chain_clusters.tsv", "no_cluster_ids.txt", "no_cluster_entries.txt"):
        shutil.copy2(args.chain_clusters_dir / name, out / name)
    for name in ("casp16_missing_ids.txt",):
        if (old_dir / name).exists():
            shutil.copy2(old_dir / name, out / name)

    report = {
        "old_dir": str(old_dir), "new_exclude": str(args.new_exclude), "gate_a": "PASS", "gate_b": "PASS",
        "added_entries": {p: {"release_date": new_entries[p].get("release_date"),
                              "temporal_split": new_entries[p]["temporal_split"],
                              "final_split": final_new[p].get("split"),
                              "cluster_promoted": final_new[p].get("cluster_promoted", False),
                              "evaluated": p in eval_new} for p in added},
        "removed_entries": removed,
        "existing_entries_changed": changed,
        "evaluated": {"before": len(eval_old), "after": len(eval_new)},
        "val": {"before": len(val_old), "after": len(val_new)},
        "test": {"before": len(test_old), "after": len(test_new)},
        "new_evaluated_placement": decisions,
        "val_test_rule": "archived assignment kept; see module docstring",
    }
    (out / "extend_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("gate_a", "gate_b", "removed_entries", "evaluated", "val", "test")}, indent=2))
    print(f"added {len(added)}: " + ", ".join(f"{p}:{v['final_split']}{'(eval)' if v['evaluated'] else ''}"
                                             for p, v in report["added_entries"].items()))
    print(f"existing entries changed: {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
