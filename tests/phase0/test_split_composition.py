"""split_composition.py: three units kept apart, read-only, filter-faithful.

What this pins: `promoted_train_to_test` is a PDB-ENTRY counter at split level,
which is why quoting it beside an evaluated-entry or chain count produced the
impossible "test 23,164 entries incl. 146,386 promoted". The holdout union is
the set `filter_holdout=True` actually blocks, normalised the way
`FaissIndex._get_protein_id` normalises it, with disjointness checked rather
than assumed. Chain counts must follow ContactDataset's own rule.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "split_composition.py"

spec = importlib.util.spec_from_file_location("sc", SCRIPT)
sc = importlib.util.module_from_spec(spec)
sys.modules["sc"] = sc
spec.loader.exec_module(sc)


def _fixture(tmp: Path, *, promoted_recorded=3):
    """2 train entries, 1 val, 2 test — of which 3 got there by promotion."""
    split_dir = tmp / "splits"; split_dir.mkdir()
    meta = {
        "1aaa": {"split": "train"},
        "1bbb": {"split": "train"},
        "2ccc": {"split": "val"},
        "3ddd": {"split": "test", "cluster_promoted": True},
        "3eee": {"split": "test", "cluster_promoted": True, "casp16_test_set": True},
        "3fff": {"split": "test", "cluster_promoted": True},
        "4ggg": {"split": "discarded"},
    }
    (split_dir / "mmcif_final_splits.json").write_text(json.dumps(meta))
    (split_dir / "cluster_split_stats.txt").write_text(json.dumps({
        "promoted_train_to_test": promoted_recorded,
        "clusters_with_test": 2, "clusters_train_only": 1,
    }))
    (split_dir / "all_train_ids.txt").write_text("1aaa\n1bbb\n")
    (split_dir / "val_holdout_ids.txt").write_text("2ccc\n")
    (split_dir / "test_ids.txt").write_text("3ddd\n3eee\n")   # 3fff not evaluated
    # 1bbb_B is short, 1aaa_C is a corrupt skip, 3fff_A is not in any id file
    (tmp / "npz_lengths.json").write_text(json.dumps({
        "1aaa_A": 300, "1aaa_C": 300, "1bbb_A": 300, "1bbb_B": 5,
        "2ccc_A": 300, "3ddd_A": 300, "3eee_A": 300, "3eee_B": 300, "3fff_A": 300,
    }))
    (tmp / "skip.txt").write_text("1aaa_C\n")
    (split_dir / "chain_clusters.tsv").write_text(
        "id\tcluster_id\n1aaa_A\t1\n1bbb_A\t1\n2ccc_A\t2\n"
        "3ddd_A\t3\n3eee_A\t3\n3eee_B\t4\n"
    )
    return split_dir


def _run(tmp: Path, split_dir: Path, extra=()):
    out = tmp / "out.tsv"
    cmd = [sys.executable, str(SCRIPT), "--split-dir", str(split_dir),
           "--out", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = {}
    for line in out.read_text().splitlines()[1:]:
        f = line.split("\t")
        rows[(f[0], f[1])] = dict(zip(("train", "val", "test", "discarded", "total"), f[2:]))
    return rows, r.stdout


def test_promotion_is_counted_in_pdb_entries_at_split_level():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        rows, _ = _run(tmp, sd)
        assert rows[("pdb_entry", "assigned to split")]["test"] == "3"
        assert rows[("pdb_entry", "of which cluster-promoted")]["test"] == "3"
        # 3fff is promoted at split level but absent from test_ids.txt: the two
        # populations are different sizes, which is the whole point.
        assert rows[("pdb_entry", "listed in the split id file")]["test"] == "2"


def test_archived_counter_is_reconciled_not_trusted():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _, stdout = _run(tmp, _fixture(tmp, promoted_recorded=3))
        assert "MATCH" in stdout
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _, stdout = _run(tmp, _fixture(tmp, promoted_recorded=999))
        assert "MISMATCH" in stdout, "a wrong archived counter must be reported"


def test_holdout_union_is_what_the_filter_blocks():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows, stdout = _run(tmp, _fixture(tmp))
        u = rows[("pdb_entry", "HOLDOUT UNION used by filter_holdout")]
        assert u["val"] == "1" and u["test"] == "2" and u["total"] == "3"
        assert "disjoint" in stdout


def test_overlapping_holdout_files_are_flagged_not_summed():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        (sd / "val_holdout_ids.txt").write_text("2ccc\n3ddd\n")   # 3ddd also in test
        rows, stdout = _run(tmp, sd)
        u = rows[("pdb_entry", "HOLDOUT UNION used by filter_holdout")]
        assert u["val"] == "2" and u["test"] == "2"
        assert u["total"] == "3", "union must not be the sum when the files overlap"
        assert "NOT disjoint" in stdout


def test_chain_counts_follow_the_dataset_rule():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        rows, _ = _run(tmp, sd, extra=("--npz-lengths", str(tmp / "npz_lengths.json"),
                                       "--skip-ids", str(tmp / "skip.txt")))
        raw = rows[("chain", "in npz_lengths, before filters")]
        kept = rows[("chain", "after min_len>=20 and skip-id files")]
        assert raw["train"] == "4" and kept["train"] == "2"   # 1bbb_B short, 1aaa_C skipped
        assert raw["test"] == "3" and kept["test"] == "3"     # 3fff_A not in test_ids
        assert kept["val"] == "1"


def test_cluster_counts_are_distinct_clusters_of_kept_chains():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        rows, _ = _run(tmp, sd, extra=("--npz-lengths", str(tmp / "npz_lengths.json"),
                                       "--skip-ids", str(tmp / "skip.txt")))
        cl = rows[("cluster", "distinct, among kept chains")]
        assert cl["train"] == "1"          # 1aaa_A and 1bbb_A share cluster 1
        assert cl["test"] == "2"           # clusters 3 and 4


def test_it_refuses_to_write_into_the_split_directory():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--split-dir", str(sd),
             "--out", str(sd / "x.tsv")], capture_output=True, text=True)
        assert r.returncode != 0 and "read-only" in (r.stdout + r.stderr)


def test_missing_inputs_are_noted_not_silently_zero():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); sd = _fixture(tmp)
        (sd / "chain_clusters.tsv").unlink()
        rows, stdout = _run(tmp, sd, extra=("--npz-lengths", str(tmp / "npz_lengths.json")))
        assert ("cluster", "distinct, among kept chains") not in rows
        assert "chain_clusters.tsv absent" in stdout


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
