"""scripts/extend_splits.py: add entries to an archived split without redrawing it.

Pins: GATE A (the archived final split must be reproduced from the archived
temporal one), dating added entries with the production splitter, the
production cluster promotion reaching EXISTING entries, archived val/test kept,
a new evaluated entry following its cluster-connected archived entries, and
GATE B refusing a group that would connect val and test.

Plain-python: `python tests/phase0/test_extend_splits.py` (gemmi stubbed if absent).
"""

import json
import sys
import tempfile
import types
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import gemmi  # noqa: F401
except ImportError:
    sys.modules["gemmi"] = types.ModuleType("gemmi")

from scripts import extend_splits as es  # noqa: E402
from scripts import prepare_data_splits as pds  # noqa: E402

CUTOFF = date(2024, 4, 30)
DATES = {"2xxx": date(2020, 1, 1), "9new": date(2025, 1, 1), "9joi": date(2025, 2, 1), "9brg": date(2025, 3, 1)}


def _fake_release(self, path):
    pid = Path(path).name[:4]
    return {"release_date": DATES[pid], "release_source": "fake", "status_code": "REL", "deposition_date": None}


def _fixture(tmp: Path, clusters: str, added: list):
    old, mm = tmp / "old", tmp / "mmCIF"
    old.mkdir()
    (tmp / "clusters.txt").write_text(clusters)
    (tmp / "casp.csv").write_text("Target;Description\nT1;target 9ccc\n")
    temporal = {
        "1aaa": {"path": "x", "split": "train", "temporal_split": "train", "release_date": "2019-01-01"},
        "1bbb": {"path": "x", "split": "train", "temporal_split": "train", "release_date": "2019-01-01"},
        "9aaa": {"path": "x", "split": "test", "temporal_split": "test", "release_date": "2025-01-01"},
        "9bbb": {"path": "x", "split": "test", "temporal_split": "test", "release_date": "2025-01-01"},
        "9ccc": {"path": "x", "split": "test", "temporal_split": "test", "release_date": "2025-01-01"},
    }
    final, _, _ = es.cluster_pass(temporal, str(tmp / "clusters.txt"), str(tmp / "casp.csv"))
    (old / "mmcif_temporal_splits.json").write_text(json.dumps(temporal))
    (old / "mmcif_final_splits.json").write_text(json.dumps(final))
    (old / "val_holdout_ids.txt").write_text("9aaa\n")
    (old / "test_ids.txt").write_text("9bbb\n9ccc\n")
    (old / "no_cluster_entries.txt").write_text("".join(f"{p}\n" for p in added) + "0zzz\n")
    cc = tmp / "cc"
    cc.mkdir()
    for name in ("chain_clusters.tsv", "no_cluster_ids.txt"):
        (cc / name).write_text("id\tcluster_id\n" if name.endswith("tsv") else "")
    (cc / "no_cluster_entries.txt").write_text("0zzz\n")
    for p in added:
        (mm / p[1:3]).mkdir(parents=True, exist_ok=True)
        (mm / p[1:3] / f"{p}.cif.gz").write_bytes(b"")
    return old, mm, cc


def _run(tmp: Path, old: Path, mm: Path, cc: Path):
    argv = sys.argv
    sys.argv = ["extend_splits", "--old-dir", str(old), "--new-exclude", str(cc / "no_cluster_entries.txt"),
                "--chain-clusters-dir", str(cc), "--mmcif-dir", str(mm), "--clusters-file", str(tmp / "clusters.txt"),
                "--casp-csv", str(tmp / "casp.csv"), "--out-dir", str(tmp / "out")]
    original = pds.TemporalSplitter.get_release_date
    pds.TemporalSplitter.get_release_date = _fake_release
    try:
        return es.main()
    finally:
        sys.argv = argv
        pds.TemporalSplitter.get_release_date = original


CLUSTERS = ("1AAA_1 9NEW_1\n"      # added test entry promotes an EXISTING train entry
            "1BBB_1\n"
            "9AAA_1 9JOI_2\n"      # added entry joins archived val
            "9BBB_1 2XXX_1\n"      # added pre-cutoff entry is promoted itself
            "9CCC_1\n")


def test_entries_are_added_with_the_production_rules_and_val_test_kept():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old, mm, cc = _fixture(tmp, CLUSTERS, ["2xxx", "9new", "9joi"])
        assert _run(tmp, old, mm, cc) == 0
        out = tmp / "out"
        final = json.loads((out / "mmcif_final_splits.json").read_text())
        rep = json.loads((out / "extend_report.json").read_text())
        assert final["2xxx"]["split"] == "test" and final["2xxx"].get("cluster_promoted"), final["2xxx"]
        assert final["9new"]["split"] == "test" and not final["9new"].get("cluster_promoted")
        assert final["1aaa"]["split"] == "test" and final["1aaa"].get("cluster_promoted")  # existing entry moved
        assert [c["entry"] for c in rep["existing_entries_changed"]] == ["1aaa"], rep["existing_entries_changed"]
        val = set((out / "val_holdout_ids.txt").read_text().split())
        test = set((out / "test_ids.txt").read_text().split())
        assert {"9aaa", "9joi"} <= val and {"9bbb", "9ccc"} <= test, (val, test)
        assert "9new" in val | test and not val & test
        assert "2xxx" not in val | test  # promoted entries are not evaluated
        assert set((out / "all_train_ids.txt").read_text().split()) == {"1bbb"}


def test_gate_a_refuses_a_split_it_cannot_reproduce():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old, mm, cc = _fixture(tmp, CLUSTERS, ["2xxx"])
        final = json.loads((old / "mmcif_final_splits.json").read_text())
        final["1bbb"]["split"] = "test"
        (old / "mmcif_final_splits.json").write_text(json.dumps(final))
        try:
            _run(tmp, old, mm, cc)
        except SystemExit as exc:
            assert "GATE A" in str(exc)
        else:
            raise AssertionError("tampered archive accepted")


def test_gate_b_refuses_a_new_entry_that_bridges_val_and_test():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old, mm, cc = _fixture(tmp, CLUSTERS + "9AAA_2 9BRG_1\n9BBB_2 9BRG_2\n", ["9brg"])
        try:
            _run(tmp, old, mm, cc)
        except SystemExit as exc:
            assert "GATE B" in str(exc)
        else:
            raise AssertionError("val/test bridge accepted")


def test_hash_rule_is_order_free():
    assert es.hash_fraction(["9b", "9a"]) == es.hash_fraction(["9a", "9b"])
    assert 0.0 <= es.hash_fraction(["x"]) < 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
