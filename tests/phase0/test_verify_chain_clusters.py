"""scripts/verify_chain_clusters.py: the independent check of the cluster labels.

The label comparison and the RCSB parsing run here on plain data; the gemmi reader is
exercised on the real snapshot in the server job, and is replaced by a fixture only
where gemmi is absent. Pins: the 9fya case is caught (the old labels fail), correct
labels pass, and a chain missing from `_entity_poly` is compared as -1, not skipped.

Plain-python: `python tests/phase0/test_verify_chain_clusters.py`.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import verify_chain_clusters as vcc  # noqa: E402

# Real values (mmCIF via gemmi, clusters_30_2026): the case behind the 2026-11-05 fix.
BY_ENTRY = {
    "9fya": {"a": "1", "b": "1", "c": "1", "A": "2", "B": "2", "C": "2"},
    "9fye": {"A": "1", "B": "1", "C": "1", "a": "2", "b": "2", "c": "2"},
}
ENTITY_CLUSTERS = {"9FYA_1": 10569, "9FYE_2": 10569, "9FYA_2": 55552, "9FYE_1": 55552}


def test_the_old_case_folded_labels_are_caught():
    old = {"9fya_A": 55552, "9fya_a": 55552, "9fye_C": 10569, "9fye_c": 10569}
    rows, counts = vcc.compare(old, BY_ENTRY, ENTITY_CLUSTERS)
    assert {r["id"] for r in rows} == {"9fya_a", "9fye_C"}, rows
    assert counts["mismatch"] == 2


def test_correct_labels_pass():
    good = {"9fya_A": 55552, "9fya_a": 10569, "9fye_C": 55552, "9fye_c": 10569}
    rows, counts = vcc.compare(good, BY_ENTRY, ENTITY_CLUSTERS)
    assert rows == [] and counts["mismatch"] == 0 and counts["entity_in_cluster_file"] == 4


def test_unknown_chain_or_unclustered_entity_means_minus_one():
    labels = {"9fya_Z": -1, "1abc_A": -1, "1abc_B": 7}
    by_entry = {**BY_ENTRY, "1abc": {"A": "3", "B": "3"}}
    rows, counts = vcc.compare(labels, by_entry, ENTITY_CLUSTERS)
    assert counts["chain_not_in_entity_poly"] == 1 and counts["entity_not_in_cluster_file"] == 2
    assert [r["id"] for r in rows] == ["1abc_B"], "a label for an entity RCSB does not cluster is wrong"
    rows, _ = vcc.compare({"9fya_Z": 5}, BY_ENTRY, ENTITY_CLUSTERS)
    assert rows and rows[0]["kind"] == "chain_not_in_entity_poly", "unknown chain must not be skipped"


def test_cluster_file_ids_are_positions_among_non_empty_lines():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.txt"
        p.write_text("1ABC_1 2DEF_1\n\n3GHI_2\n")
        got = vcc.read_entity_clusters(p)
        assert got == {"1ABC_1": 0, "2DEF_1": 0, "3GHI_2": 1}, got


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
