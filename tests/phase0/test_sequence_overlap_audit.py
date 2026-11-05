"""scripts/sequence_overlap_audit.py on chain NPZs written to disk.

Drives the real `main()`; only the production aligner is replaced where parasail is
absent (position-wise comparison, enough for the equal-length near matches used here).
Pins: population rule, exact and near matches, the two coverage rules, the evaluation
cap with the CASP16 exemption, cluster-balanced shares, case twins kept apart, and the
refusals (missing skip file, chain without a cluster).

Plain-python: `python tests/phase0/test_sequence_overlap_audit.py`.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import sequence_overlap_audit as soa  # noqa: E402

# Unrelated 40-residue sequences; RNG-free so failures are reproducible.
S1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
S2 = "GDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQT"
S3 = "PEDWCHYTNMRISLGAFKVEQPDWYCHTNMRLSGAKFVEI"
S4 = "WWYHCCPPNNDDGGRRTTEEKKSSQQAAVVLLIIMMFFWY"
S1_MUT = S1[:20] + "A" + S1[21:]   # one substitution: 97.5 % identity


def _aligner(q, t):
    """Position-wise stand-in for the production aligner (equal lengths in these tests)."""
    n = min(len(q), len(t))
    m = sum(a == b for a, b in zip(q, t))
    return {"seq_identity_aligned": m / n, "query_coverage": n / len(q), "template_coverage": n / len(t)}


def _fixture(tmp: Path, chains: dict, split: dict, clusters: dict, casp=()):
    proc, sp = tmp / "proc", tmp / "split"
    proc.mkdir()
    sp.mkdir()
    for stem, seq in chains.items():
        np.savez_compressed(proc / f"{stem}.npz", seq=np.array(seq, dtype=object))
    (proc / "npz_lengths.json").write_text(json.dumps({s: len(q) for s, q in chains.items()}))
    for name, entries in split.items():
        (sp / name).write_text("".join(f"{e}\n" for e in entries))
    (sp / "test_casp16_ids.txt").write_text("".join(f"{e}\n" for e in casp))
    (sp / "chain_clusters.tsv").write_text(
        "id\tcluster_id\n" + "".join(f"{s}\t{c}\n" for s, c in clusters.items()))
    return proc, sp


def _run(tmp, proc, sp, extra=()):
    argv, aligner = sys.argv, soa.alignment_identity
    sys.argv = ["x", "--processed-dir", str(proc), "--split-dir", str(sp), "--out-dir", str(tmp / "out"),
                "--workers", "1", "--kmer", "6", "--cap", "1", *extra]
    soa.alignment_identity = _aligner
    try:
        assert soa.main() == 0
    finally:
        sys.argv, soa.alignment_identity = argv, aligner
    return json.loads((tmp / "out" / "summary.json").read_text())


CHAINS = {
    "1aaa_A": S1, "1bbb_A": S3,                      # train
    "9aaa_A": S1,                                    # test, exact copy of a train chain, other cluster
    "9bbb_A": S1_MUT,                                # test, near copy
    "9ccc_A": S2,                                    # test, unrelated
    "9ccc_B": S2,                                    # same cluster as 9ccc_A, removed by the cap
    "8vvv_A": S3,                                    # val, exact copy
    "1sss_A": S1[:10],                               # too short, dropped
}
SPLIT = {"all_train_ids.txt": ["1aaa", "1bbb", "1sss"], "val_holdout_ids.txt": ["8vvv"],
         "test_ids.txt": ["9aaa", "9bbb", "9ccc"]}
CLUSTERS = {"1aaa_A": 1, "1bbb_A": 2, "9aaa_A": 3, "9bbb_A": 4, "9ccc_A": 5, "9ccc_B": 5, "8vvv_A": 6, "1sss_A": 7}


def test_exact_and_near_copies_are_found_and_shares_are_cluster_balanced():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc, sp = _fixture(tmp, CHAINS, SPLIT, CLUSTERS)
        s = _run(tmp, proc, sp)
        assert s["chains"] == {"train": 2, "val": 1, "test": 4}, s["chains"]  # short chain dropped
        t = s["populations"]["test"]
        assert t["exact"]["chains"] == 1 and t["same_protein_both_cov"]["chains"] == 2, t
        # Clusters 3, 4, 5: exact flags 1, 0, 0/0 -> 1/3 per cluster; chains 1/4.
        assert abs(t["exact"]["share_cluster_balanced"] - 1 / 3) < 1e-9, t["exact"]
        assert abs(t["exact"]["share_chains"] - 1 / 4) < 1e-9
        assert s["populations"]["val"]["exact"]["chains"] == 1
        assert s["train_chains_with_an_exact_eval_copy"] == {"chains": 2, "clusters": 2}
        rows = (tmp / "out" / "per_chain.tsv").read_text().splitlines()
        near = next(r for r in rows if r.startswith("9bbb_A")).split("\t")
        assert near[7] == "1aaa_A" and abs(float(near[8]) - 0.975) < 1e-9, near
        assert s["partial_smoke_run"] is False


def test_cap_keeps_first_sorted_stem_and_casp16_is_exempt():
    assert soa.apply_cap(["2b_A", "2a_B", "2a_A"], {"2a_A": 1, "2a_B": 1, "2b_A": 1}, 1, set()) == {"2a_A"}
    assert soa.apply_cap(["2a_A", "2a_B"], {"2a_A": 1, "2a_B": 1}, 1, {"2a_B"}) == {"2a_A", "2a_B"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc, sp = _fixture(tmp, CHAINS, SPLIT, CLUSTERS, casp=["9ccc"])
        s = _run(tmp, proc, sp)
        assert s["populations"]["test_capped"]["chains"] == 4, "both CASP16 chains stay"
        assert s["populations"]["casp16"]["chains"] == 2
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc, sp = _fixture(tmp, CHAINS, SPLIT, CLUSTERS)
        s = _run(tmp, proc, sp)
        assert s["populations"]["test_capped"]["chains"] == 3, "cap 1 drops 9ccc_B"


def test_coverage_rules_differ_on_a_fragment():
    rows = [{"stem": "a", "exact": False, "identity": 1.0, "query_cov": 1.0, "train_cov": 0.5}]
    s = soa.summarize(rows, {"a": 1}, 0.95, 0.9)
    assert s["same_protein_both_cov"]["chains"] == 0 and s["same_protein_either_cov"]["chains"] == 1, s
    rows = [{"stem": "a", "exact": False, "identity": float("nan"), "query_cov": float("nan"), "train_cov": float("nan")}]
    s = soa.summarize(rows, {"a": 1}, 0.95, 0.9)
    assert s["same_protein_either_cov"]["chains"] == 0, "no candidate is not a match"


def test_population_rule_keeps_case_twins_apart_and_applies_skip_lists():
    lengths = {"1abc_A": 30, "1abc_a": 30, "1ABC_B": 30, "1abc_C": 10, "2xyz_A": 30}
    got = soa.select_chains(lengths, {"1abc"}, {"1abc_a"}, 20)
    assert got == ["1ABC_B", "1abc_A"], got


def test_missing_skip_file_and_unclustered_chain_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc, sp = _fixture(tmp, CHAINS, SPLIT, CLUSTERS)
        try:
            _run(tmp, proc, sp, extra=["--skip-ids-files", str(tmp / "missing.txt")])
        except SystemExit as exc:
            assert "skip ids file not found" in str(exc)
        else:
            raise AssertionError("missing skip list accepted")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc, sp = _fixture(tmp, CHAINS, SPLIT, {**CLUSTERS, "9ccc_B": -1})
        try:
            _run(tmp, proc, sp)
        except SystemExit as exc:
            assert "without a cluster" in str(exc)
        else:
            raise AssertionError("unclustered chain accepted")


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
