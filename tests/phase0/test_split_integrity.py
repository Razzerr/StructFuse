"""Data-integrity invariants for the rebuilt pipeline.

These are the properties every leakage claim in the paper rests on. The checks
themselves live in `scripts/verify_data_integrity.py` so they can be run against
the real artifacts on the server; this file exercises them on synthetic fixtures
so they are verified on every commit, and re-runs them against the real files
when those happen to be present.

Plain-python: `python tests/phase0/test_split_integrity.py`.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_data_integrity.py"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _build_fixture(
    tmp: Path,
    *,
    index_cluster_overrides: dict | None = None,
    train_entries: str = "1aaa\n",
    index_extra: list | None = None,
) -> dict:
    """A minimal but complete artifact set: 3 entries, 1 chain each, 3 clusters."""
    chains = {"1aaa_A": 10, "2bbb_A": 20, "3ccc_A": 30, "4ddd_A": -1}
    rows = ["id\tcluster_id\tsource\tentity_id"]
    for stem, cid in chains.items():
        rows.append(f"{stem}\t{cid}\tentity\t{stem.split('_')[0]}_1")
    _write(tmp / "chain_clusters.tsv", "\n".join(rows) + "\n")
    _write(tmp / "no_cluster_ids.txt", "4ddd_A\n")
    _write(tmp / "corrupt_ids.txt", "")
    _write(tmp / "train.txt", train_entries)
    _write(tmp / "val.txt", "2bbb\n")
    _write(tmp / "test.txt", "3ccc\n")

    overrides = index_cluster_overrides or {}
    meta = [
        {"id": stem, "npz": f"{stem}.npz", "cluster_id": overrides.get(stem, cid)}
        for stem, cid in chains.items()
        if cid != -1
    ]
    meta.extend(index_extra or [])
    index_dir = tmp / "index"
    index_dir.mkdir(exist_ok=True)
    _write(index_dir / "ids.json", json.dumps(meta, indent=2))
    return {"index_dir": index_dir}


def _run(tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--index-dirs", str(tmp / "index"),
            "--chain-clusters-tsv", str(tmp / "chain_clusters.tsv"),
            "--no-cluster-ids", str(tmp / "no_cluster_ids.txt"),
            "--corrupt-ids", str(tmp / "corrupt_ids.txt"),
            "--train-ids", str(tmp / "train.txt"),
            "--val-ids", str(tmp / "val.txt"),
            "--test-ids", str(tmp / "test.txt"),
        ],
        capture_output=True,
        text=True,
    )


def test_clean_pipeline_passes_every_invariant():
    with tempfile.TemporaryDirectory() as tmp:
        _build_fixture(Path(tmp))
        result = _run(Path(tmp))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "All data-integrity checks passed" in result.stdout


def test_unclustered_chain_in_the_index_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        _build_fixture(
            Path(tmp),
            index_extra=[{"id": "4ddd_A", "npz": "4ddd_A.npz", "cluster_id": -1}],
        )
        result = _run(Path(tmp))
        assert result.returncode == 1, result.stdout
        assert "cluster_id >= 0" in result.stdout
        assert "no unclustered chain indexed" in result.stdout


def test_index_cluster_disagreeing_with_the_resolver_is_rejected():
    """ids.json must not drift from chain_clusters.tsv — that drift IS the bug class."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_fixture(Path(tmp), index_cluster_overrides={"3ccc_A": 999})
        result = _run(Path(tmp))
        assert result.returncode == 1, result.stdout
        assert "agrees with chain_clusters.tsv" in result.stdout


def test_chain_level_overlap_between_train_and_test_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        _build_fixture(Path(tmp), train_entries="1aaa\n3ccc\n")
        result = _run(Path(tmp))
        assert result.returncode == 1, result.stdout
        assert "train ∩ test == ∅ (chains)" in result.stdout


def test_cluster_level_overlap_is_rejected_even_without_shared_chains():
    """The check the split never had: distinct chains, same 30 %-identity cluster."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_fixture(root)
        # 1aaa_A (train) and 3ccc_A (test) are different chains in cluster 10.
        _write(root / "chain_clusters.tsv", "\n".join([
            "id\tcluster_id\tsource\tentity_id",
            "1aaa_A\t10\tentity\t1aaa_1",
            "2bbb_A\t20\tentity\t2bbb_1",
            "3ccc_A\t10\tentity\t3ccc_1",
            "4ddd_A\t-1\tentity_unclustered\t4ddd_1",
        ]) + "\n")
        _write(root / "index" / "ids.json", json.dumps([
            {"id": "1aaa_A", "npz": "1aaa_A.npz", "cluster_id": 10},
            {"id": "2bbb_A", "npz": "2bbb_A.npz", "cluster_id": 20},
            {"id": "3ccc_A", "npz": "3ccc_A.npz", "cluster_id": 10},
        ], indent=2))
        result = _run(root)
        assert result.returncode == 1, result.stdout
        assert "train ∩ test == ∅ (clusters)" in result.stdout
        assert "train ∩ test == ∅ (chains)" not in result.stdout.split("[FAIL]")[-1]


def test_no_cluster_ids_must_match_the_resolver_output():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_fixture(root)
        _write(root / "no_cluster_ids.txt", "")  # resolver says 4ddd_A is unclustered
        result = _run(root)
        assert result.returncode == 1, result.stdout
        assert "matches the -1 rows" in result.stdout


def test_real_artifacts_when_present():
    """Opportunistic: on the server this runs against the actual pipeline output."""
    tsv = ROOT / "data" / "output_splits" / "chain_clusters.tsv"
    if not tsv.exists():
        print("       (skipped — no real artifacts in this checkout)")
        return
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


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
