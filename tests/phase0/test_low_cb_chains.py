"""Exclusion of Cα-only chains after the Cβ switch (scripts/list_low_cb_chains.py)
and chain-level skip lists in scripts/build_index.py.

Plain-python: `python tests/phase0/test_low_cb_chains.py`.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT / "scripts" / "list_low_cb_chains.py"


def _npz(d: Path, stem: str, src):
    n = 5 if src is None else len(src)
    kw = {"seq": np.array("A" * n), "L": np.int32(n)}
    if src is not None:
        kw["cb_source"] = np.array(src, dtype=np.uint8)
    np.savez_compressed(d / f"{stem}.npz", **kw)


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(LIST), *map(str, args)], capture_output=True, text=True)


def test_rule_counts_fallback_among_residues_with_a_ca():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _npz(d, "1abc_A", [1, 1, 2, 3, 1])           # 0 % fallback
        _npz(d, "1abc_B", [4, 4, 4, 1, 2])           # 60 % -> listed
        _npz(d, "1abc_C", [4, 4, 1, 2, 0, 0])        # 2 of 4 with Cα = 50 % -> not listed (strict >)
        _npz(d, "1abc_a", [4, 4, 4, 4, 0])           # case twin of 1abc_A, 100 % -> listed, A not
        (d / "npz_lengths.json").write_text(json.dumps({"1abc_A": 5, "1abc_B": 5, "1abc_C": 6, "1abc_a": 5}))
        r = _run("--processed-dir", d, "--out", d / "out" / "low.txt", "--workers", 1)
        assert r.returncode == 0, r.stderr
        assert (d / "out" / "low.txt").read_text().split() == ["1abc_B", "1abc_a"]
        meta = json.loads((d / "out" / "low.json").read_text())
        assert meta["chains_listed"] == 2 and meta["chains_with_any_fallback"] == 3, meta


def test_ca_generation_npz_is_an_error_not_a_pass():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _npz(d, "1abc_A", None)
        (d / "npz_lengths.json").write_text(json.dumps({"1abc_A": 5}))
        r = _run("--processed-dir", d, "--out", d / "low.txt", "--workers", 1)
        assert r.returncode != 0 and "cb_source" in r.stderr, r.stderr


def test_build_index_refuses_a_missing_skip_file():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "cc.tsv").write_text("id\tcluster_id\n1abc_A\t3\n")
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "build_index.py"), "--processed_dir", str(d),
                            "--out_dir", str(d / "idx"), "--exclude_ids", "", "--chain_clusters_tsv", str(d / "cc.tsv"),
                            "--skip_ids_files", str(d / "missing.txt")], capture_output=True, text=True)
        assert r.returncode != 0 and "skip ids file not found" in r.stderr, r.stderr[-500:]


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
