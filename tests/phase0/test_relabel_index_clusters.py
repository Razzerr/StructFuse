"""Audit copy of an index with corrected cluster labels (scripts/relabel_index_clusters.py).

The copy must keep the rows and vectors (symlinks), look labels up by the EXACT,
case-sensitive chain id (the whole point of the 2026-11-05 fix), refuse rows it
cannot label, and never overwrite an existing directory.

Plain-python: `python tests/phase0/test_relabel_index_clusters.py`.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.relabel_index_clusters import read_chain_clusters, relabel  # noqa: E402

SCRIPT = ROOT / "scripts" / "relabel_index_clusters.py"
META = [
    {"id": "9fya_A", "seq_len": 188, "npz": "x/9fya_A.npz", "cluster_id": 55552},
    {"id": "9fya_a", "seq_len": 133, "npz": "x/9fya_a.npz", "cluster_id": 55552},
    {"id": "9fye_C", "seq_len": 187, "npz": "x/9fye_C.npz", "cluster_id": 10569},
]
TSV = ("id\tcluster_id\tsource\tentity_id\n"
       "9fya_A\t55552\tentity\t9fya_2\n"
       "9fya_a\t10569\tentity\t9fya_1\n"
       "9fye_C\t55552\tentity\t9fye_1\n")


def _index(tmp: Path) -> Path:
    src = tmp / "index"
    src.mkdir()
    (src / "ids.json").write_text(json.dumps(META))
    (src / "faiss.index").write_bytes(b"F")
    (src / "embeddings.npy").write_bytes(b"E")
    (tmp / "cc.tsv").write_text(TSV)
    return src


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_labels_follow_exact_chain_case():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _index(tmp)
        new, stats = relabel(META, read_chain_clusters(tmp / "cc.tsv"))
        got = {m["id"]: m["cluster_id"] for m in new}
        assert got == {"9fya_A": 55552, "9fya_a": 10569, "9fye_C": 55552}, got
        assert stats["changed"] == 2 and stats["now_unclustered"] == 0, stats
        assert [m["npz"] for m in new] == [m["npz"] for m in META]  # rows untouched


def test_copy_symlinks_vectors_and_records_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _index(tmp)
        out = tmp / "relabel"
        r = _run("--index-dir", src, "--chain-clusters", tmp / "cc.tsv", "--out-dir", out)
        assert r.returncode == 0, r.stderr
        for name in ("faiss.index", "embeddings.npy"):
            assert (out / name).is_symlink() and os.path.realpath(out / name) == str((src / name).resolve())
        manifest = json.loads((out / "relabel_manifest.json").read_text())
        assert manifest["rows"] == 3 and manifest["changed"] == 2, manifest
        assert json.loads((src / "ids.json").read_text()) == META  # source untouched


def test_missing_row_and_nonempty_out_dir_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _index(tmp)
        (tmp / "short.tsv").write_text("\n".join(TSV.splitlines()[:3]) + "\n")  # no 9fye_C
        r = _run("--index-dir", src, "--chain-clusters", tmp / "short.tsv", "--out-dir", tmp / "a")
        assert r.returncode != 0 and "absent" in (r.stderr + r.stdout), r.stderr
        busy = tmp / "busy"
        busy.mkdir()
        (busy / "keep").write_text("x")
        r = _run("--index-dir", src, "--chain-clusters", tmp / "cc.tsv", "--out-dir", busy)
        assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout), r.stderr


def test_unclustered_rows_are_kept_and_counted():
    meta = [{"id": "1abc_A", "cluster_id": 3}]
    new, stats = relabel(meta, {"1abc_A": -1})
    assert new[0]["cluster_id"] == -1 and stats["now_unclustered_ids"] == ["1abc_A"], stats


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
