"""retrieval_identity_audit.py refuses incomplete inputs instead of narrowing.

Reproduces the failure the author found: with one reference chain absent from
the index and no control file, the audit used to finish cleanly with n=1 and
NaN deltas. These tests drive the real ``main()`` with a fake index and tiny
NPZ files and pin the contract: the named control must exist, reference and
control populations must be identical, every reference chain must be indexed.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "retrieval_identity_audit.py"

spec = importlib.util.spec_from_file_location("ria_e2e", SCRIPT)
ria = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ria)


class _FakeIndex:
    """Stands in for LightweightFaissIndex: two indexed queries, one template."""

    def __init__(self, index_dir: Path):
        meta = json.loads((index_dir / "ids.json").read_text())
        self.meta = meta
        self.row2id = [m["id"] for m in meta]
        self.id2npz = {m["id"]: Path(m["npz"]) for m in meta}
        self.id2cluster = {m["id"]: int(m["cluster_id"]) for m in meta}
        self.chain2cluster = dict(self.id2cluster)

    def clusters_for_chain(self, chain_id, prot_id=None):
        c = self.chain2cluster.get(chain_id, -1)
        return {c} if c != -1 else set()

    def topk_precomputed(self, query_name, k, min_similarity=0.0):
        # the template is the only chain from a different protein and cluster
        return [("tpl_A", 0.97)]


def _fake_identity(q, t):
    return {"matches": 5, "aligned_pairs": 10, "query_residues": len(q), "template_residues": len(t),
            "seq_identity_aligned": 0.5, "seq_identity_query_len": 0.5,
            "query_coverage": 1.0, "template_coverage": 1.0}


def _write_npz(path: Path, seq: str):
    np.savez(path, seq=np.array(seq, dtype=object))


def _setup(tmp: Path, *, ref_ids, ctl_ids, indexed_ids, write_control=True):
    idx = tmp / "index"; idx.mkdir()
    (idx / "faiss.index").write_bytes(b"")
    meta = []
    for cid, name in enumerate(list(indexed_ids) + ["tpl_A"], start=1):
        npz = tmp / f"{name}.npz"; _write_npz(npz, "ACDEFGHIKL")
        meta.append({"id": name, "npz": str(npz), "cluster_id": 100 + cid})
    (idx / "ids.json").write_text(json.dumps(meta))

    def tsv(path, ids, val):
        lines = ["sample_id\tsubset\tcluster_id\tbest_tpl_sim\t" + "\t".join(ria.METRIC_COLUMNS)]
        for i, sid in enumerate(ids):
            lines.append(f"{sid}\tgold\t{100 + i + 1}\t0.97\t" + "\t".join(str(val) for _ in ria.METRIC_COLUMNS))
        path.write_text("\n".join(lines) + "\n")
    ref = tmp / "ref.tsv"; tsv(ref, ref_ids, 0.6)
    ctl = tmp / "ctl.tsv"
    if write_control:
        tsv(ctl, ctl_ids, 0.4)
    return idx, ref, ctl


def _run_main(tmp, idx, ref, ctl, control_arg=None):
    out = tmp / "out"
    argv = ["audit", "--index-dir", str(idx), "--data-root", str(tmp), "--out-dir", str(out),
            "--per-sample-tsv", str(ref), "--identity-scope", "crop", "--n-bootstrap", "0",
            "--allow-score-mismatch",
            "--control-per-sample-tsv", str(ctl) if control_arg is None else control_arg]
    old_argv, old_index, old_ident = sys.argv, ria.LightweightFaissIndex, ria.alignment_identity
    sys.argv, ria.LightweightFaissIndex, ria.alignment_identity = argv, _FakeIndex, _fake_identity
    try:
        ria.main()
    finally:
        sys.argv, ria.LightweightFaissIndex, ria.alignment_identity = old_argv, old_index, old_ident
    return out


def _expect(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


def test_the_reproduced_failure_now_raises():
    """Author's repro: two reference chains, one absent from the index, no
    control file. Must not finish with n=1 and NaN deltas."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A", "q2_A"], ctl_ids=[], indexed_ids=["q1_A"], write_control=False)
        _expect(FileNotFoundError, lambda: _run_main(tmp, idx, ref, ctl))


def test_missing_control_file_is_an_error_not_an_empty_control():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A"], ctl_ids=[], indexed_ids=["q1_A"], write_control=False)
        _expect(FileNotFoundError, lambda: _run_main(tmp, idx, ref, ctl))


def test_reference_chain_absent_from_index_is_an_error_not_a_smaller_n():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A", "q2_A"], ctl_ids=["q1_A", "q2_A"], indexed_ids=["q1_A"])
        _expect(KeyError, lambda: _run_main(tmp, idx, ref, ctl))


def test_control_population_must_equal_reference_population():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A", "q2_A"], ctl_ids=["q1_A"], indexed_ids=["q1_A", "q2_A"])
        _expect(ValueError, lambda: _run_main(tmp, idx, ref, ctl))


def test_explicit_empty_control_runs_a_descriptive_audit():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A", "q2_A"], ctl_ids=[], indexed_ids=["q1_A", "q2_A"], write_control=False)
        out = _run_main(tmp, idx, ref, ctl, control_arg="")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["populations"] == {"reference_chains": 2, "control_chains": 0, "audited_chains": 2, "unknown_query_cluster": 0}


def test_complete_inputs_audit_every_chain_with_finite_deltas():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        idx, ref, ctl = _setup(tmp, ref_ids=["q1_A", "q2_A"], ctl_ids=["q1_A", "q2_A"], indexed_ids=["q1_A", "q2_A"])
        out = _run_main(tmp, idx, ref, ctl)
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["populations"]["audited_chains"] == 2
        assert manifest["aggregation_unit"] == "cluster"
        glob_rows = (out / "global_summary.tsv").read_text().splitlines()
        header, allrow = glob_rows[0].split("\t"), glob_rows[1].split("\t")
        row = dict(zip(header, allrow))
        assert row["n"] == "2" and row["n_clusters"] == "2"
        assert abs(float(row["mean_delta_P@L_long"]) - 0.2) < 1e-9, row["mean_delta_P@L_long"]
        assert (out / "gain_by_crop_identity_bin.tsv").exists()
        assert not (out / "gain_by_full_identity_bin.tsv").exists()  # scope was crop only


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
