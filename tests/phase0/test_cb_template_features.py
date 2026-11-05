"""Template distance histograms follow the atom of the contact labels.

Drives the real `PriorBuilder.build_one` on template NPZs written to disk; only the
FAISS lookup and the parasail alignment are replaced. Pins: a Cβ-generation NPZ
feeds `coords_cb` into the histogram, a Cα-generation NPZ keeps `coords`, a template
residue without coordinates contributes "unknown" rather than a distance to the
origin, and an index pointing at mixed generations stops the run.

Plain-python: `python tests/phase0/test_cb_template_features.py`.
"""

import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import parasail  # noqa: F401
except ModuleNotFoundError:
    stub_p = types.ModuleType("parasail")
    stub_p.matrix_create = lambda *a, **k: None
    sys.modules["parasail"] = stub_p
try:
    import blosum  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("blosum")
    stub.BLOSUM = lambda *a, **k: defaultdict(lambda: defaultdict(int))
    sys.modules["blosum"] = stub

from src.data.components.dataset import PriorBuilder  # noqa: E402
from src.data.utils.template_selection import SAME_PROTEIN_COVERAGE, SAME_PROTEIN_IDENTITY  # noqa: E402
from src.data.utils import align  # noqa: E402

L = 30


def _coords():
    """Cα of residues 2 and 20 are 11 Å apart (bin 5, 10-12 Å), their Cβ 3 Å apart (bin 1)."""
    ca = np.zeros((L, 3), np.float32)
    ca[:, 0] = np.arange(L) * 100.0
    ca[2], ca[20] = (0, 0, 0), (11, 0, 0)
    cb = ca.copy()
    cb[2], cb[20] = (4, 0, 0), (7, 0, 0)
    return ca, cb


def _write(d: Path, stem: str, generation: str, mask=None, seq="C" * L) -> Path:
    ca, cb = _coords()
    kw = dict(seq=np.array(seq, dtype=object), coords=ca, contact=np.zeros((L, L), np.uint8),
              mask=np.ones(L, np.uint8) if mask is None else mask, L=np.int32(L))
    if generation == "CB":
        kw.update(coords_cb=cb, cb_source=np.ones(L, np.uint8), contact_atom="CB")
    path = d / f"{stem}.npz"
    np.savez_compressed(path, **kw)
    return path


def _builder(paths: dict, hits: list) -> PriorBuilder:
    b = PriorBuilder.__new__(PriorBuilder)
    b.faiss_index = types.SimpleNamespace(topk_precomputed=lambda *a, **k: hits)
    b.topk, b.use_blosum, b.only_positive_transfer = len(hits), False, False
    b.min_seq_sep, b.min_template_similarity, b.random_retrieval = 0, 0.0, False
    b.compute_dist_bins, b._needs_coords, b._tpl_atom = True, True, None
    b.same_protein_identity, b.same_protein_coverage = SAME_PROTEIN_IDENTITY, SAME_PROTEIN_COVERAGE
    b._id_to_npz, b._tpl_cache, b._max_tpl_cache = paths, {}, 100
    return b


def _bins(builder) -> np.ndarray:
    return _build(builder)["dist_bins"]  # (9, L, L)


def _build(builder) -> dict:
    original = align.needleman_wunsch
    align.needleman_wunsch = lambda *a, **k: ("", "", np.arange(L), np.arange(L))
    try:
        return builder.build_one("q_A", "A" * L, 0, L)
    finally:
        align.needleman_wunsch = original


def test_cb_generation_histogram_reads_cb():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "1abc_A", "CB")
        dist = _bins(_builder({"1abc_A": p}, [("1abc_A", 0.9)]))
        assert dist[1, 2, 20] > 0.99, dist[:, 2, 20]  # bin 1 (0-4 Å): the Cβ distance


def test_ca_generation_histogram_still_reads_ca():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "1abc_A", "CA")
        dist = _bins(_builder({"1abc_A": p}, [("1abc_A", 0.9)]))
        assert dist[1, 2, 20] < 0.01 and dist[5, 2, 20] > 0.99, dist[:, 2, 20]  # bin 5: 10-12 Å


def test_template_residue_without_coordinates_is_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        mask = np.ones(L, np.uint8)
        mask[20] = 0
        p = _write(Path(tmp), "1abc_A", "CB", mask=mask)
        dist = _bins(_builder({"1abc_A": p}, [("1abc_A", 0.9)]))
        assert dist[0, 2, 20] > 0.99 and dist[1:, 2, 20].sum() < 0.01, dist[:, 2, 20]
        assert dist[0, 20, :].min() > 0.99, "every pair touching the residue stays unknown"


def test_same_protein_template_is_refused_and_the_next_one_is_used():
    """Query is all alanine: the first hit is identical, the second unrelated."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = {"1abc_A": _write(d, "1abc_A", "CB", seq="A" * L), "2xyz_A": _write(d, "2xyz_A", "CB", seq="C" * L)}
        fetched = []

        def fetch(pid, n, **kw):
            fetched.append(n)
            return [("1abc_A", 0.99), ("2xyz_A", 0.8)]

        b = _builder(paths, [])
        b.topk = 1
        b.faiss_index = types.SimpleNamespace(topk_precomputed=fetch)
        out = _build(b)
        assert out["n_templates_retrieved"] == 1, out["n_templates_retrieved"]
        assert abs(out["best_tpl_sim"] - 0.8) < 1e-6, "the refused chain must not set the template stats"
        assert fetched[0] > 1, "must fetch beyond k so a refused template can be replaced"


def test_only_same_protein_candidates_leave_no_template():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        out = _build(_builder({"1abc_A": _write(d, "1abc_A", "CB", seq="A" * L)}, [("1abc_A", 0.99)]))
        assert out["n_templates_retrieved"] == 0 and out["dist_bins"].sum() == 0


def test_mixed_generations_stop_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = {"1abc_A": _write(d, "1abc_A", "CB"), "2xyz_A": _write(d, "2xyz_A", "CA")}
        builder = _builder(paths, [("1abc_A", 0.9), ("2xyz_A", 0.8)])
        try:
            _bins(builder)
        except RuntimeError as exc:
            assert "mixed data generations" in str(exc)
        else:
            raise AssertionError("mixed Cα/Cβ templates were accepted")


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
