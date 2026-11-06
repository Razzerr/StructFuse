"""Template distance histograms follow the atom of the contact labels.

Drives the real `PriorBuilder.build_one` on template NPZs written to disk; only the
FAISS lookup and the parasail alignments are replaced. Pins: a Cβ-generation NPZ
feeds `coords_cb` into the histogram, a Cα-generation NPZ keeps `coords`, a template
residue without coordinates contributes "unknown" rather than a distance to the
origin, an index pointing at mixed generations stops the run, and the old aligner
changes the features only, never the same-protein rule.

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
from src.data.utils.template_selection import SAME_PROTEIN_COVERAGE, SAME_PROTEIN_IDENTITY, alignment_weights  # noqa: E402
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
    b.alignment, b.template_weighting, b.per_template = "local", "alignment", False
    b.same_protein_identity, b.same_protein_coverage = SAME_PROTEIN_IDENTITY, SAME_PROTEIN_COVERAGE
    b._id_to_npz, b._tpl_cache, b._max_tpl_cache = paths, {}, 100
    return b


def _bins(builder) -> np.ndarray:
    return _build(builder)["dist_bins"]  # (9, L, L)


def _build(builder) -> dict:
    original = align.local_alignment
    align.local_alignment = lambda *a, **k: ("", "", np.arange(L), np.arange(L))
    try:
        return builder.build_one("q_A", "A" * L, 0, L)
    finally:
        align.local_alignment = original


def _build_with_both_aligners(builder, seq="A" * L):
    """Local maps i -> i, the old aligner i -> i-1; records who was called."""
    calls = []

    def local(q, t):
        calls.append("local")
        return "", "", np.arange(L), np.arange(L)

    def legacy(q, t):
        calls.append("legacy")
        return "", "", np.arange(L) - 1, np.arange(L)

    saved = align.local_alignment, align.needleman_wunsch
    align.local_alignment, align.needleman_wunsch = local, legacy
    try:
        return builder.build_one("q_A", seq, 0, L), calls
    finally:
        align.local_alignment, align.needleman_wunsch = saved


def test_local_alignment_is_the_only_aligner_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "1abc_A", "CB")
        out, calls = _build_with_both_aligners(_builder({"1abc_A": p}, [("1abc_A", 0.9)]))
        assert set(calls) == {"local"}, calls
        assert out["dist_bins"][1, 2, 20] > 0 and out["dist_bins"][1, 3, 21] == 0


def test_old_aligner_moves_the_features_and_the_rule_stays_local():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "1abc_A", "CB")
        b = _builder({"1abc_A": p}, [("1abc_A", 0.9)])
        b.alignment = "global_legacy"
        out, calls = _build_with_both_aligners(b)
        assert calls == ["local", "legacy"], calls
        assert out["dist_bins"][1, 3, 21] > 0 and out["dist_bins"][1, 2, 20] == 0
        # Identical template: the rule refuses it through the local map in either arm.
        out, calls = _build_with_both_aligners(_builder({"1abc_A": p}, [("1abc_A", 0.9)]), seq="C" * L)
        assert out["n_templates_retrieved"] == 0 and calls == ["local"], (out["n_templates_retrieved"], calls)
        b = _builder({"1abc_A": p}, [("1abc_A", 0.9)])
        b.alignment = "global_legacy"
        out, calls = _build_with_both_aligners(b, seq="C" * L)
        assert out["n_templates_retrieved"] == 0 and calls == ["local"], (out["n_templates_retrieved"], calls)


def test_unknown_alignment_name_is_refused():
    for kw, text in (({"alignment": "nw"}, "alignment must be one of"),
                     ({"template_weighting": "rank"}, "template_weighting must be one of")):
        try:
            PriorBuilder("unused", **kw)
        except ValueError as exc:
            assert text in str(exc)
        else:
            raise AssertionError(f"accepted {kw}")


def _two_templates(tmp):
    """Query "A"*L: 15 identical residues in the first, 5 in the second, same score.
    Residue 2 has no coordinates in the second, so that pair is "unknown" there and
    the histogram of pair (2, 20) shows the weight of the first template directly."""
    mask = np.ones(L, np.uint8)
    mask[2] = 0
    first = _write(Path(tmp), "1abc_A", "CB", seq="A" * 15 + "C" * 15)
    second = _write(Path(tmp), "2xyz_A", "CB", seq="A" * 5 + "C" * 25, mask=mask)
    return _builder({"1abc_A": first, "2xyz_A": second}, [("1abc_A", 0.9), ("2xyz_A", 0.9)])


def test_templates_are_weighted_by_identical_aligned_residues():
    with tempfile.TemporaryDirectory() as tmp:
        dist = _bins(_two_templates(tmp))
        assert abs(dist[1, 2, 20] - 0.75) < 1e-5 and abs(dist[0, 2, 20] - 0.25) < 1e-5, dist[:, 2, 20]


def test_cosine_weights_stay_available_and_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        b = _two_templates(tmp)
        b.template_weighting = "cosine"
        dist = _bins(b)
        assert abs(dist[1, 2, 20] - 0.5) < 1e-5 and abs(dist[0, 2, 20] - 0.5) < 1e-5, dist[:, 2, 20]


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
