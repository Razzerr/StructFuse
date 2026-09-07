"""Does a template contact land on the query residue it actually aligns to?

The template prior is transferred through a Needleman-Wunsch alignment. A
one-residue error there displaces every transferred contact by one residue —
invisible at evaluation time, but it feeds the model a systematically wrong prior.

These tests supply the alignment EXPLICITLY (`query_to_template`) and hand-write
the expected query indices, so nothing is checked against the production mapping
that is itself under test. Insertions and deletions are covered because those are
where an off-by-one hides: with a pure 1:1 alignment a shifted implementation and
a correct one can agree.
"""
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# align.py imports parasail at module top for needleman_wunsch. These tests pass
# `query_to_template` explicitly, so no alignment is computed and the dependency
# is not exercised — stub it so the suite runs in the local env.
try:  # pragma: no cover
    import parasail  # noqa: F401
except ModuleNotFoundError:
    stub_p = types.ModuleType("parasail")
    stub_p.matrix_create = lambda *a, **k: None
    sys.modules["parasail"] = stub_p
try:  # pragma: no cover
    import blosum  # noqa: F401
except ModuleNotFoundError:
    # Only used for BLOSUM-weighted priors; every test here passes
    # use_blosum=False, so a substitution matrix of zeros is never consulted.
    stub = types.ModuleType("blosum")
    stub.BLOSUM = lambda *a, **k: defaultdict(lambda: defaultdict(int))
    sys.modules["blosum"] = stub

from src.data.utils.align import project_prior, project_distance  # noqa: E402


def _contact_map(n: int, pairs) -> np.ndarray:
    c = np.zeros((n, n), dtype=np.float32)
    for i, j in pairs:
        c[i, j] = c[j, i] = 1.0
    return c


def test_identity_alignment_transfers_a_contact_to_the_same_indices():
    n = 30
    tpl = _contact_map(n, [(2, 20)])
    a2t = np.arange(n)
    out = project_prior("A" * n, "A" * n, tpl, min_seq_sep=0,
                        use_blosum=False, query_to_template=a2t)
    assert out[2, 20] == 1 and out[20, 2] == 1, out[2, 20]
    assert (out == 1).sum() == 2, "exactly one symmetric transferred pair expected"


def test_deletion_in_the_query_shifts_the_target_indices():
    """Query is missing template residue 10, so template k>10 maps to query k-1.

    Template contact (5, 20) must land on query (5, 19) — NOT (5, 20).
    """
    n_t, n_q = 30, 29
    tpl = _contact_map(n_t, [(5, 20)])
    a2t = np.array([k if k < 10 else k + 1 for k in range(n_q)])
    assert a2t[19] == 20 and a2t[5] == 5
    out = project_prior("A" * n_q, "A" * n_t, tpl, min_seq_sep=0,
                        use_blosum=False, query_to_template=a2t)
    assert out[5, 19] == 1, f"expected the contact at (5,19), got {out[5, 19]}"
    assert out[5, 20] != 1, "contact landed at the unshifted index — off by one"
    assert (out == 1).sum() == 2, (out == 1).sum()


def test_insertion_in_the_query_shifts_indices_and_gaps_receive_nothing():
    """Query has 3 extra residues at 12..14 aligned to nothing (-1).

    Everything after the insertion shifts by 3, so both template contacts move,
    and the inserted positions themselves must receive no contact.
    """
    n_t, n_q = 30, 33
    tpl = _contact_map(n_t, [(5, 20), (13, 25)])
    a2t = np.array([k if k < 12 else (-1 if k < 15 else k - 3) for k in range(n_q)])
    assert a2t[12] == -1 and a2t[23] == 20 and a2t[5] == 5
    # template 13 and 25 are still reachable, from query 16 and 28
    assert a2t[16] == 13 and a2t[28] == 25
    out = project_prior("A" * n_q, "A" * n_t, tpl, min_seq_sep=0,
                        use_blosum=False, query_to_template=a2t)
    assert out[5, 23] == 1, f"(5,20)->(5,23) expected, got {out[5, 23]}"
    assert out[16, 28] == 1, f"(13,25)->(16,28) expected, got {out[16, 28]}"
    assert out[13, 25] != 1, "contact must not stay at the pre-insertion indices"
    # A query position aligned to a gap is "unknown" (-1 when use_blosum=False),
    # not "no contact" — so assert no CONTACT was transferred there, not that the
    # row sums to zero.
    gaps = np.flatnonzero(a2t == -1)
    assert (out[gaps] == 1).sum() == 0, "no contact may be transferred onto a gap"
    assert (out[:, gaps] == 1).sum() == 0, "same on the column axis"
    # two template contacts, each symmetric
    assert (out == 1).sum() == 4, (out == 1).sum()


def test_both_axes_are_mapped_not_just_the_row():
    """A row-only implementation would still place (5,20) somewhere in row 5."""
    n_t, n_q = 40, 38
    tpl = _contact_map(n_t, [(6, 30)])
    a2t = np.array([k if k < 15 else k + 2 for k in range(n_q)])
    assert a2t[6] == 6 and a2t[28] == 30
    out = project_prior("A" * n_q, "A" * n_t, tpl, min_seq_sep=0,
                        use_blosum=False, query_to_template=a2t)
    assert out[6, 28] == 1, f"column must be remapped too; got row6={np.flatnonzero(out[6])}"
    assert out[6, 30] != 1


def test_min_seq_sep_blanks_only_the_near_diagonal():
    n = 40
    tpl = _contact_map(n, [(10, 13), (10, 30)])
    a2t = np.arange(n)
    out = project_prior("A" * n, "A" * n, tpl, min_seq_sep=6,
                        use_blosum=False, query_to_template=a2t)
    assert out[10, 30] == 1, "a separated pair must survive"
    assert out[10, 13] != 1, "|i-j|=3 must be blanked by min_seq_sep=6"


def test_distance_projection_uses_the_same_index_mapping():
    """Same deletion as above, checked on distances rather than contacts."""
    n_t, n_q = 30, 29
    coords = np.zeros((n_t, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n_t) * 100.0     # far apart by default
    coords[20] = coords[5] + np.array([3.0, 0.0, 0.0])   # 3 A -> bin 1
    a2t = np.array([k if k < 10 else k + 1 for k in range(n_q)])
    bins = project_distance("A" * n_q, "A" * n_t, coords, a2t, min_seq_sep=0)
    assert bins.shape == (n_q, n_q)
    assert bins[5, 19] == 1, f"3 A must fall in the first finite bin, got {bins[5, 19]}"
    assert bins[5, 20] != 1, "distance landed at the unshifted index — off by one"
    assert bins[5, 19] == bins[19, 5], "distance bins must be symmetric"


def test_gap_positions_get_the_unknown_distance_bin():
    n_t, n_q = 30, 32
    coords = np.zeros((n_t, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n_t) * 5.0
    a2t = np.array([k if k < 10 else (-1 if k < 12 else k - 2) for k in range(n_q)])
    bins = project_distance("A" * n_q, "A" * n_t, coords, a2t, min_seq_sep=0)
    for gap in (10, 11):
        assert bins[gap].sum() == 0 and bins[:, gap].sum() == 0, (
            f"query {gap} aligns to a gap; every bin must stay 0 (unknown)")


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
