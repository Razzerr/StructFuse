"""The same-protein rule (src/data/utils/template_selection.py).

Plain-python: `python tests/phase0/test_template_selection.py`.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.utils.template_selection import (  # noqa: E402
    alignment_weights,
    CANDIDATE_MARGIN,
    is_same_protein,
    select_templates,
)

Q = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues


def _identity_map(n):
    return np.arange(n)


def _align(q, t):
    m = np.full(len(q), -1)
    n = min(len(q), len(t))
    m[:n] = np.arange(n)
    return m


def test_identity_threshold_is_inclusive_over_aligned_pairs():
    t95 = "W" * 5 + Q[5:]   # 95 of 100 identical
    t94 = "W" * 6 + Q[6:]   # the first six residues of Q contain no W
    assert is_same_protein(Q, t95, _identity_map(100))
    assert not is_same_protein(Q, t94, _identity_map(100))


def test_coverage_of_either_chain_is_enough():
    # Template is a fragment: 30 residues, fully aligned, 30 % of the query.
    frag = Q[:30]
    q2t = np.full(100, -1)
    q2t[:30] = np.arange(30)
    assert is_same_protein(Q, frag, q2t), "fragment inside the query: 100 % of the template"
    # Query is the fragment of a longer template.
    assert is_same_protein(frag, Q, _identity_map(30)), "query inside a longer template"
    # Neither chain covered: 50 of 100 aligned on both sides (a shared domain).
    q2t = np.full(100, -1)
    q2t[:50] = np.arange(50)
    assert not is_same_protein(Q, Q[:50] + "G" * 50, q2t), "a shared domain alone is not the same protein"


def test_unaligned_query_is_not_the_same_protein():
    assert not is_same_protein(Q, Q, np.full(100, -1))


def test_refused_chain_is_replaced_in_retrieval_order():
    seqs = {"same": Q, "a": "G" * 100, "b": "W" * 100}
    kept, refused = select_templates(lambda n: [("same", .99), ("a", .9), ("b", .8)], seqs.get, Q, 2, _align)
    assert [k[0] for k in kept] == ["a", "b"] and refused == 1, kept
    assert kept[0][1] == .9 and kept[0][2].shape == (100,)


def test_rule_off_keeps_identical_chains():
    kept, refused = select_templates(lambda n: [("same", .99)], {"same": Q}.get, Q, 1, _align, identity=None)
    assert [k[0] for k in kept] == ["same"] and refused == 0


def test_fetches_more_when_the_margin_is_used_up():
    pool = [(f"same{i}", 1.0) for i in range(200)] + [("other", 0.5)]
    seqs = {t: Q for t, _ in pool[:-1]}
    seqs["other"] = "G" * 100
    requests = []

    def fetch(n):
        requests.append(n)
        return pool[:n]

    calls = []

    def align(q, t):
        calls.append(t)
        return _align(q, t)

    kept, refused = select_templates(fetch, seqs.get, Q, 1, align)
    assert [k[0] for k in kept] == ["other"] and refused == 200, (kept, refused)
    assert requests[0] == 1 + CANDIDATE_MARGIN and len(requests) > 1, requests
    assert len(calls) == 201, f"each template is aligned once across fetches, got {len(calls)}"


def test_unreadable_template_is_skipped_and_short_supply_returns_what_exists():
    kept, _ = select_templates(lambda n: [("bad", .9), ("a", .8)], {"a": "G" * 100}.get, Q, 4, _align)
    assert [k[0] for k in kept] == ["a"], kept


def test_weights_count_identical_aligned_residues_and_sum_to_one():
    q = "ACDEFGHIKL"
    full = ("ACDEFGHIKL", np.arange(10))                          # 10 identical
    half = ("ACDEFWWWWW", np.arange(10))                          # 5 identical
    part = ("ACDEF", np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1]))  # 5, aligned over half
    w = alignment_weights(q, [full, half, part])
    assert np.allclose(w, [0.5, 0.25, 0.25]) and abs(w.sum() - 1) < 1e-6, w


def test_weights_are_uniform_when_nothing_matches():
    q = "AAAA"
    none = ("WWWW", np.arange(4))
    unaligned = ("WWWW", np.full(4, -1))
    assert np.allclose(alignment_weights(q, [none, unaligned]), [0.5, 0.5])


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
