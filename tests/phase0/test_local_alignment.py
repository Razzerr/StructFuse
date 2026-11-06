"""src/data/utils/align.py: `local_alignment` and the index maps it returns.

The mapping is checked on canned parasail results with hand-written expected indices
(gaps on both sides, an alignment that does not start at residue 0, nothing aligned).
The `real_` tests run the actual aligner, which the local env does not have; they are
skipped there, not passed, and run under `pytest tests/phase0` on the server.

Plain-python: `python tests/phase0/test_local_alignment.py`.
"""

import random
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import parasail  # noqa: F401
    HAS_PARASAIL = True
except ModuleNotFoundError:
    HAS_PARASAIL = False
    stub_p = types.ModuleType("parasail")
    stub_p.matrix_create = lambda *a, **k: None
    sys.modules["parasail"] = stub_p
try:
    import blosum  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("blosum")
    stub.BLOSUM = lambda *a, **k: defaultdict(lambda: defaultdict(int))
    sys.modules["blosum"] = stub

from src.data.utils import align  # noqa: E402
from src.data.utils.template_selection import is_same_protein  # noqa: E402


def _canned(query, ref, end_query, end_ref, score=30):
    result = types.SimpleNamespace(
        score=score, end_query=end_query, end_ref=end_ref,
        traceback=types.SimpleNamespace(query=query, ref=ref))
    saved = align.parasail
    align.parasail = types.SimpleNamespace(sw_trace_scan_16=lambda *a: result, blosum62=None)
    return saved


def test_segment_inside_both_chains_is_mapped_at_its_real_offsets():
    saved = _canned("ACDEFG", "ACDEFG", end_query=7, end_ref=8)
    try:
        _, _, q2t, t2q = align.local_alignment("WWACDEFGWW", "KKKACDEFGKK")
    finally:
        align.parasail = saved
    assert q2t.tolist() == [-1, -1, 3, 4, 5, 6, 7, 8, -1, -1], q2t
    assert t2q.tolist() == [-1, -1, -1, 2, 3, 4, 5, 6, 7, -1, -1], t2q


def test_gaps_on_both_sides_keep_the_offsets_right():
    # query:    A C - D E F
    # template: A C K D - F      query starts at 1, template at 2
    saved = _canned("AC-DEF", "ACKD-F", end_query=5, end_ref=6)
    try:
        _, _, q2t, t2q = align.local_alignment("WACDEF", "KKACKDF")
    finally:
        align.parasail = saved
    assert q2t.tolist() == [-1, 2, 3, 5, -1, 6], q2t
    assert t2q.tolist() == [-1, -1, 1, 2, -1, 3, 5], t2q


def test_nothing_aligned_when_no_segment_scores():
    saved = _canned("", "", end_query=-1, end_ref=-1, score=0)
    try:
        qa, ta, q2t, t2q = align.local_alignment("AAAA", "WWWWW")
    finally:
        align.parasail = saved
    assert qa == "" and ta == "" and (q2t == -1).all() and (t2q == -1).all() and q2t.size == 4 and t2q.size == 5


AA = "ACDEFGHIKLMNPQRSTVWY"


def _random_seq(rng, n):
    return "".join(rng.choice(AA) for _ in range(n))


def real_fragment_is_found_at_its_position():
    rng = random.Random(0)
    t = _random_seq(rng, 400)
    q = t[150:230]
    _, _, q2t, _ = align.local_alignment(q, t)
    assert q2t.tolist() == list(range(150, 230)), q2t[:5]
    assert is_same_protein(q, t, q2t), "a real fragment is the same protein"


def real_short_chain_does_not_scatter_over_a_long_unrelated_one():
    rng = random.Random(1)
    refused = 0
    for _ in range(50):
        q, t = _random_seq(rng, 25), _random_seq(rng, 1000)
        q2t = align.local_alignment(q, t)[2]
        refused += is_same_protein(q, t, q2t)
        # The global aligner the rule used to run on refused nearly all of these.
    assert refused == 0, refused


def real_aligned_letters_are_the_letters_of_the_sequences():
    rng = random.Random(2)
    t = _random_seq(rng, 300)
    q = _random_seq(rng, 40) + t[50:120] + "Y" + t[120:170] + _random_seq(rng, 30)
    qa, ta, q2t, t2q = align.local_alignment(q, t)
    aligned = np.flatnonzero(q2t >= 0)
    assert aligned.size >= 110, aligned.size
    pairs = [(a, b) for a, b in zip(qa, ta) if a != "-" and b != "-"]
    assert [q[i] for i in aligned] == [a for a, _ in pairs]
    assert [t[q2t[i]] for i in aligned] == [b for _, b in pairs]
    assert all(t2q[q2t[i]] == i for i in aligned)


def real_first_gap_position_costs_gap_open_and_the_next_gap_extend():
    p = sys.modules["parasail"]
    q = _random_seq(random.Random(3), 60)

    def score(a, b):
        return p.sw_trace_scan_16(a, b, align.GAP_OPEN, align.GAP_EXTEND, p.blosum62).score

    full = score(q, q)
    one = score(q[:30] + q[31:], q)  # residue 30 deleted: its own score is lost as well
    two = score(q[:30] + q[32:], q)
    assert full - one == score(q[30], q[30]) + align.GAP_OPEN, (full, one)
    assert one - two == score(q[31], q[31]) + align.GAP_EXTEND, (one, two)


for _name in [n for n in list(globals()) if n.startswith("real_")]:
    def _wrap(fn):
        def test():
            if not HAS_PARASAIL:
                raise unittest.SkipTest("parasail absent")
            fn()
        test.__name__ = f"test_{fn.__name__}"
        return test
    globals()[f"test_{_name}"] = _wrap(globals()[_name])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
