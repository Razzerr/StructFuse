"""Phase 0 — B1 random_retrieval contract tests: O(K) sampling, stable-hash
determinism, same-protein/same-cluster exclusion, <k fallback. Plain-python:
`python tests/phase0/test_random_retrieval.py`.
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# faiss is imported at module top of src/models/utils/faiss.py but is only used in
# __init__ (which we bypass via __new__). Stub it if unavailable in env AI.
try:
    import faiss  # noqa: F401
except Exception:  # noqa: BLE001
    sys.modules["faiss"] = types.ModuleType("faiss")

from src.models.utils.faiss import FaissIndex, _get_protein_id  # noqa: E402
from scripts.rebuild_index_cluster_metadata import resolve_cluster  # noqa: E402


# A cluster id no fake index ever assigns, so a query carrying it collides with
# nothing and every template is admissible. NOT the empty set: since 2026-08-05
# an empty query-cluster set means "cluster unknown" and blocks everything.
NO_COLLISION = {10**9}


def _make_index(n, random_seed=0, clusters=None, holdout=None):
    idx = FaissIndex.__new__(FaissIndex)
    idx.row2id = [f"aa{i:05d}_A" for i in range(n)]
    # Default to a distinct known cluster per row. All -1 would now mean "every
    # template has an unknown cluster" and therefore block the whole pool.
    idx.row2cluster = clusters if clusters is not None else list(range(n))
    idx.chain2cluster = {
        chain_id: int(idx.row2cluster[i]) for i, chain_id in enumerate(idx.row2id)
    }
    idx.prot2clusters = {}
    idx.prot2cluster = {}
    for i, chain_id in enumerate(idx.row2id):
        cid = int(idx.row2cluster[i])
        if cid == -1:
            continue
        prot_id = _get_protein_id(chain_id)
        idx.prot2clusters.setdefault(prot_id, set()).add(cid)
        idx.prot2cluster.setdefault(prot_id, cid)
    idx.holdout_prot_ids = set(holdout or [])
    idx.random_seed = random_seed
    idx._row2prot = None
    idx._row2cluster_arr = None
    idx._pool_all = None
    idx._pool_nonholdout = None
    return idx


def test_returns_exactly_k_when_pool_large():
    idx = _make_index(50000)
    out = idx._random_topk("q1_A", "qprot", NO_COLLISION, 4, filter_holdout=False)
    assert len(out) == 4, len(out)
    ids = [r[0] for r in out]
    assert len(set(ids)) == 4, "duplicates returned"
    assert all(sim == 0.5 for _, sim in out)


def _emit_selection():
    """Build a fixed fake index (seed=7) and print one query's selection — used by
    the cross-PROCESS determinism test (separate interpreter ⇒ different builtin
    hash() randomization; blake2b must still produce identical output)."""
    idx = _make_index(20000, random_seed=7)
    out = idx._random_topk("chainX_A", "qprot", NO_COLLISION, 8, filter_holdout=False)
    print(",".join(r[0] for r in out))


def test_deterministic_across_processes():
    import subprocess
    runs = [
        subprocess.run([sys.executable, __file__, "--emit"], capture_output=True, text=True)
        for _ in range(2)
    ]
    outs = [r.stdout.strip() for r in runs]
    assert outs[0], f"empty emit output (stderr: {runs[0].stderr[-300:]})"
    assert outs[0] == outs[1], "stable-hash determinism violated ACROSS processes"
    # in-process: different seed -> different selection
    a = _make_index(20000, random_seed=7)._random_topk("chainX_A", "qprot", NO_COLLISION, 8, filter_holdout=False)
    c = _make_index(20000, random_seed=99)._random_topk("chainX_A", "qprot", NO_COLLISION, 8, filter_holdout=False)
    assert a != c, "different random_seed should change the selection"


def test_excludes_same_protein_and_same_cluster():
    n = 2000
    clusters = [5 if i % 2 == 0 else 3 for i in range(n)]  # half in cluster 5
    idx = _make_index(n, clusters=clusters)
    same_prot = _get_protein_id(idx.row2id[1])  # row 1 is cluster 3 (admissible cluster)
    out = idx._random_topk("q_A", same_prot, {5}, 20, filter_holdout=False)
    ids = {r[0] for r in out}
    assert idx.row2id[1] not in ids, "same-protein row not excluded"
    for rid in ids:
        row = idx.row2id.index(rid)
        assert idx.row2cluster[row] != 5, "same-cluster row not excluded"


def test_returns_all_when_fewer_than_k():
    # 6 rows: rows 0,1 in cluster 0 (admissible), rows 2-5 in cluster 5 (excluded).
    clusters = [0, 0, 5, 5, 5, 5]
    idx = _make_index(6, clusters=clusters)
    out = idx._random_topk("q_A", "qprot", {5}, 4, filter_holdout=False)  # query cluster 5
    assert len(out) == 2, f"expected all 2 admissible, got {len(out)}"  # no raise


def test_holdout_pool_excludes_holdout_proteins():
    n = 1000
    holdout = {_get_protein_id(f"aa{i:05d}_A") for i in range(100)}  # first 100 prots
    idx = _make_index(n, holdout=holdout)
    out = idx._random_topk("q_A", "qprot", NO_COLLISION, 10, filter_holdout=True)
    for rid, _ in out:
        assert _get_protein_id(rid) not in holdout, "holdout protein leaked"


def test_cluster_sets_filter_unknown_template_chain_by_entry_union():
    idx = FaissIndex.__new__(FaissIndex)
    idx.chain2cluster = {"query_A": -1, "tpl_H": -1}
    idx.prot2clusters = {"query": {737}, "tpl": {737}}
    assert idx._clusters_for_chain("query_A", "query") == {737}
    assert idx._same_cluster({737}, "tpl_H", -1), "template entry-level cluster fallback failed"


def test_known_chain_cluster_does_not_block_sibling_entities():
    """A chain with a known cluster is filtered on THAT cluster alone.

    Unioning sibling-entity clusters would block legitimate remote homologs of
    unrelated chains that merely share a crystal (and would inflate search_k).
    """
    idx = FaissIndex.__new__(FaissIndex)
    # PDB `q` holds two entities: chain A in cluster 737, chain B in cluster 999.
    idx.chain2cluster = {"q_A": 737, "q_B": 999, "t_X": 999, "t_Y": 737}
    idx.prot2clusters = {"q": {737, 999}, "t": {737, 999}}

    assert idx._clusters_for_chain("q_A", "q") == {737}, "sibling entity leaked into query set"
    assert idx._same_cluster({737}, "t_Y", 737), "true same-cluster template not blocked"
    assert not idx._same_cluster({737}, "t_X", 999), "sibling-entity cluster wrongly blocked"


def test_unknown_cluster_blocks_instead_of_admitting():
    """Unknown cluster on EITHER side must block (2026-08-05).

    RCSB omits some entries/entities from `clusters_30.txt`. The old predicate
    fell through to `set().intersection(...)` → falsy → admitted, which let
    identical-sequence templates from unclustered entries (e.g. 8AIQ, 8ZFJ)
    through: 3.29 % of best-retrieved templates, median identity 1.000.
    """
    idx = FaissIndex.__new__(FaissIndex)
    idx.chain2cluster = {"q_A": 737, "ghost_B": -1}
    # `ghost` is absent from clusters_30.txt entirely — no clustered sibling.
    idx.prot2clusters = {"q": {737}}

    assert idx._same_cluster({737}, "ghost_B", -1), "unclustered template was admitted"
    # A query whose own cluster is unknown cannot prove anything is admissible.
    assert idx._clusters_for_chain("ghost_B", "ghost") == set()
    assert idx._same_cluster(set(), "q_A", 737), "unknown query admitted a template"


def test_unknown_cluster_template_pool_is_empty_end_to_end():
    """The blocking policy reaches _random_topk, and it terminates cleanly."""
    idx = _make_index(500, clusters=[-1] * 500)  # nothing in the index is clustered
    out = idx._random_topk("q_A", "qprot", {42}, 4, filter_holdout=False)
    assert out == [], f"unclustered pool must yield no templates, got {len(out)}"


def test_cluster_metadata_resolution_prefers_mmcif_entity_over_exact_id():
    member2cluster = {
        "6xmx_1": 16149,
        "6xmx_2": 737,
        "af_test_1": 9,
    }
    assert resolve_cluster("6xmx_H", member2cluster, {"6xmx_h": "6xmx_2"}) == (
        737,
        "entity",
    )
    assert resolve_cluster("6xmx_1", member2cluster, {}) == (
        -1,
        "numeric_exact_suppressed",
    )
    assert resolve_cluster("AF_TEST_1", member2cluster, {}) == (9, "exact")


def _naive_scan_topk(idx, query_prot_id, query_cluster, k):
    """The OLD O(N)-per-query algorithm — full scan of row2id — as the speedup
    baseline. Mirrors the pre-Phase-0 behaviour."""
    valid = []
    for i, tpl_id in enumerate(idx.row2id):
        if _get_protein_id(tpl_id) == query_prot_id:
            continue
        if query_cluster != -1 and idx.row2cluster[i] == query_cluster:
            continue
        valid.append(i)
    return valid[:k]


def test_speedup_vs_naive_scan_at_least_20x():
    n, q = 20000, 100
    idx = _make_index(n)
    idx._ensure_random_pools()  # pre-build (mirrors the pre-fork eager build)
    t0 = time.perf_counter()
    for i in range(q):
        idx._random_topk(f"q{i}_A", "qprot", NO_COLLISION, 4, filter_holdout=False)
    t_new = time.perf_counter() - t0
    t1 = time.perf_counter()
    for i in range(q):
        _naive_scan_topk(idx, "qprot", -1, 4)
    t_naive = time.perf_counter() - t1
    speedup = t_naive / max(t_new, 1e-9)
    assert speedup >= 20.0, f"speedup only {speedup:.1f}× (new={t_new:.3f}s naive={t_naive:.3f}s)"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--emit" in sys.argv:
        _emit_selection()
        sys.exit(0)
    sys.exit(_run_all())
