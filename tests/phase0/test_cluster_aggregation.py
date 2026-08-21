"""Headline metrics aggregate over sequence clusters, not chains.

A mean over chains weights each protein family by how often it was deposited.
These tests pin the estimator: within-cluster mean first, unweighted mean over
clusters, NaN-aware at both levels, with the per-chain macro preserved beside it
so nothing changes meaning under an existing key.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.utils.metrics import (  # noqa: E402
    chain_macro,
    cluster_macro,
    log_macro_test_metrics,
)

NAN = float("nan")




def rows(*spec):
    """spec: (cluster_id, value) pairs -> per-sample rows."""
    return [
        {"cluster_id": c, "P@L_long": v, "n_valid_long_pairs": 1, "subset": "gold"}
        for c, v in spec
    ]


def test_one_big_family_cannot_outvote_many_small_ones():
    # 9 chains of a single family at 0.9, one chain each from 3 families at 0.1
    rs = rows(*([(1, 0.9)] * 9 + [(2, 0.1), (3, 0.1), (4, 0.1)]))
    chain, n_chain = chain_macro(rs, "P@L_long")
    clust, n_clust = cluster_macro(rs, "P@L_long")
    assert n_chain == 12 and n_clust == 4
    assert abs(chain - 0.7) < 1e-9, chain          # family-weighted
    assert abs(clust - 0.3) < 1e-9, clust          # family-balanced
    assert clust < chain, "the redundant family must lose its extra votes"


def test_equals_chain_macro_when_every_cluster_is_a_singleton():
    rs = rows((1, 0.2), (2, 0.4), (3, 0.6))
    assert abs(cluster_macro(rs, "P@L_long")[0]
               - chain_macro(rs, "P@L_long")[0]) < 1e-12


def test_within_cluster_mean_then_outer_mean():
    rs = rows((1, 0.0), (1, 1.0), (2, 0.5))
    v, n = cluster_macro(rs, "P@L_long")
    assert n == 2 and abs(v - 0.5) < 1e-12, (v, n)


def test_nan_chain_drops_from_its_cluster_but_keeps_the_cluster():
    rs = rows((1, NAN), (1, 0.8), (2, 0.2))
    v, n = cluster_macro(rs, "P@L_long")
    assert n == 2 and abs(v - 0.5) < 1e-12, (v, n)


def test_all_nan_cluster_drops_out_entirely():
    rs = rows((1, NAN), (1, NAN), (2, 0.4))
    v, n = cluster_macro(rs, "P@L_long")
    assert n == 1 and abs(v - 0.4) < 1e-12, (v, n)


def test_unknown_cluster_is_excluded_not_pooled():
    # -1 rows must not silently form one giant pseudo-cluster
    rs = rows((-1, 0.9), (-1, 0.9), (1, 0.1))
    v, n = cluster_macro(rs, "P@L_long")
    assert n == 1 and abs(v - 0.1) < 1e-12, (v, n)


def test_no_clusters_at_all_reports_nan_and_zero():
    v, n = cluster_macro(rows((-1, 0.5)), "P@L_long")
    assert n == 0 and v != v, (v, n)


def _logged(rs):
    out = {}

    def log_fn(key, value, **_):
        out[key] = value

    log_macro_test_metrics(log_fn, rs, ["gold"])
    return out


def test_canonical_key_is_cluster_balanced_and_chainmacro_is_kept():
    rs = rows(*([(1, 0.9)] * 9 + [(2, 0.1), (3, 0.1), (4, 0.1)]))
    out = _logged(rs)
    assert abs(out["test/P@L_long"] - 0.3) < 1e-9, out["test/P@L_long"]
    assert abs(out["test/P@L_long_chainmacro"] - 0.7) < 1e-9
    assert out["test/n_clusters"] == 4.0
    assert out["test/cluster_balanced"] == 1.0
    assert out["test/n_proteins"] == 12.0


def test_fallback_is_flagged_when_cluster_ids_are_missing():
    rs = rows((-1, 0.4), (-1, 0.6))
    out = _logged(rs)
    assert out["test/cluster_balanced"] == 0.0
    assert out["test/n_clusters"] == 0.0
    # canonical key falls back to the chain macro rather than NaN
    assert abs(out["test/P@L_long"] - 0.5) < 1e-12, out["test/P@L_long"]


def test_subsets_are_cluster_balanced_too():
    rs = rows(*([(1, 0.9)] * 9 + [(2, 0.1), (3, 0.1), (4, 0.1)]))
    out = _logged(rs)
    assert abs(out["test/gold/P@L_long"] - 0.3) < 1e-9
    assert abs(out["test/gold/P@L_long_chainmacro"] - 0.7) < 1e-9
    assert out["test/gold/n_clusters"] == 4.0

def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
    sys.exit(_run_all())
