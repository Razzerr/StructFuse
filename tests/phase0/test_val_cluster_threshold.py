"""Cluster-balanced validation threshold selection (Methods 4.11).

`val/f1_long`, the selected threshold and therefore checkpoint selection must use
the SAME estimand as the headline test metrics: per-chain F1 -> mean within
cluster -> unweighted mean over clusters. The pooled (micro) statistic that used
to drive selection is kept only under `val/*_micro`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from src.models.utils.metrics import (  # noqa: E402
    cluster_macro_curves,
    per_chain_range_counts,
)


def _counts(chains, T):
    """chains: list of (tp_row, fp_row, fn_row) each length T."""
    tp = np.array([c[0] for c in chains], dtype=float)
    fp = np.array([c[1] for c in chains], dtype=float)
    fn = np.array([c[2] for c in chains], dtype=float)
    return tp, fp, fn


def test_matches_hand_computed_cluster_mean():
    # 3 chains, 1 threshold. Cluster 7 has two chains, cluster 9 has one.
    tp, fp, fn = _counts([([10],[0],[0]), ([0],[10],[10]), ([5],[5],[5])], 1)
    f1, pr, rc, n = cluster_macro_curves([7, 7, 9], tp, fp, fn)
    # chain F1s: 1.0, 0.0, 0.5 -> cluster 7 = 0.5, cluster 9 = 0.5 -> mean 0.5
    assert n == 2, n
    assert abs(f1[0] - 0.5) < 1e-6, f1
    assert abs(pr[0] - (0.5 * (1.0 + 0.0) + 0.5) / 2) < 1e-6, pr


def test_big_cluster_does_not_dominate():
    # 100 chains in one cluster all perfect; 1 chain in another cluster at zero.
    T = 1
    chains = [([10],[0],[0])] * 100 + [([0],[10],[10])]
    tp, fp, fn = _counts(chains, T)
    cids = [1] * 100 + [2]
    f1, _, _, n = cluster_macro_curves(cids, tp, fp, fn)
    assert n == 2, n
    # cluster-balanced: (1.0 + 0.0)/2 = 0.5, NOT 100/101 = 0.99
    assert abs(f1[0] - 0.5) < 1e-6, f1
    chain_macro = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean()
    assert chain_macro > 0.98, chain_macro


def test_macro_and_micro_select_different_thresholds():
    """The whole point of the change: pooled and cluster-balanced curves can
    disagree about the best threshold, and selection must follow the reported
    estimand."""
    # threshold A favours one huge chain; threshold B favours many small ones.
    # cluster 1: one big chain.  cluster 2..6: five small chains.
    big = ([1000, 400], [0, 0], [0, 600])          # A: F1=1.0   B: F1~0.57
    small = ([1, 10], [9, 0], [9, 0])              # A: F1=0.1   B: F1=1.0
    chains = [big] + [small] * 5
    cids = [1] + [2, 3, 4, 5, 6]
    tp, fp, fn = _counts(chains, 2)

    f1_macro, _, _, n = cluster_macro_curves(cids, tp, fp, fn)
    assert n == 6, n
    f1_micro = 2 * tp.sum(0) / (2 * tp.sum(0) + fp.sum(0) + fn.sum(0) + 1e-8)

    assert int(np.argmax(f1_micro)) == 0, f1_micro   # pooled prefers threshold A
    assert int(np.argmax(f1_macro)) == 1, f1_macro   # families prefer threshold B


def test_unknown_cluster_rows_excluded_not_pooled():
    tp, fp, fn = _counts([([10],[0],[0]), ([0],[10],[10]), ([0],[10],[10])], 1)
    f1, _, _, n = cluster_macro_curves([5, -1, -1], tp, fp, fn)
    # only the cluster-5 chain survives; the two -1 rows must NOT become a cluster
    assert n == 1, n
    assert abs(f1[0] - 1.0) < 1e-6, f1


def test_returns_none_when_no_cluster_is_known():
    tp, fp, fn = _counts([([1],[1],[1]), ([1],[1],[1])], 1)
    assert cluster_macro_curves([-1, -1], tp, fp, fn) is None


def test_singleton_clusters_reduce_to_chain_macro():
    tp, fp, fn = _counts([([10],[0],[0]), ([0],[10],[10]), ([5],[5],[5])], 1)
    f1, _, _, n = cluster_macro_curves([1, 2, 3], tp, fp, fn)
    chain_macro = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean(axis=0)
    assert n == 3, n
    assert abs(f1[0] - chain_macro[0]) < 1e-6, (f1, chain_macro)


def test_curves_have_one_value_per_threshold():
    T = 20
    rng = np.random.RandomState(0)
    tp = rng.randint(0, 50, size=(11, T)).astype(float)
    fp = rng.randint(0, 50, size=(11, T)).astype(float)
    fn = rng.randint(0, 50, size=(11, T)).astype(float)
    f1, pr, rc, n = cluster_macro_curves([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], tp, fp, fn)
    assert f1.shape == (T,) and pr.shape == (T,) and rc.shape == (T,)
    assert n == 6, n
    assert np.all((f1 >= 0) & (f1 <= 1)), f1


def test_per_chain_rows_sum_to_the_old_pooled_counts():
    """`val/*_micro` must keep reproducing the pre-change statistic exactly —
    the pooled totals are now derived by summing the per-chain rows."""
    torch.manual_seed(0)
    B, L, T = 4, 12, 20
    prob = torch.rand(B, L, L)
    contact = (torch.rand(B, L, L) > 0.7).float()
    mask = (torch.rand(B, L, L) > 0.3).float()
    thresholds = torch.linspace(0.05, 0.99, T)

    tp, fp, fn, tn, kept = per_chain_range_counts(prob, contact, mask, thresholds)
    assert kept == [0, 1, 2, 3], kept

    # old pooled path, verbatim
    p_flat = prob[mask > 0].flatten()
    t_flat = contact[mask > 0].flatten()
    preds = (p_flat.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
    t_exp = t_flat.unsqueeze(0).expand_as(preds)
    for got, want in (
        (tp.sum(0), (preds * t_exp).sum(dim=1)),
        (fp.sum(0), (preds * (1 - t_exp)).sum(dim=1)),
        (fn.sum(0), ((1 - preds) * t_exp).sum(dim=1)),
        (tn.sum(0), ((1 - preds) * (1 - t_exp)).sum(dim=1)),
    ):
        assert torch.allclose(got, want, atol=1e-4), (got[:3], want[:3])


def test_chain_without_valid_pairs_is_dropped_not_scored_zero():
    B, L, T = 3, 8, 5
    prob = torch.rand(B, L, L)
    contact = (torch.rand(B, L, L) > 0.5).float()
    mask = torch.ones(B, L, L)
    mask[1] = 0.0                      # chain 1 has no valid pairs
    thresholds = torch.linspace(0.05, 0.99, T)

    tp, _, _, _, kept = per_chain_range_counts(prob, contact, mask, thresholds)
    assert kept == [0, 2], kept        # not [0, 1, 2] with a zero row
    assert tp.shape == (2, T), tp.shape


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
