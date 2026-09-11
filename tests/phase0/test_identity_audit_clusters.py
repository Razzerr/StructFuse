"""retrieval_identity_audit.py summarises per query cluster, not per chain.

The audit's summaries are the paper's evidence on template redundancy and on
where the retrieval gain lives. These tests drive the summarisation functions
on synthetic rows where the cluster-balanced answer is known by construction.
The FAISS replay itself needs the server and is not exercised here.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "retrieval_identity_audit.py"

spec = importlib.util.spec_from_file_location("ria", SCRIPT)
ria = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ria)


def _row(sid, cid, ident, score, ref, ctl, subset="gold", ref_cluster=None, hits=1):
    r = {
        "sample_id": sid,
        "subset": subset,
        "query_cluster_id": str(cid),
        "ref_cluster_id": "" if ref_cluster is None else str(ref_cluster),
        "n_hits": hits,
        "score_for_bin": score,
        "score_bin": ria.bin_label(score),
        "best_template_cluster_id": 999,
        "crop_seq_identity_aligned": ident,
        "crop_seq_identity_query_len": ident,
        "crop_query_coverage": 1.0,
        "crop_identity_bin": ria.identity_bin_label(ident),
        "full_identity_bin": "missing",
    }
    for m in ria.METRIC_COLUMNS:
        r[f"reference_{m}"] = ref
        r[f"control_{m}"] = ctl
        r[f"delta_{m}"] = ref - ctl
    return r


def _rows(unknown=0):
    """id>=0.99 bin: one 40-chain family with gain 0.30 + four singletons with
    gain 0.00 -> cluster gain 0.06, chain gain 0.30*40/44. id<0.20 bin: two
    singletons with gain 0.10."""
    rows = [_row(f"big_{i}", 1, 0.995, 0.9995, 0.6, 0.3) for i in range(40)]
    rows += [_row(f"s_{i}", 10 + i, 0.995, 0.9995, 0.5, 0.5) for i in range(4)]
    rows += [_row(f"lo_{i}", 50 + i, 0.15, 0.90, 0.3, 0.2) for i in range(2)]
    rows += [_row(f"unk_{i}", -1, 0.995, 0.9995, 0.9, 0.0) for i in range(unknown)]
    return rows


def test_identity_bins_have_closed_left_edges_and_a_missing_label():
    assert ria.identity_bin_label(0.30) == "0.30-0.40"
    assert ria.identity_bin_label(0.2999) == "0.20-0.30"
    assert ria.identity_bin_label(0.99) == "id>=0.99"
    assert ria.identity_bin_label(1.0) == "id>=0.99"
    assert ria.identity_bin_label(0.0) == "id<0.20"
    assert ria.identity_bin_label(float("nan")) == "missing"


def test_cluster_balanced_bin_gain_is_not_the_chain_gain():
    rows = _rows()
    labels = [l for l, _, _ in ria.IDENTITY_BINS]
    out = ria.summarize_by(rows, "crop_identity_bin", labels, ["crop"], "cluster", 300, 0)
    hi = next(r for r in out if r["crop_identity_bin"] == "id>=0.99")
    assert hi["n"] == 44 and hi["n_clusters"] == 5, (hi["n"], hi["n_clusters"])
    assert abs(hi["mean_delta_P@L_long"] - 0.30 / 5) < 1e-12, hi["mean_delta_P@L_long"]
    assert abs(hi["mean_delta_P@L_long_chain"] - 0.30 * 40 / 44) < 1e-12
    # cluster-balanced reference/control means are also per cluster
    assert abs(hi["mean_reference_P@L_long"] - (0.6 + 4 * 0.5) / 5) < 1e-12
    lo = next(r for r in out if r["crop_identity_bin"] == "id<0.20")
    assert lo["n_clusters"] == 2 and abs(lo["mean_delta_P@L_long"] - 0.10) < 1e-12


def test_chain_unit_reproduces_the_old_per_chain_summary():
    rows = _rows()
    labels = [l for l, _, _ in ria.IDENTITY_BINS]
    out = ria.summarize_by(rows, "crop_identity_bin", labels, ["crop"], "chain", 300, 0)
    hi = next(r for r in out if r["crop_identity_bin"] == "id>=0.99")
    assert hi["unit"] == "chain" and hi["n_clusters"] == 44
    assert abs(hi["mean_delta_P@L_long"] - 0.30 * 40 / 44) < 1e-12


def test_cluster_ci_cannot_rule_out_zero_when_one_family_carries_the_gain():
    rows = _rows()
    labels = [l for l, _, _ in ria.IDENTITY_BINS]
    c = next(r for r in ria.summarize_by(rows, "crop_identity_bin", labels, ["crop"], "cluster", 500, 0)
             if r["crop_identity_bin"] == "id>=0.99")
    n = next(r for r in ria.summarize_by(rows, "crop_identity_bin", labels, ["crop"], "chain", 500, 0)
             if r["crop_identity_bin"] == "id>=0.99")
    assert n["ci95_lo_delta_P@L_long"] > 0.2, n["ci95_lo_delta_P@L_long"]
    assert c["ci95_lo_delta_P@L_long"] <= 0.0, c["ci95_lo_delta_P@L_long"]


def test_reference_cluster_id_wins_over_index_cluster_and_unknown_is_excluded():
    r = _row("a", 5, 0.9, 0.99, 0.5, 0.4, ref_cluster=7)
    assert ria.cluster_of(r) == 7
    r = _row("b", 5, 0.9, 0.99, 0.5, 0.4)
    assert ria.cluster_of(r) == 5
    r = _row("c", -1, 0.9, 0.99, 0.5, 0.4)
    assert ria.cluster_of(r) == -1
    rows = _rows(unknown=3)
    labels = [l for l, _, _ in ria.IDENTITY_BINS]
    hi = next(r for r in ria.summarize_by(rows, "crop_identity_bin", labels, ["crop"], "cluster", 0, 0)
              if r["crop_identity_bin"] == "id>=0.99")
    assert hi["n"] == 47 and hi["n_clusters"] == 5  # unknown rows counted, not pooled
    assert abs(hi["mean_delta_P@L_long"] - 0.30 / 5) < 1e-12


def test_cluster_balanced_fraction_of_near_duplicates():
    rows = _rows()
    g = ria.global_summary(rows, "cluster", 0, 0)[0]
    # chains: 44 of 46 have identity >= 0.90; clusters: 5 of 7
    assert abs(g["frac_crop_identity_ge_90pct_chain"] - 44 / 46) < 1e-12
    assert abs(g["frac_crop_identity_ge_90pct"] - 5 / 7) < 1e-12
    assert abs(g["frac_crop_identity_lt_30pct"] - 2 / 7) < 1e-12
    assert g["n_clusters"] == 7 and g["n_unknown_query_cluster"] == 0
    # nothing retrieved from the query's own cluster in this fixture
    assert g["frac_best_template_same_cluster_as_query"] == 0.0


def test_casp16_gets_its_own_global_row():
    rows = _rows()
    rows[0]["subset"] = "casp16"; rows[41]["subset"] = "casp16"
    out = ria.global_summary(rows, "cluster", 0, 0)
    assert [r["scope"] for r in out] == ["all", "casp16"]
    assert out[1]["n"] == 2 and out[1]["n_clusters"] == 2


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
