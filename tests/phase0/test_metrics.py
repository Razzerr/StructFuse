"""Phase 0 — metric-correctness tests (B0 unique pairs, K source, macro, NaN AUC,
val-threshold helper, per-sample schema). Plain-python runnable: `python tests/phase0/test_metrics.py`.

Independent (numpy/hand) expected values — never call the function-under-test to
build its own oracle.
"""
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.utils.metrics import (  # noqa: E402
    unique_pair_mask,
    precision_at_k_masked,
    auc_pr_masked,
    range_metrics_at_threshold,
    per_sample_metric_rows,
    log_macro_test_metrics,
    export_per_sample_tsv,
)


def test_unique_pair_mask_keeps_only_upper_triangle():
    m = torch.ones(1, 5, 5)
    u = unique_pair_mask(m)
    # 5x5 upper triangle (strict) has 10 entries; lower + diagonal are zero.
    assert u.sum().item() == 10, u.sum().item()
    assert u[0].tril().sum().item() == 0  # nothing on/below diagonal


def test_auc_undefined_returns_nan_when_requested_else_zero():
    # All-positive long-range targets -> AUC undefined (no negatives).
    L = 30
    prob = torch.rand(1, L, L)
    contact = torch.ones(1, L, L)
    mask = torch.ones(1, L, L)
    default = auc_pr_masked(prob, contact, mask, range_type="long")
    nan_val = auc_pr_masked(prob, contact, mask, range_type="long", undefined_value=float("nan"))
    assert default == 0.0, default
    assert math.isnan(nan_val), nan_val


def test_range_metrics_uses_val_threshold_not_test_argmax():
    thresholds = torch.tensor([0.2, 0.5, 0.9])
    # Craft so F1 is maximized at idx 0, but pred_threshold (val) points to idx 2.
    tp = torch.tensor([10.0, 5.0, 1.0])
    fp = torch.tensor([0.0, 0.0, 0.0])
    fn = torch.tensor([0.0, 5.0, 9.0])
    tn = torch.tensor([10.0, 10.0, 10.0])
    out = range_metrics_at_threshold(tp, fp, fn, tn, thresholds, pred_threshold=0.9)
    # idx 2: precision=1/1=1.0, recall=1/10=0.1, f1=2*.1/1.1
    expected_f1 = 2 * 1.0 * 0.1 / (1.0 + 0.1)
    assert abs(out["f1"] - expected_f1) < 1e-6, out
    # NOT the argmax-on-test (idx0 f1=1.0)
    assert out["f1"] < 0.99, out


def _full_mask_no_diag(L, min_sep):
    i = torch.arange(L).unsqueeze(1)
    j = torch.arange(L).unsqueeze(0)
    return (torch.abs(i - j) >= min_sep).float().unsqueeze(0)


def test_per_sample_K_uses_nominal_seq_len_not_active_rows():
    # L_nominal=30 but residues 25..29 fully masked (missing coords) -> 25 active.
    L = 30
    mask = _full_mask_no_diag(L, 6)
    mask[0, 25:, :] = 0
    mask[0, :, 25:] = 0
    # Build prob/contact on upper-tri valid pairs: 18 true (high prob), rest false (low).
    prob = torch.zeros(1, L, L)
    contact = torch.zeros(1, L, L)
    ut = (torch.triu(torch.ones(L, L), diagonal=1) * mask[0]).bool()
    idx = ut.nonzero(as_tuple=False)
    assert idx.shape[0] >= 30
    for n, (i, j) in enumerate(idx.tolist()):
        if n < 18:
            prob[0, i, j] = 0.9
            contact[0, i, j] = 1.0
        else:
            prob[0, i, j] = 0.1
    rows_nominal = per_sample_metric_rows(
        prob, contact, mask, ["1abc_A"], ["all"], threshold=0.5, seq_lens=[30]
    )
    rows_active = per_sample_metric_rows(
        prob, contact, mask, ["1abc_A"], ["all"], threshold=0.5, seq_lens=None
    )
    # K=30 -> P@L = 18/30 = 0.6 ; K=active(25) -> 18/25 = 0.72 (float32 tol)
    assert abs(rows_nominal[0]["P@L"] - 18 / 30) < 1e-5, rows_nominal[0]["P@L"]
    assert abs(rows_active[0]["P@L"] - 18 / 25) < 1e-5, rows_active[0]["P@L"]


def test_per_sample_schema_and_no_rounding_and_nan_auc():
    L = 40
    mask = _full_mask_no_diag(L, 6)
    prob = torch.rand(1, L, L) * 0.4  # all below 0.5 threshold
    contact = torch.zeros(1, L, L)
    # Make P@L exactly 1/3: among top-L unique pairs, 1/3 true. Simpler: assert
    # full precision via a crafted 1/3 — set 1 true of top-3 by using seq_len so K=3.
    ut = (torch.triu(torch.ones(L, L), diagonal=1) * mask[0]).bool()
    idx = ut.nonzero(as_tuple=False).tolist()
    # highest 3 probs -> 1 true => P@3 = 1/3
    prob[0, idx[0][0], idx[0][1]] = 0.99
    contact[0, idx[0][0], idx[0][1]] = 1.0
    prob[0, idx[1][0], idx[1][1]] = 0.98
    prob[0, idx[2][0], idx[2][1]] = 0.97
    rows = per_sample_metric_rows(
        prob, contact, mask, ["1abc_A"], ["all"], threshold=0.5, seq_lens=[3]
    )
    r = rows[0]
    required = {
        "sample_id", "pdb_id", "chain_id", "subset", "seq_len",
        "P@L", "P@L/2", "P@L/5", "P@L_short", "P@L_medium", "P@L_long",
        "P@L/2_long", "P@L/5_long", "AUC-PR_long",
        "f1_long", "precision_long", "recall_long",
        "n_valid_long_pairs",
    }
    missing = required - set(r.keys())
    assert not missing, f"missing columns: {missing}"
    # Full precision: 1/3 preserved (float32 tol), and NOT rounded to 4 decimals
    assert abs(r["P@L"] - 1.0 / 3.0) < 1e-5, r["P@L"]
    assert r["P@L"] != round(1.0 / 3.0, 4), "value was rounded"
    # No positives anywhere -> long-range AUC undefined -> NaN
    assert isinstance(r["AUC-PR_long"], float)
    assert math.isnan(r["AUC-PR_long"]), r["AUC-PR_long"]
    assert r["n_valid_long_pairs"] >= 0


def test_no_valid_long_pairs_are_nan_and_excluded_from_macro():
    L = 24
    mask = _full_mask_no_diag(L, 6)
    rows = per_sample_metric_rows(
        torch.rand(1, L, L),
        torch.zeros(1, L, L),
        mask,
        ["short_A"],
        ["gold"],
        threshold=0.5,
        seq_lens=[L],
    )
    row = rows[0]
    assert row["n_valid_long_pairs"] == 0
    for key in (
        "P@L_long",
        "P@L/2_long",
        "P@L/5_long",
        "AUC-PR_long",
        "f1_long",
        "precision_long",
        "recall_long",
    ):
        assert math.isnan(row[key]), (key, row[key])


def test_valid_long_pairs_without_contacts_keep_zero_ranking_and_prf():
    L = 30
    mask = _full_mask_no_diag(L, 6)
    rows = per_sample_metric_rows(
        torch.rand(1, L, L),
        torch.zeros(1, L, L),
        mask,
        ["negative_A"],
        ["gold"],
        threshold=0.5,
        seq_lens=[L],
    )
    row = rows[0]
    assert row["n_valid_long_pairs"] > 0
    for key in (
        "P@L_long",
        "P@L/2_long",
        "P@L/5_long",
        "f1_long",
        "precision_long",
        "recall_long",
    ):
        assert row[key] == 0.0, (key, row[key])
    assert math.isnan(row["AUC-PR_long"])


def test_macro_logger_is_nan_aware_and_writes_canonical_keys_once():
    metric_keys = (
        "P@L", "P@L/2", "P@L/5", "P@L_short", "P@L_medium", "P@L_long",
        "P@L/2_long", "P@L/5_long", "AUC-PR_long",
        "f1_long", "precision_long", "recall_long", "f1", "precision", "recall",
    )

    def make_row(sample_id, subset, value, n_valid):
        row = {key: float(value) for key in metric_keys}
        row.update({
            "sample_id": sample_id,
            "subset": subset,
            "n_valid_long_pairs": n_valid,
        })
        if n_valid == 0:
            for key in (
                "P@L_long", "P@L/2_long", "P@L/5_long", "AUC-PR_long",
                "f1_long", "precision_long", "recall_long",
            ):
                row[key] = float("nan")
        return row

    # Equivalent to uneven source batches: one row plus a later two-row batch.
    rows = [
        make_row("a_A", "gold", 0.2, 10),
        make_row("b_A", "gold", 0.8, 20),
        make_row("c_A", "casp16", 0.4, 0),
    ]
    logged = {}

    def capture(key, value, **_kwargs):
        assert key not in logged, f"duplicate logger write: {key}"
        logged[key] = float(value)

    log_macro_test_metrics(capture, rows, ["gold", "casp16"])
    assert abs(logged["test/P@L_long"] - 0.5) < 1e-12
    assert logged["test/n_proteins"] == 3.0
    assert logged["test/n_proteins_long_defined"] == 2.0
    assert logged["test/gold/n_proteins_long_defined"] == 2.0
    assert logged["test/casp16/n_proteins_long_defined"] == 0.0
    assert not any(key.endswith("_batchavg") or key.endswith("_micro") for key in logged)


def test_tsv_export_writes_nan_literal_losslessly():
    with tempfile.TemporaryDirectory() as tmp:
        rows = [{"sample_id": "short_A", "P@L_long": float("nan"), "P@L": 1 / 3}]
        export_per_sample_tsv(rows, SimpleNamespace(log_dir=tmp))
        text = (Path(tmp) / "per_sample_metrics.tsv").read_text()
        assert "\tnan\t" in text
        assert repr(1 / 3) in text


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
    sys.exit(_run_all())
