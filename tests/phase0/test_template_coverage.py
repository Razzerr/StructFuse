"""template_coverage.py: per-chain export, cluster-balanced summary, fail-loud.

The three defects this pins (found by the author, 2026-09-13): the script
aggregated over chains only, wrote no per-chain rows, and wrapped the test
loader in `except Exception` so a failure could be written out as a completed
validation-only scan. A fourth requirement: a prior that is non-zero only in
the local band must not count as available for the long-range task.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "template_coverage.py"

# The script is a Hydra entry point; the functions under test need only torch.
# Stub the launcher machinery so the module imports in a plain environment.
for name, attrs in (
    ("rootutils", {"setup_root": lambda *a, **k: ROOT}),
    ("hydra", {"main": lambda **k: (lambda fn: fn), "utils": types.SimpleNamespace(instantiate=None)}),
    ("omegaconf", {"DictConfig": dict}),
):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

spec = importlib.util.spec_from_file_location("tc", SCRIPT)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def _batch(prior_specs, cluster_ids, L=40, min_sep=6, subsets=None):
    """prior_specs[b] = "none" | "local" | "long" | "full" — where the prior is
    non-zero. L=40 gives both a local band (6<=|i-j|<24) and a long band."""
    B = len(prior_specs)
    idx = torch.arange(L)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    pair = torch.ones(B, L, L)
    for b in range(B):
        pair[b].fill_diagonal_(0.0)
    long_mask = pair * (sep >= min_sep).float()
    prior = torch.zeros(B, 1, L, L)
    for b, spec_ in enumerate(prior_specs):
        if spec_ == "local":
            m = (sep >= min_sep) & (sep < tc.LONG_RANGE_MIN_SEP)
        elif spec_ == "long":
            m = sep >= tc.LONG_RANGE_MIN_SEP
        elif spec_ == "full":
            m = sep >= min_sep
        else:
            m = torch.zeros(L, L, dtype=torch.bool)
        prior[b, 0][m] = 0.7
    return {
        "pid": [f"c{b}_A" for b in range(B)],
        "subset": subsets or ["gold"] * B,
        "cluster_id": cluster_ids,
        "seq_len": torch.full((B,), L, dtype=torch.long),
        "pair_mask": pair,
        "long_mask": long_mask,
        "prior": prior,
        "count": prior.clone(),
        "n_templates_retrieved": torch.full((B,), 4, dtype=torch.long),
        "best_tpl_sim": torch.full((B,), 0.9),
    }


def test_a_local_only_prior_is_not_available_for_the_long_range_task():
    rows = tc._per_chain_rows(_batch(["local", "long", "none", "full"], [1, 2, 3, 4]), "test")
    by = {r["sample_id"]: r for r in rows}
    assert by["c0_A"]["has_prior"] == 1 and by["c0_A"]["has_prior_long"] == 0
    assert by["c1_A"]["has_prior"] == 1 and by["c1_A"]["has_prior_long"] == 1
    assert by["c2_A"]["has_prior"] == 0 and by["c2_A"]["has_prior_long"] == 0
    assert by["c3_A"]["has_prior"] == 1 and by["c3_A"]["has_prior_long"] == 1
    # fractions are over unique pairs (j>i), never both triangles
    assert 0.0 < by["c0_A"]["prior_nz_frac"] < 1.0
    assert by["c0_A"]["prior_nz_frac_long"] == 0.0
    assert abs(by["c3_A"]["prior_nz_frac"] - 1.0) < 1e-9
    assert abs(by["c3_A"]["prior_nz_frac_long"] - 1.0) < 1e-9


def test_pair_counts_use_unique_pairs_only():
    rows = tc._per_chain_rows(_batch(["full"], [1], L=40, min_sep=6), "test")
    r = rows[0]
    L, s = 40, 6
    expected = sum(L - d for d in range(s, L))          # j>i, |i-j|>=6
    expected_long = sum(L - d for d in range(tc.LONG_RANGE_MIN_SEP, L))
    assert r["n_valid_pairs"] == expected, (r["n_valid_pairs"], expected)
    assert r["n_valid_long_pairs"] == expected_long


def test_cluster_balanced_differs_from_the_chain_mean():
    """One family of 20 chains with a prior, four singletons without: the chain
    mean says 83 % coverage, the cluster-balanced mean says 20 %."""
    specs = ["full"] * 20 + ["none"] * 4
    cids = [1] * 20 + [10, 11, 12, 13]
    rows = tc._per_chain_rows(_batch(specs, cids), "test")
    s = tc._summarize(rows, "test")[0]
    assert s["n_chains"] == 24 and s["n_clusters"] == 5
    assert abs(s["has_prior_chain"] - 20 / 24) < 1e-6
    assert abs(s["has_prior"] - 1 / 5) < 1e-6


def test_unknown_cluster_rows_are_excluded_not_pooled():
    rows = tc._per_chain_rows(_batch(["full", "full", "none"], [-1, -1, 7]), "test")
    s = tc._summarize(rows, "test")[0]
    assert s["n_unknown_cluster"] == 2 and s["n_clusters"] == 1
    # the two unknown rows would have raised the cluster-balanced value to 2/3
    assert s["has_prior"] == 0.0
    assert abs(s["has_prior_chain"] - 2 / 3) < 1e-6


def test_chains_without_long_range_pairs_are_nan_not_zero():
    """A short chain has no |i-j|>=24 pair at all; it cannot be scored on the
    headline task and must not count as a coverage failure."""
    rows = tc._per_chain_rows(_batch(["full"], [1], L=20), "test")
    r = rows[0]
    assert r["n_valid_long_pairs"] == 0
    assert isinstance(r["has_prior_long"], float) and math.isnan(r["has_prior_long"])
    assert math.isnan(r["prior_nz_frac_long"])
    s = tc._summarize(rows, "test")[0]
    assert s["n_long_defined"] == 0
    assert math.isnan(s["has_prior_long"])          # not 0.0
    assert s["has_prior"] == 1.0                     # the short chain is still covered


def test_a_short_scan_raises_unless_partial_is_allowed():
    batches = [_batch(["full"], [1])]
    try:
        tc._run_loader(iter(batches), 5, None, "test", False, allow_partial=False)
    except RuntimeError as exc:
        assert "scanned 1 chains" in str(exc) and "allow_partial" in str(exc)
    else:
        raise AssertionError("a partial scan must raise")
    rows = tc._run_loader(iter(batches), 5, None, "test", False, allow_partial=True)
    assert len(rows) == 1


def test_duplicate_sample_ids_raise():
    b = _batch(["full"], [1])
    try:
        tc._run_loader(iter([b, b]), 2, None, "test", False, allow_partial=True)
    except RuntimeError as exc:
        assert "duplicate sample_ids" in str(exc)
    else:
        raise AssertionError("duplicate chains must raise")


def test_a_failing_loader_propagates_rather_than_writing_a_partial_table():
    """The old script caught this and still wrote coverage_summary.tsv."""
    def broken():
        yield _batch(["full"], [1])
        raise OSError("worker died")
    try:
        tc._run_loader(broken(), None, None, "test", False, allow_partial=True)
    except OSError:
        return
    raise AssertionError("loader failure must propagate")


def test_summary_reports_both_units_for_every_averaged_field():
    rows = tc._per_chain_rows(_batch(["full", "long"], [1, 2]), "test")
    s = tc._summarize(rows, "test")[0]
    for f in tc.SUMMARY_FIELDS:
        assert f in s and f"{f}_chain" in s, f


def test_per_chain_export_carries_what_offline_reaggregation_needs():
    rows = tc._per_chain_rows(_batch(["full"], [3]), "test")
    for k in ("sample_id", "cluster_id", "has_prior", "has_prior_long",
              "prior_nz_frac", "prior_nz_frac_long", "n_valid_long_pairs"):
        assert k in rows[0] and k in tc.PER_CHAIN_FIELDS, k


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
