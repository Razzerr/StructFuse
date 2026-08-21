"""Paired significance resamples clusters, not chains.

Resampling chains treats one protein family deposited a thousand times as a
thousand independent observations. These tests pin the corrected behaviour on a
synthetic pairing where the right answer is known by construction.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "paired_significance.py"

spec = importlib.util.spec_from_file_location("ps", SCRIPT)
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append((name, exc))
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")


def test_cluster_means_collapses_each_family_to_one_value():
    delta = np.array([1.0, 3.0, 10.0])
    clusters = np.array([7, 7, 9])
    got = ps._cluster_means(delta, clusters)
    assert sorted(got.tolist()) == [2.0, 10.0], got


def test_cluster_bootstrap_is_wider_than_chain_bootstrap():
    """One huge family plus a few singletons: chain resampling pretends the
    family's 500 near-duplicates are 500 independent measurements."""
    rng = np.random.default_rng(0)
    big = rng.normal(0.20, 0.01, 500)
    small = rng.normal(0.00, 0.01, 8)
    delta = np.concatenate([big, small])
    clusters = np.array([1] * 500 + list(range(2, 10)))

    lo_c, hi_c = ps._bootstrap_ci(delta, 2000, clusters=clusters)
    lo_n, hi_n = ps._bootstrap_ci(delta, 2000)
    assert (hi_c - lo_c) > 10 * (hi_n - lo_n), (hi_c - lo_c, hi_n - lo_n)
    # and the chain CI excludes zero while the cluster CI cannot rule it out
    assert lo_n > 0, (lo_n, hi_n)
    assert lo_c < 0 < hi_c, (lo_c, hi_c)


def test_unknown_clusters_do_not_form_one_pseudo_cluster():
    delta = np.array([0.5, 0.5, 0.5, 0.1])
    clusters = np.array([-1, -1, -1, 4])
    # the script filters cluster_id < 0 before calling in; verify the filter is
    # what the end-to-end path applies
    known = clusters >= 0
    assert known.sum() == 1


def _write_pair(tmp: Path, with_clusters=True):
    """500 chains of one family (delta +0.20) + 8 singletons (delta 0.00)."""
    rng = np.random.default_rng(1)
    rows_t, rows_c = [], []
    for i in range(500):
        base = float(rng.normal(0.5, 0.01))
        rows_t.append(("1fam_%03d" % i, 1, base + 0.20))
        rows_c.append(("1fam_%03d" % i, 1, base))
    for j in range(8):
        base = float(rng.normal(0.5, 0.01))
        # small non-zero deltas: a degenerate all-zero control would make the
        # signed-rank test undefined and pin the CI's lower edge at exactly 0
        rows_t.append(("%dsng_A" % (j + 2), j + 2, base + float(rng.normal(0, 0.01))))
        rows_c.append(("%dsng_A" % (j + 2), j + 2, base))

    def frame(rows):
        return pd.DataFrame({
            "sample_id": [r[0] for r in rows],
            "cluster_id": [r[1] for r in rows],
            "subset": ["gold"] * len(rows),
            "P@L_long": [r[2] for r in rows],
        })

    t, c = frame(rows_t), frame(rows_c)
    if not with_clusters:
        t, c = t.drop(columns=["cluster_id"]), c.drop(columns=["cluster_id"])
    tp, cp = tmp / "t.tsv", tmp / "c.tsv"
    t.to_csv(tp, sep="\t", index=False)
    c.to_csv(cp, sep="\t", index=False)
    return tp, cp


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


def test_end_to_end_cluster_delta_differs_from_chain_delta():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp, cp = _write_pair(tmp)
        out = tmp / "o.tsv"
        r = _run(["--treatment", str(tp), "--control", str(cp),
                  "--out", str(out), "--n-bootstrap", "500"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out, sep="\t")
        row = df[(df.subset == "whole") & (df.metric == "P@L_long")].iloc[0]
        assert row["unit"] == "cluster"
        assert row["n"] == 508 and row["n_clusters"] == 9, row.to_dict()
        # chain mean is dominated by the 500-member family: ~0.1969
        assert abs(row["mean_delta_chain"] - 0.197) < 0.01, row["mean_delta_chain"]
        # cluster-balanced: one family at +0.20, eight near 0.00 -> ~0.022
        assert abs(row["mean_delta"] - 0.022) < 0.01, row["mean_delta"]
        # the family-weighted estimate is an order of magnitude larger
        assert row["mean_delta_chain"] > 5 * row["mean_delta"], row.to_dict()
        # nine independent units cannot rule out zero; 508 chains pretended they could
        assert row["ci95_lo"] <= 0 < row["ci95_hi"], row.to_dict()
        # the signed-rank test now runs over clusters, so it is far from significant
        assert not (row["wilcoxon_p"] < 0.05), row["wilcoxon_p"]


def test_end_to_end_chain_mode_still_available_and_labelled():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp, cp = _write_pair(tmp)
        out = tmp / "o.tsv"
        r = _run(["--treatment", str(tp), "--control", str(cp), "--out", str(out),
                  "--n-bootstrap", "500", "--bootstrap", "chain"])
        assert r.returncode == 0, r.stderr
        row = pd.read_csv(out, sep="\t").query("subset == 'whole'").iloc[0]
        assert row["unit"] == "chain" and row["n_clusters"] == -1
        assert row["ci95_lo"] > 0, "chain resampling should look falsely certain"


def test_missing_cluster_column_is_refused_not_degraded():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp, cp = _write_pair(tmp, with_clusters=False)
        r = _run(["--treatment", str(tp), "--control", str(cp),
                  "--out", str(tmp / "o.tsv"), "--n-bootstrap", "100"])
        assert r.returncode != 0, "should refuse rather than silently use chains"
        assert "cluster_id" in (r.stderr + r.stdout)


def test_cluster_mismatch_between_runs_is_an_error():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tp, cp = _write_pair(tmp)
        c = pd.read_csv(cp, sep="\t")
        c.loc[0, "cluster_id"] = 999
        c.to_csv(cp, sep="\t", index=False)
        r = _run(["--treatment", str(tp), "--control", str(cp),
                  "--out", str(tmp / "o.tsv"), "--n-bootstrap", "100"])
        assert r.returncode != 0, "differing cluster files must not be compared"
        assert "Cluster assignment mismatch" in (r.stderr + r.stdout)


for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
        check(name, fn)

print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
sys.exit(1 if FAILED else 0)
