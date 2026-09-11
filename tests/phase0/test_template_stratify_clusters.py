"""template_stratify.py aggregates each similarity bin per cluster, not per chain.

The stratified gain is the paper's evidence that retrieval relevance drives
the improvement. If a bin is dominated by one redundant family, a per-chain
mean reports that family's gain as the bin's gain. These tests construct
exactly that situation and pin the cluster-balanced answer.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "template_stratify.py"


def _make_inputs(tmp: Path, with_cluster: bool = True, unknown_rows: int = 0):
    """One bin (sim>=0.999) holds a 50-chain family with gain 0.30 and five
    singletons with gain 0.00; the other bin (sim<0.95) holds three singletons
    with gain 0.10. Cluster-balanced gain in the big bin is 0.30/6 = 0.05;
    per-chain gain is 0.30*50/55 = 0.2727."""
    rows_ref, rows_ctl = [], []
    def add(sid, cid, sim, ref, ctl):
        r = {"sample_id": sid, "best_tpl_sim": sim, "n_templates_retrieved": 4,
             "P@L_long": ref, "P@L": ref}
        c = {"sample_id": sid, "P@L_long": ctl, "P@L": ctl}
        if with_cluster:
            r["cluster_id"] = cid
            c["cluster_id"] = cid
        rows_ref.append(r); rows_ctl.append(c)
    for i in range(50):
        add(f"big_{i}", 1, 0.9995, 0.50, 0.20)
    for i in range(5):
        add(f"single_{i}", 10 + i, 0.9995, 0.40, 0.40)
    for i in range(3):
        add(f"low_{i}", 100 + i, 0.90, 0.30, 0.20)
    for i in range(unknown_rows):
        add(f"unk_{i}", -1, 0.9995, 0.90, 0.00)
    ref = tmp / "ref.tsv"; ctl = tmp / "ctl.tsv"
    pd.DataFrame(rows_ref).to_csv(ref, sep="\t", index=False)
    pd.DataFrame(rows_ctl).to_csv(ctl, sep="\t", index=False)
    return ref, ctl


def _run(tmp: Path, ref: Path, ctl: Path, *extra):
    out = tmp / "out.tsv"
    cmd = [sys.executable, str(SCRIPT), "--inputs", f"ref={ref}", f"ctl={ctl}",
           "--reference", "ref", "--sim-bin-preset", "high", "--metrics", "P@L_long",
           "--n-bootstrap", "300", "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, (pd.read_csv(out, sep="\t") if out.exists() else None)


def test_cluster_mode_reports_the_family_balanced_bin_gain():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp)
        proc, df = _run(tmp, ref, ctl)
        assert proc.returncode == 0, proc.stderr
        big = df[(df.stratifier == "sim_bin") & (df["bin"] == "sim>=0.999")].iloc[0]
        assert big["unit"] == "cluster"
        assert int(big["n_clusters"]) == 6, big["n_clusters"]
        assert abs(big["mean_delta_P@L_long"] - 0.05) < 1e-9, big["mean_delta_P@L_long"]
        # the chain-weighted number is kept beside it, and is very different
        assert abs(big["mean_delta_P@L_long_chain"] - 0.30 * 50 / 55) < 1e-9


def test_chain_mode_reproduces_the_old_per_chain_number():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp)
        proc, df = _run(tmp, ref, ctl, "--bootstrap", "chain")
        assert proc.returncode == 0, proc.stderr
        big = df[(df.stratifier == "sim_bin") & (df["bin"] == "sim>=0.999")].iloc[0]
        assert big["unit"] == "chain"
        assert abs(big["mean_delta_P@L_long"] - 0.30 * 50 / 55) < 1e-9


def test_cluster_ci_does_not_pretend_fifty_near_duplicates_are_independent():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp)
        _, dc = _run(tmp, ref, ctl)
        _, dn = _run(tmp, ref, ctl, "--bootstrap", "chain")
        sel = lambda df: df[(df.stratifier == "sim_bin") & (df["bin"] == "sim>=0.999")].iloc[0]
        c, n = sel(dc), sel(dn)
        wc = c["ci95_hi_delta_P@L_long"] - c["ci95_lo_delta_P@L_long"]
        wn = n["ci95_hi_delta_P@L_long"] - n["ci95_lo_delta_P@L_long"]
        assert wc > 2 * wn, (wc, wn)
        # chain resampling declares the bin gain solidly positive; cluster
        # resampling, with one gaining family out of six, cannot rule out zero
        assert n["ci95_lo_delta_P@L_long"] > 0.2, n["ci95_lo_delta_P@L_long"]
        assert c["ci95_lo_delta_P@L_long"] <= 0.0, c["ci95_lo_delta_P@L_long"]


def test_unknown_cluster_rows_are_dropped_not_pooled():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp, unknown_rows=4)
        proc, df = _run(tmp, ref, ctl)
        assert proc.returncode == 0, proc.stderr
        assert "dropping 4 rows" in proc.stdout + proc.stderr
        big = df[(df.stratifier == "sim_bin") & (df["bin"] == "sim>=0.999")].iloc[0]
        assert int(big["n_clusters"]) == 6  # the -1 rows did not become a 7th cluster
        assert abs(big["mean_delta_P@L_long"] - 0.05) < 1e-9


def test_cluster_mode_refuses_a_tsv_without_cluster_ids():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp, with_cluster=False)
        proc, df = _run(tmp, ref, ctl)
        assert proc.returncode != 0
        assert "cluster_id" in proc.stderr
        # ...but chain mode still works on such a file
        proc2, df2 = _run(tmp, ref, ctl, "--bootstrap", "chain")
        assert proc2.returncode == 0, proc2.stderr


def test_bins_are_defined_by_the_reference_only():
    """Control rows carry no retrieval metadata; every reference chain must
    still land in exactly one bin and be paired."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); ref, ctl = _make_inputs(tmp)
        proc, df = _run(tmp, ref, ctl)
        sim = df[df.stratifier == "sim_bin"]
        assert int(sim["n_pairs"].sum()) == 58, sim["n_pairs"].tolist()


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
