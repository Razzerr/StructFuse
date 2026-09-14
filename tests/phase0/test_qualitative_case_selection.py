"""select_qualitative_cases.py picks families by a rule fixed before any map is seen.

The figure this feeds is the paper's most cherry-pickable artifact, so the
tests pin the properties that make the rule defensible: family unit, one
family per case, extremes guarded by eligibility, the failure category needing
a prior to blame, refusal of mismatched inputs, and left-closed regimes. The
last test checks the archived real selection against Tables 1 / 12c.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import select_qualitative_cases as sq  # noqa: E402


# ----------------------------------------------------------------------------
# fixture
# ----------------------------------------------------------------------------

class World:
    """Builds the four per-chain tables from a compact chain spec."""

    def __init__(self):
        self.rows = []

    def add(self, sid, cluster, identity, ref, ctl, *, L=120, npos=None, qlen=None,
            prior_long=1, subset="gold"):
        npos = L if npos is None else npos
        qlen = L if qlen is None else qlen
        self.rows.append(dict(sample_id=sid, cluster=cluster, identity=identity, ref=ref,
                              ctl=ctl, L=L, npos=npos, qlen=qlen, prior_long=prior_long,
                              subset=subset))
        return self

    def write(self, tmp: Path, *, shuffle_seed=None, drop_from_control=None,
              perturb_identity_delta=False):
        rows = list(self.rows)
        if shuffle_seed is not None:
            rng = np.random.RandomState(shuffle_seed)
            rng.shuffle(rows)
        ref, ctl, idt, cov = [], [], [], []
        for r in rows:
            pdb, chain = r["sample_id"].split("_")
            base = dict(sample_id=r["sample_id"], pdb_id=pdb, chain_id=chain,
                        subset=r["subset"], cluster_id=r["cluster"], seq_len=min(r["L"], 384),
                        n_valid_long_pairs=max(r["L"] * 3, 1), n_pos_long=r["npos"],
                        best_tpl_sim=0.99, n_templates_retrieved=4)
            ref.append({**base, "P@L_long": r["ref"]})
            ctl.append({**base, "P@L_long": r["ctl"], "best_tpl_sim": 0.0,
                        "n_templates_retrieved": 0})
            crop_len = min(r["L"], 384)
            delta = r["ref"] - r["ctl"] + (0.01 if perturb_identity_delta else 0.0)
            idt.append(dict(sample_id=r["sample_id"], query_len=r["qlen"], crop_len=crop_len,
                            crop_start=0, crop_end=crop_len, best_template_id="1tpl_A",
                            best_score=0.99, crop_seq_identity_aligned=r["identity"],
                            crop_query_coverage=1.0, **{"delta_P@L_long": delta}))
            cov.append(dict(split="test", sample_id=r["sample_id"], has_prior_long=r["prior_long"],
                            prior_nz_frac_long=0.02 if r["prior_long"] else 0.0))
        paths = {}
        for name, data in (("reference", ref), ("control", ctl), ("identity", idt), ("coverage", cov)):
            if name == "control" and drop_from_control:
                data = [d for d in data if d["sample_id"] != drop_from_control]
            p = tmp / f"{name}.tsv"
            pd.DataFrame(data).to_csv(p, sep="\t", index=False)
            paths[name] = p
        return paths


def _args(paths, out, **kw):
    d = dict(reference=paths["reference"], control=paths["control"], identity=paths["identity"],
             coverage=paths["coverage"], out_dir=out, metric="P@L_long", min_len=80,
             max_len=384, min_pos_frac=0.5, n_backups=1)
    d.update(kw)
    return argparse.Namespace(**d)


def _baseline_world() -> World:
    """Every regime populated with several families; nothing degenerate."""
    w = World()
    # near-dup regime: family 1 big and strong, families 2-5 singletons
    for i in range(6):
        w.add(f"1big_{i}", 1, 0.99, 0.80, 0.40)
    for i, g in enumerate([0.10, 0.20, 0.30, 0.50]):
        w.add(f"1nd{i}_A", 2 + i, 0.95, 0.40 + g, 0.40)
    # homologous regime
    for i, g in enumerate([0.00, 0.02, 0.05, 0.08, 0.20]):
        w.add(f"2hm{i}_A", 10 + i, 0.50, 0.40 + g, 0.40)
    # remote regime
    for i, g in enumerate([-0.05, 0.00, 0.00, 0.01, 0.03]):
        w.add(f"3rm{i}_A", 20 + i, 0.20, 0.30 + g, 0.30)
    # a clear failure family with a prior present
    w.add("4bad_A", 30, 0.60, 0.10, 0.45, prior_long=1)
    return w


def _run(world: World, **write_kw):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = world.write(tmp, **write_kw)
        sel = sq.run(_args(paths, tmp / "out"))
        man = json.loads((tmp / "out" / "manifest.json").read_text())
        summ = pd.read_csv(tmp / "out" / "strata_summary.tsv", sep="\t")
        return sel, man, summ


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def test_selection_is_deterministic_under_row_shuffle():
    a, _, _ = _run(_baseline_world())
    b, _, _ = _run(_baseline_world(), shuffle_seed=7)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_near_mean_target_is_over_family_means_not_chains():
    """Homologous regime: one 8-chain family at +0.30 and five singletons at 0.
    Chain mean = 0.185 (nearest: the big family); family mean = 0.05 (nearest:
    a singleton at 0). The near-mean pick must be a singleton, not the big
    family — the unit is the family, as in Table 12c."""
    w = World()
    for i in range(8):
        w.add(f"1big_{i}", 1, 0.50, 0.70, 0.40)
    for i in range(5):
        w.add(f"2sg{i}_A", 2 + i, 0.50, 0.40, 0.40)
    # other regimes / categories need something to pick, keep them trivial
    w.add("3nd_A", 50, 0.95, 0.60, 0.40)
    w.add("3nd_B", 53, 0.95, 0.55, 0.40)   # largest_gain's backup takes one near-dup family
    w.add("4rm_A", 51, 0.10, 0.30, 0.30)
    w.add("5bad_A", 52, 0.95, 0.10, 0.45)   # failure lives outside the regime under test
    sel, _, summ = _run(w)
    row = sel[(sel.category == "near_mean_gain_homologous") & (sel.role == "primary")].iloc[0]
    assert row.family_mean_gain == 0.0 and row.cluster_id != 1, row.to_dict()
    hm = summ[summ.regime == "0.30-0.90"].iloc[0]
    # the summary TSV is written with 6 significant digits
    assert abs(hm.chain_gain_mean - 0.30 * 8 / 13) < 1e-5
    assert abs(hm.family_mean_gain_mean - 0.05) < 1e-5


def test_one_family_per_case_including_backups():
    sel, _, _ = _run(_baseline_world())
    assert sel.cluster_id.is_unique, sel[["category", "role", "cluster_id"]]
    assert set(sel.category) == {c.name for c in sq.CATEGORIES}
    assert (sel.groupby("category").size() == 2).all(), "primary + 1 backup each"


def test_ineligible_extreme_family_is_skipped_not_chosen():
    """A 20-residue singleton with gain +1.0 has the largest family mean; it is
    not a legible case and must be skipped. The rank column shows the skip."""
    w = _baseline_world().add("9tiny_A", 99, 0.99, 1.0, 0.0, L=20, npos=20)
    sel, _, _ = _run(w)
    lg = sel[(sel.category == "largest_gain") & (sel.role == "primary")].iloc[0]
    assert lg.cluster_id != 99 and lg.family_rank_incl_skipped == 2, lg.to_dict()
    # and a chain cropped out of a longer protein is also ineligible
    w2 = _baseline_world().add("9long_A", 98, 0.99, 1.0, 0.0, L=384, npos=384, qlen=900)
    sel2, _, _ = _run(w2)
    lg2 = sel2[(sel2.category == "largest_gain") & (sel2.role == "primary")].iloc[0]
    assert lg2.cluster_id != 98


def test_failure_requires_a_long_range_prior_to_blame():
    w = _baseline_world().add("9worst_A", 97, 0.60, 0.00, 0.60, prior_long=0)
    sel, _, _ = _run(w)
    f = sel[(sel.category == "failure") & (sel.role == "primary")].iloc[0]
    assert f.cluster_id != 97 and f.has_prior_long == 1, f.to_dict()
    assert f.cluster_id == 30, "the worst family WITH a prior is the failure case"


def test_refuses_population_mismatch():
    try:
        _run(_baseline_world(), drop_from_control="2hm0_A")
    except ValueError as exc:
        assert "population mismatch" in str(exc)
    else:
        raise AssertionError("nothing raised")


def test_refuses_identity_table_from_another_pair():
    try:
        _run(_baseline_world(), perturb_identity_delta=True)
    except ValueError as exc:
        assert "different pair" in str(exc)
    else:
        raise AssertionError("nothing raised")


def test_regime_boundaries_are_left_closed_like_table_12():
    assert sq.identity_regime(0.30) == sq.REGIME_HOMOLOGOUS
    assert sq.identity_regime(0.2999) == sq.REGIME_REMOTE
    assert sq.identity_regime(0.90) == sq.REGIME_NEAR_DUP
    assert sq.identity_regime(0.8999) == sq.REGIME_HOMOLOGOUS
    assert sq.identity_regime(float("nan")) == "missing"


def test_categories_cover_the_planned_rows_and_are_ordered_extremes_first():
    names = [c.name for c in sq.CATEGORIES]
    assert names[:2] == ["largest_gain", "failure"], "extremes claim families first"
    assert sorted(c.plan_row for c in sq.CATEGORIES if c.plan_row) == [1, 2, 3, 4]
    assert {c.effect for c in sq.CATEGORIES} == {"helps", "no_effect", "hurts"}
    fail = next(c for c in sq.CATEGORIES if c.name == "failure")
    assert fail.require_prior_long and fail.quantile == "min"
    assert all(c.quantile == "mean" for c in sq.CATEGORIES if c.name.startswith("near_mean_gain")), \
        "near-mean-gain cases sit at the reported family-balanced gain, not at a median"
    assert not any("typical" in c.name for c in sq.CATEGORIES), \
        "'typical' overclaims: these cases match the mean delta, not typical difficulty"


def test_archived_real_selection_reproduces_tables_1_and_12c():
    """The strata summary's family-mean column IS the paper's cluster-macro
    delta: +7.23 pp overall and +0.91 / +5.39 / +17.40 pp by regime."""
    p = ROOT / "paper" / "qualitative_cases" / "artifacts_650m_s42_2026" / "strata_summary.tsv"
    if not p.exists():
        print("       (skipped — archived selection not in this checkout)")
        return
    s = pd.read_csv(p, sep="\t").set_index("regime")["family_mean_gain_mean"]
    for regime, expect in (("all", 0.0723), ("<0.30", 0.0091), ("0.30-0.90", 0.0539),
                           (">=0.90", 0.1740)):
        assert abs(round(float(s[regime]), 4) - expect) < 1e-9, (regime, s[regime])
    sel = pd.read_csv(p.parent / "selected_cases.tsv", sep="\t")
    assert sel.cluster_id.is_unique and len(sel) == 2 * len(sq.CATEGORIES)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
