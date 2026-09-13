"""Figure 2/3 generators: family-unit statistics, declared Holm families, panels.

What this pins (the defects found by the author, 2026-09-13): both generators
bootstrapped CLUSTERS but computed the mean, the delta and the Wilcoxon input
over CHAINS, so a headline interval sat next to a point estimate from a
different estimand; Figure 2 mixed stacks without saying so and still carried
the 2025-only K-sweep; Figure 3 drew rows from two fusion stacks against a
single zero line and declared no Holm families at all in 2.3.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paired_significance import _cluster_means, paired_metric_rows  # noqa: E402


def _load(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises on an unregistered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g23 = _load("paper/results_2_3_relevant_retrieval/generate_artifacts.py", "g23")
g24 = _load("paper/results_2_4_architecture_ablation/generate_artifacts.py", "g24")


# --------------------------------------------------------------------------
# Fixtures: one big redundant family plus singletons — the shape that makes
# chain-weighting and family-weighting disagree.
# --------------------------------------------------------------------------

def _frames(big_gain=0.40, small_gain=0.02, n_big=50, n_small=6):
    rows_t, rows_c = [], []
    for i in range(n_big):
        rows_t.append(("big_%02d" % i, 1, 0.50 + big_gain))
        rows_c.append(("big_%02d" % i, 1, 0.50))
    for i in range(n_small):
        rows_t.append(("small_%02d" % i, 10 + i, 0.50 + small_gain))
        rows_c.append(("small_%02d" % i, 10 + i, 0.50))

    def build(rows):
        return pd.DataFrame({
            "sample_id": [r[0] for r in rows],
            "subset": ["gold"] * len(rows),
            "cluster_id": [r[1] for r in rows],
            "P@L_long": [r[2] for r in rows],
        })

    return build(rows_t), build(rows_c)


def _row(treatment, control, **kwargs):
    rows = paired_metric_rows(
        treatment, control, metrics=("P@L_long",), n_bootstrap=200, **kwargs
    )
    return next(r for r in rows if r["subset"] == "whole")


def test_point_estimate_and_delta_are_family_balanced_not_chain_weighted():
    """50 near-duplicates gaining 40 pp and 6 singletons gaining 2 pp: the chain
    mean says +35.8 pp, the family-balanced mean says +7.4 pp."""
    t, c = _frames()
    row = _row(t, c, unit="cluster")
    expected_cluster = (0.40 + 6 * 0.02) / 7
    expected_chain = (50 * 0.40 + 6 * 0.02) / 56
    assert abs(row["mean_delta"] - expected_cluster) < 1e-12
    assert abs(row["mean_delta_chain"] - expected_chain) < 1e-12
    assert abs(row["mean_delta"] - row["mean_delta_chain"]) > 0.2   # not cosmetic
    assert row["n_clusters"] == 7 and row["n"] == 56
    # The model-level means are re-weighted too, not only the delta.
    assert abs(row["mean_treatment"] - (0.90 + 6 * 0.52) / 7) < 1e-12
    assert abs(row["mean_treatment_chain"] - (50 * 0.90 + 6 * 0.52) / 56) < 1e-12


def test_confidence_interval_brackets_the_family_estimate_not_the_chain_one():
    t, c = _frames()
    row = _row(t, c, unit="cluster")
    assert row["ci95_lo"] <= row["mean_delta"] <= row["ci95_hi"]
    # The chain estimate is far outside the family interval — the pre-fix
    # tables printed exactly that pairing.
    assert row["mean_delta_chain"] > row["ci95_hi"]


def test_wilcoxon_runs_on_family_means_not_on_chains():
    """The signed-rank statistic must see 7 families, not 56 chains."""
    t, c = _frames()
    row = _row(t, c, unit="cluster")
    delta = (t["P@L_long"].to_numpy() - c["P@L_long"].to_numpy())
    clusters = t["cluster_id"].to_numpy()
    by_family = _cluster_means(delta, clusters)
    expected = stats.wilcoxon(by_family, zero_method="wilcox",
                              alternative="two-sided", correction=False)[1]
    chain_p = stats.wilcoxon(delta, zero_method="wilcox",
                             alternative="two-sided", correction=False)[1]
    assert abs(row["wilcoxon_p"] - expected) < 1e-12
    assert row["wilcoxon_p"] > 100 * chain_p        # the two differ by orders


def test_unknown_cluster_rows_are_dropped_not_pooled():
    t, c = _frames(n_big=4, n_small=6)
    t.loc[t.index[:2], "cluster_id"] = -1
    c.loc[c.index[:2], "cluster_id"] = -1
    row = _row(t, c, unit="cluster")
    # 2 unknown chains dropped; the remaining big family plus 6 singletons.
    assert row["n"] == 8 and row["n_clusters"] == 7
    assert not any(cid == -1 for cid in (row["n_clusters"],))


def test_cluster_unit_requires_cluster_ids():
    t, c = _frames(n_big=4)
    t, c = t.drop(columns=["cluster_id"]), c.drop(columns=["cluster_id"])
    try:
        _row(t, c, unit="cluster")
    except ValueError as exc:
        assert "cluster_id" in str(exc)
        return
    raise AssertionError("cluster-unit statistics on a TSV without cluster_id must refuse")


# --------------------------------------------------------------------------
# Holm families
# --------------------------------------------------------------------------

def _matrix(n_clusters=40, chains_per=3):
    rng = np.random.default_rng(7)
    rows = []
    for cid in range(n_clusters):
        for j in range(chains_per):
            base = rng.uniform(0.2, 0.6)
            rows.append({
                "sample_id": f"c{cid}_{j}",
                "subset": "gold",
                "cluster_id": cid,
                "ref": base,
                "worse": base - 0.01 - rng.uniform(0, 0.002),
                "much_worse": base - 0.05 - rng.uniform(0, 0.002),
                "other_ref": base + 0.1,
                "other_worse": base + 0.1 - 0.02 - rng.uniform(0, 0.002),
            })
    return pd.DataFrame(rows)


def test_holm_is_applied_within_a_declared_family_not_across_all_tests():
    """Two families of 2 and 1: the 2-test family multiplies by 2, never by 3."""
    matrix = _matrix()
    contrasts = (
        g24.Contrast("a_vs_ref", "family_one", ("worse",), ("ref",), "ref"),
        g24.Contrast("b_vs_ref", "family_one", ("much_worse",), ("ref",), "ref"),
        g24.Contrast("c_vs_other", "family_two", ("other_worse",), ("other_ref",), "other_ref"),
    )
    table = g24.contrast_table(matrix, contrasts, n_resamples=200, seed=0, unit="cluster")
    sizes = dict(zip(table["comparison"], table["holm_family_size"]))
    assert sizes == {"a_vs_ref": 2, "b_vs_ref": 2, "c_vs_other": 1}
    lone = table[table["comparison"] == "c_vs_other"].iloc[0]
    # A family of one is not adjusted at all.
    assert abs(lone["wilcoxon_p_holm"] - lone["wilcoxon_p"]) < 1e-18
    pair = table[table["family"] == "family_one"].sort_values("wilcoxon_p")
    smallest = pair.iloc[0]
    assert abs(smallest["wilcoxon_p_holm"] - min(1.0, 2 * smallest["wilcoxon_p"])) < 1e-18


def test_every_adjusted_p_names_the_set_it_is_adjusted_over():
    """Holm controls FWER inside the declared family only. The row must carry
    that scope, so a `wilcoxon_p_holm` can never be read out of the TSV as a
    paper-wide corrected value."""
    matrix = _matrix()
    contrasts = (
        g24.Contrast("a_vs_ref", "family_one", ("worse",), ("ref",), "ref"),
        g24.Contrast("b_vs_ref", "family_one", ("much_worse",), ("ref",), "ref"),
        g24.Contrast("c_vs_other", "family_two", ("other_worse",), ("other_ref",), "other_ref"),
    )
    table = g24.contrast_table(matrix, contrasts, n_resamples=200, seed=0, unit="cluster")
    assert "holm_scope" in table.columns
    scopes = dict(zip(table["comparison"], table["holm_scope"]))
    assert scopes == {"a_vs_ref": "family_one", "b_vs_ref": "family_one",
                      "c_vs_other": "family_two"}
    # The scope is exactly the grouping the adjustment used.
    for _, row in table.iterrows():
        same_scope = table[table["holm_scope"] == row["holm_scope"]]
        assert len(same_scope) == row["holm_family_size"]


def test_manifests_state_the_two_scoping_limits():
    """Both limits must survive in the artifact, not only in a session: the
    adjustment is family-scoped, and the p-value is not the interval's estimand."""
    for module in (g23, g24):
        source = Path(module.__file__).read_text()
        assert "wilcoxon_estimand" in source, module.__name__
        # states the limit rather than only the method
        assert "symmetric about zero" in source, module.__name__
        assert "nowhere else" in source, module.__name__


def test_holm_in_figure_2_is_scoped_to_family_subset_and_metric():
    """A Holm family is one question on one metric on one subset — never a
    pool across metrics, which would be a different and stricter claim."""
    families = {c.family for c in g23.COMPARISONS}
    assert families == set(g23.FAMILY_DESCRIPTIONS)
    grouped = [c for c in g23.COMPARISONS if c.family == "retrieval_grouped_8M"]
    assert len(grouped) == 3          # real-vs-none, real-vs-random, random-vs-none
    trufor = [c for c in g23.COMPARISONS if c.family == "retrieval_trufor_8M"]
    assert len(trufor) == 1


# --------------------------------------------------------------------------
# Figure composition
# --------------------------------------------------------------------------

def test_every_forest_row_names_the_reference_it_is_read_against():
    references = {c.reference for c in g24.FOREST_CONTRASTS}
    assert references == {g24.TRUFOR_REF, g24.GROUPED_REF}
    for reference in references:
        assert reference in g24.REFERENCE_TITLE, reference
    for contrast in g24.FOREST_CONTRASTS:
        # The reference must actually be the subtracted model.
        assert contrast.minus == (contrast.reference,), contrast.label
        assert contrast.row_label and contrast.category in g24.CATEGORY_COLOR


def test_the_two_stacks_are_not_mixed_inside_one_reference_block():
    for contrast in g24.FOREST_CONTRASTS:
        variant, reference = contrast.plus[0], contrast.reference
        if reference == g24.GROUPED_REF:
            # A grouped-referenced row must be a grouped-stack variant.
            assert g24.RUNS[variant].stack == "per-group", contrast.label
        else:
            assert reference == g24.TRUFOR_REF


def test_difference_in_differences_reports_no_model_level_mean():
    """Summing two models' scores is not a model's score; the composite rows
    must leave those columns empty and name their two components instead."""
    matrix = _matrix()
    did = g24.Contrast("did", "fam", ("ref", "other_worse"), ("worse", "other_ref"),
                       "—", components=("a", "b"))
    row = g24.contrast_row(matrix, did, n_resamples=200, seed=0, unit="cluster")
    assert math.isnan(row["mean_treatment"]) and math.isnan(row["mean_control"])
    assert row["components"] == "a vs b"
    assert row["expression"] == "ref + other_worse - worse - other_ref"
    expected = (matrix["ref"] + matrix["other_worse"]
                - matrix["worse"] - matrix["other_ref"]).to_numpy()
    by_family = _cluster_means(expected, matrix["cluster_id"].to_numpy())
    assert abs(row["mean_delta"] - by_family.mean()) < 1e-12


def test_grouped_factorial_only_uses_grouped_cells():
    """The distance x triangle interaction is a per-group-stack result: the
    cross-attention (-d,-t) and (-d,+t) cells were never trained."""
    factorial = [c for c in g24.COMPONENT_CONTRASTS if c.family == "grouped_factorial"]
    assert factorial
    for contrast in factorial:
        for label in contrast.plus + contrast.minus:
            assert g24.RUNS[label].stack == "per-group", (contrast.label, label)


def test_figure_2_carries_no_k_sweep():
    """The only K sweep that exists is 2025-generation (per-chain macro,
    uncapped test set); it cannot share an axis with 2026 panels."""
    assert not hasattr(g23, "K_SWEEP")
    assert not any(key.startswith("k") and key[1:].isdigit() for key in g23.RUNS)
    source = (ROOT / "paper/results_2_3_relevant_retrieval/generate_artifacts.py").read_text()
    assert "k_sweep_summary" not in source


def test_figure_2_panels_declare_their_stacks():
    """Panel a is the per-group 8M stack (the only one with a random control);
    panel b is the 650M cross-attention identity audit."""
    assert "grouped_random_k4" in g23.RUNS and "grouped_real_k4" in g23.RUNS
    assert "trufor_no_template" in g23.RUNS      # headline kill-switch still reported
    assert "650m_trufor_s42_2026" in str(g23.IDENTITY_TABLE)
    source = (ROOT / "paper/results_2_3_relevant_retrieval/generate_artifacts.py").read_text()
    assert "ESM2-8M, per-group fusion" in source
    assert "ESM2-650M, cross-attention" in source


def test_panel_a_is_one_point_per_family():
    frames = {
        "a": pd.DataFrame({"sample_id": ["x1", "x2", "y1"], "subset": ["gold"] * 3,
                           "cluster_id": [1, 1, 2], "P@L_long": [0.4, 0.6, 0.9]}),
        "b": pd.DataFrame({"sample_id": ["x1", "x2", "y1"], "subset": ["gold"] * 3,
                           "cluster_id": [1, 1, 2], "P@L_long": [0.1, 0.3, 0.5]}),
    }
    panel = g23.per_cluster_panel(frames, {"a": "real", "b": "random"})
    assert list(panel["cluster_id"]) == [1, 2]
    assert list(panel["n_chains"]) == [2, 1]
    assert abs(panel.loc[0, "real"] - 0.5) < 1e-12      # mean of 0.4 and 0.6
    assert abs(panel.loc[0, "random"] - 0.2) < 1e-12


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
