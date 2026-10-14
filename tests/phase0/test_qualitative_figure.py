"""generate_figure.py (Figure 4): draws only from gated exports, in the
pre-registered order, with the truth in the upper triangle.

The panel composer is pure and pinned exactly; the input gates are exercised
on synthetic manifests; the archived real artifacts are checked for order and
for annotation values equal to the frozen selection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper" / "qualitative_cases" / "generate_figure.py"
ART = ROOT / "paper" / "qualitative_cases" / "artifacts_650m_s42_2026"
MAPS = ROOT / "paper" / "qualitative_cases" / "maps_650m_s42_2026"

spec = importlib.util.spec_from_file_location("qfig", SCRIPT)
qfig = importlib.util.module_from_spec(spec)
sys.modules["qfig"] = qfig
spec.loader.exec_module(qfig)


def _cmap():
    import matplotlib
    matplotlib.use("Agg")
    return matplotlib.colormaps["Blues"]


def test_panel_puts_truth_above_the_diagonal_and_the_map_below():
    L = 40
    values = np.zeros((L, L), np.float32); values[30, 2] = 1.0; values[2, 30] = 1.0   # symmetric input
    contact = np.zeros((L, L), np.uint8); contact[2, 30] = contact[30, 2] = 1
    pair_mask = np.ones((L, L), np.uint8); pair_mask[5, :] = 0; pair_mask[:, 5] = 0
    img = qfig.compose_panel(values, contact, pair_mask, _cmap(), long_range_min=24)
    assert img.shape == (L, L, 4)
    # (2,30): upper triangle, contact -> black
    assert np.allclose(img[2, 30, :3], 0.0)
    # (30,2): lower triangle -> colormap(1.0) = darkest blue, NOT black
    dark = np.array(_cmap()(1.0)[:3])
    assert np.allclose(img[30, 2, :3], dark) and not np.allclose(img[30, 2, :3], 0.0)
    # a plain upper-triangle non-contact stays white; inside the band it is dimmed
    assert np.allclose(img[1, 35, :3], 1.0) and img[1, 10, 0] < 1.0
    # no-coordinate row/column is grey regardless of triangle
    assert np.allclose(img[5, 20, :3], 0.82) and np.allclose(img[20, 5, :3], 0.82)
    # diagonal marked
    assert np.allclose(img[7, 7, :3], 0.7)


def test_refuses_exports_that_did_not_pass_their_gates():
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "structfuse"; run.mkdir()
        (run / "exported_cases.tsv").write_text("sample_id\tP@L_long\nx_A\t0.5\n")
        for ident, gate, needle in (("MATCH", "FAIL", "gate 'FAIL'"),
                                    ("MISMATCH", "PASS", "identity 'MISMATCH'")):
            (run / "manifest.json").write_text(json.dumps({
                "tag": "structfuse", "checkpoint_identity_vs_evaluation": {"status": ident},
                "gate": {"status": gate}}))
            try:
                qfig.load_export(run, "structfuse")
            except ValueError as exc:
                assert needle in str(exc), str(exc)
            else:
                raise AssertionError(f"accepted identity={ident} gate={gate}")
        (run / "manifest.json").write_text(json.dumps({
            "tag": "no_templates", "checkpoint_identity_vs_evaluation": {"status": "MATCH"},
            "gate": {"status": "PASS"}}))
        try:
            qfig.load_export(run, "structfuse")
        except ValueError as exc:
            assert "tag" in str(exc)
        else:
            raise AssertionError("accepted a mis-tagged export")


def test_refuses_an_export_that_is_not_of_the_frozen_selection():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sel = tmp / "selected_cases.tsv"
        pd.DataFrame([{"category": "largest_gain", "role": "primary", "sample_id": "x_A",
                       "ref": 0.9, "ctl": 0.4}]).to_csv(sel, sep="\t", index=False)
        ok = {"cases": pd.DataFrame([{"sample_id": "x_A", "P@L_long": 0.9}]).set_index("sample_id")}
        ctl = {"cases": pd.DataFrame([{"sample_id": "x_A", "P@L_long": 0.4}]).set_index("sample_id")}
        assert list(qfig.load_cases(sel, ok, ctl).index) == ["x_A"]
        drift = {"cases": pd.DataFrame([{"sample_id": "x_A", "P@L_long": 0.91}]).set_index("sample_id")}
        try:
            qfig.load_cases(sel, drift, ctl)
        except ValueError as exc:
            assert "not of the frozen selection" in str(exc)
        else:
            raise AssertionError("a drifted export was accepted")


def test_row_order_is_the_plan_order_and_supplementary_is_separate():
    assert list(qfig.MAIN_ROWS) == ["largest_gain", "near_mean_gain_homologous",
                                    "near_mean_gain_remote", "failure"]
    assert list(qfig.SUPP_ROWS) == ["near_mean_gain_near_dup"]
    assert qfig.LONG_RANGE_MIN == 24
    assert [c for c, _ in qfig.COLUMNS] == ["prior", "prob_no_templates", "prob_structfuse"]


def test_archived_figure_rows_equal_the_frozen_selection():
    rows_tsv = ART / "figure_4_rows.tsv"
    if not rows_tsv.exists() or not (ART / "selected_cases.tsv").exists():
        print("       (skipped — archived figure not in this checkout)")
        return
    rows = pd.read_csv(rows_tsv, sep="\t")
    sel = pd.read_csv(ART / "selected_cases.tsv", sep="\t")
    sel = sel[sel.role == "primary"].set_index("category")
    assert rows.category.tolist() == list(qfig.MAIN_ROWS) + list(qfig.SUPP_ROWS)
    assert rows.figure.tolist() == ["4"] * 4 + ["S"]
    for _, r in rows.iterrows():
        s = sel.loc[r.category]
        assert r.sample_id == s.sample_id
        assert abs(r["delta_P@L_long"] - (s.ref - s.ctl)) < 1e-9
        assert abs(r["P@L_long_structfuse"] - s.ref) < 1e-9 and abs(r["P@L_long_no_templates"] - s.ctl) < 1e-9
    for stem in ("figure_4_qualitative_cases", "figure_S_qualitative_near_dup"):
        for ext in (".pdf", ".svg", ".png"):
            assert (ART / f"{stem}{ext}").exists(), f"{stem}{ext} missing"
    man = json.loads((ART / "figure_manifest.json").read_text())
    for tag in ("structfuse", "no_templates"):
        assert man["exports"][tag]["identity"] == "MATCH" and man["exports"][tag]["gate"] == "PASS"


def test_real_generation_into_a_temp_dir_when_exports_are_archived():
    if not (MAPS / "structfuse" / "manifest.json").exists():
        print("       (skipped — archived exports not in this checkout)")
        return
    with tempfile.TemporaryDirectory() as d:
        rc = qfig.main(["--out-dir", d])
        assert rc == 0
        produced = sorted(p.name for p in Path(d).iterdir())
        assert "figure_4_qualitative_cases.pdf" in produced and "figure_4_rows.tsv" in produced
        assert len([p for p in produced if p.startswith("figure_") and p.endswith((".pdf", ".svg", ".png"))]) == 6


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
