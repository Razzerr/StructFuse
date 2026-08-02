"""Phase 0 regressions for paper-analysis scripts."""

import os
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# template_coverage uses rootutils only to put the repository root on sys.path.
# The test already did that explicitly, so provide a no-op shim in minimal envs.
if "rootutils" not in sys.modules:
    rootutils = types.ModuleType("rootutils")
    rootutils.setup_root = lambda *args, **kwargs: None
    sys.modules["rootutils"] = rootutils

from scripts.template_coverage import TRAIN_SAMPLE_SPLIT, _accumulate  # noqa: E402
from scripts.ceiling import _flatten_valid_pairs, _per_range_metrics  # noqa: E402
from scripts.feature_correlation import _extract_pair_scalars  # noqa: E402
from scripts.paired_significance import _paired_frame as _significance_paired_frame  # noqa: E402
from scripts.template_stratify import (  # noqa: E402
    DEFAULT_METRICS,
    SIM_BIN_PRESETS,
    _aggregate_paired,
    _load,
    _paired_frame,
)


def test_ceiling_uses_unique_pairs_and_nominal_sequence_length():
    L = 60
    i = torch.arange(L).unsqueeze(1)
    j = torch.arange(L).unsqueeze(0)
    mask = (torch.abs(i - j) >= 6).float().unsqueeze(0)
    # Simulate ten residues with missing coordinates. Nominal L remains 60.
    mask[:, 50:, :] = 0
    mask[:, :, 50:] = 0

    feature = torch.zeros(1, 1, L, L)
    target = torch.zeros(1, L, L)
    long_unique = (
        mask[0].bool()
        & (j - i >= 24)
        & torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    )
    pairs = long_unique.nonzero(as_tuple=False)
    assert pairs.shape[0] > 60
    for row, col in pairs[:40].tolist():
        feature[0, 0, row, col] = 1.0
        target[0, row, col] = 1.0

    X, y = _flatten_valid_pairs(feature, target, mask)
    assert X.shape[0] == int(torch.triu(mask[0], diagonal=1).sum().item())
    assert y.shape[0] == X.shape[0]

    nominal = _per_range_metrics(
        feature.squeeze(1), target, mask, torch.tensor([60])
    )
    active = _per_range_metrics(
        feature.squeeze(1), target, mask, torch.tensor([50])
    )
    assert abs(nominal["long"] - 40 / 60) < 1e-6
    assert abs(active["long"] - 40 / 50) < 1e-6


def test_feature_correlation_counts_each_symmetric_pair_once():
    L = 30
    mask = torch.ones(1, L, L)
    batch = {
        "long_mask": mask,
        "contact": torch.zeros(1, L, L),
        "prior": torch.zeros(1, 1, L, L),
    }
    pairs = _extract_pair_scalars(batch)
    assert pairs["_y"].size == L * (L - 1) // 2
    assert pairs["prior"].size == pairs["_y"].size


def test_paired_significance_requires_unique_ids_and_matching_subsets():
    treatment = pd.DataFrame({
        "sample_id": ["a_A", "b_A"],
        "subset": ["gold", "casp16"],
        "P@L_long": [0.8, 0.6],
    })
    control = pd.DataFrame({
        "sample_id": ["b_A", "a_A"],
        "subset": ["casp16", "gold"],
        "P@L_long": [0.4, 0.5],
    })
    paired = _significance_paired_frame(treatment, control)
    assert paired["sample_id"].tolist() == ["a_A", "b_A"]
    assert paired["subset"].tolist() == ["gold", "casp16"]

    duplicated = pd.concat([control, control.iloc[[0]]], ignore_index=True)
    try:
        _significance_paired_frame(treatment, duplicated)
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("duplicate sample_id was not rejected")

    mismatched = control.copy()
    mismatched.loc[mismatched["sample_id"] == "a_A", "subset"] = "casp16"
    try:
        _significance_paired_frame(treatment, mismatched)
    except ValueError as exc:
        assert "Subset assignment mismatch" in str(exc)
    else:
        raise AssertionError("subset mismatch was not rejected")


def test_paired_stratification_uses_reference_bins_and_control_needs_no_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference_path = root / "reference.tsv"
        control_path = root / "control.tsv"
        pd.DataFrame({
            "sample_id": ["a_A", "b_A", "c_A"],
            "best_tpl_sim": [0.2, 0.6, 0.8],
            "n_templates_retrieved": [1, 4, 4],
            "P@L_long": [0.5, 0.7, 0.9],
            "f1_long": [0.4, 0.6, 0.8],
        }).to_csv(reference_path, sep="\t", index=False)
        # Deliberately omit best_tpl_sim/n_templates_retrieved in the control.
        pd.DataFrame({
            "sample_id": ["c_A", "a_A", "b_A"],
            "P@L_long": [0.4, 0.2, 0.3],
            "f1_long": [0.3, 0.1, 0.2],
        }).to_csv(control_path, sep="\t", index=False)

        # The assertions below reference the "broad" bin labels (sim>0.7).
        sim_bins, sim_labels = SIM_BIN_PRESETS["broad"]
        reference = _load(
            reference_path,
            require_retrieval=True,
            sim_bins=sim_bins,
            sim_labels=sim_labels,
        )
        control = _load(
            control_path,
            require_retrieval=False,
            sim_bins=sim_bins,
            sim_labels=sim_labels,
        )
        merged = _paired_frame(reference, control, DEFAULT_METRICS)
        result = _aggregate_paired(
            merged,
            "sim_bin",
            "frontier",
            "control",
            DEFAULT_METRICS,
            n_bootstrap=200,  # assertions below check means/counts, not CI width
        )

        high = result[result["bin"] == "sim>0.7"].iloc[0]
        assert high["n_pairs"] == 1
        assert abs(high["mean_delta_P@L_long"] - 0.5) < 1e-12
        assert "mean_delta_f1_long" in result.columns
        assert "mean_delta_f1" not in result.columns


def test_coverage_default_skips_train_and_train_label_is_explicit():
    cfg = OmegaConf.load("configs/experiment/diagnostics/ceiling.yaml")
    assert cfg.coverage.include_train is False
    assert TRAIN_SAMPLE_SPLIT == "train_epoch_sample"
    assert TRAIN_SAMPLE_SPLIT != "train"


def test_coverage_counts_only_usable_pair_mask_positions():
    batch = {
        "contact": torch.zeros(1, 3, 3),
        "pair_mask": torch.tensor([[
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]], dtype=torch.float32),
        # Two symmetric nonzeros are usable; two outside pair_mask must not count.
        "prior": torch.tensor([[[[
            0, 1, 1,
            1, 0, 0,
            0, 0, 1,
        ]]]], dtype=torch.float32).reshape(1, 1, 3, 3),
        "n_templates_retrieved": torch.tensor([1]),
        "best_tpl_sim": torch.tensor([0.7]),
        "subset": ["gold"],
    }
    acc = defaultdict(lambda: defaultdict(float))
    _accumulate(batch, acc)
    assert acc["gold"]["n_with_prior"] == 1.0
    assert abs(acc["gold"]["sum_nz_frac"] - 1.0) < 1e-12


def _run_all():
    fns = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
