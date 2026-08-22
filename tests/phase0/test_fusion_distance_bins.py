from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.components.fusion_strategies import (
    GroupedFeatureFusion,
    StandardFusion,
    TruForFusion,
)
from src.models.components.pair2d_head import Pair2DHead


def _fusion_inputs(dist_channels: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        "pair_feat": torch.randn(2, 8, 5, 5),
        "prior": torch.randn(2, 1, 5, 5),
        "count": torch.randn(2, 1, 5, 5),
        "rel": torch.randn(2, 2, 5, 5),
        "esm_contacts": torch.randn(2, 1, 5, 5),
        "tpl_dist_bins": torch.randn(2, dist_channels, 5, 5),
    }


def test_standard_fusion_preserves_legacy_shape_without_distance_bins():
    fusion = StandardFusion(d_pair=8, d_rel=2)
    inputs = _fusion_inputs()

    output = fusion(**inputs)

    assert fusion.tpl_dist_channels == 0
    assert fusion.out_channels == 12
    assert output.shape == (2, 12, 5, 5)


def test_standard_fusion_consumes_distance_bins():
    fusion = StandardFusion(d_pair=8, d_rel=2, tpl_dist_channels=3)
    inputs = _fusion_inputs()

    output = fusion(**inputs)
    changed = fusion(
        **{
            **inputs,
            "tpl_dist_bins": inputs["tpl_dist_bins"] + 1.0,
        }
    )

    assert fusion.out_channels == 15
    assert output.shape == (2, 15, 5, 5)
    assert not torch.equal(output, changed)


@pytest.mark.parametrize(
    "tpl_dist_bins,match",
    [
        (None, "is required"),
        (torch.randn(2, 4, 5, 5), "Expected 3"),
        (torch.randn(2, 3, 4, 4), "must match prior"),
    ],
)
def test_standard_fusion_validates_distance_bins(tpl_dist_bins, match):
    fusion = StandardFusion(d_pair=8, d_rel=2, tpl_dist_channels=3)
    inputs = _fusion_inputs()
    inputs["tpl_dist_bins"] = tpl_dist_bins

    with pytest.raises(ValueError, match=match):
        fusion(**inputs)


def test_trufor_fusion_preserves_legacy_template_encoder():
    fusion = TruForFusion(d_pair=8, d_rel=2, num_heads=2)
    inputs = _fusion_inputs()

    output = fusion(**inputs)

    assert fusion.tpl_dist_channels == 0
    assert fusion.template_encoder[0].in_channels == 2
    assert output.shape == (2, 10, 5, 5)


def test_trufor_fusion_consumes_distance_bins_and_backpropagates():
    fusion = TruForFusion(
        d_pair=8,
        d_rel=2,
        num_heads=2,
        tpl_dist_channels=3,
    ).train()
    inputs = _fusion_inputs()
    inputs["tpl_dist_bins"].requires_grad_(True)

    output = fusion(**inputs)
    output.square().mean().backward()

    first_conv = fusion.template_encoder[0]
    assert first_conv.in_channels == 5
    assert inputs["tpl_dist_bins"].grad is not None
    assert inputs["tpl_dist_bins"].grad.abs().sum() > 0
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad[:, 2:].abs().sum() > 0


@pytest.mark.parametrize(
    "tpl_dist_bins,match",
    [
        (None, "is required"),
        (torch.randn(2, 4, 5, 5), "Expected 3"),
    ],
)
def test_trufor_fusion_validates_distance_bins(tpl_dist_bins, match):
    fusion = TruForFusion(
        d_pair=8,
        d_rel=2,
        num_heads=2,
        tpl_dist_channels=3,
    )
    inputs = _fusion_inputs()
    inputs["tpl_dist_bins"] = tpl_dist_bins

    with pytest.raises(ValueError, match=match):
        fusion(**inputs)


@pytest.mark.parametrize("strategy", ["standard", "trufor"])
def test_pair2d_head_dispatches_distance_bins_to_non_grouped_fusion(strategy):
    head = Pair2DHead(
        d_pair=8,
        width=8,
        depth=0,
        rel_ch=2,
        fusion_strategy=strategy,
        fusion_num_heads=2,
        fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 3},
        use_tpl_dist_bins=True,
        head_type="cnn",
    ).eval()
    inputs = _fusion_inputs()

    with torch.no_grad():
        output = head(**inputs)
        changed = head(
            **{
                **inputs,
                "tpl_dist_bins": inputs["tpl_dist_bins"] + 1.0,
            }
        )

    assert output.shape == (2, 1, 5, 5)
    assert not torch.equal(output, changed)


def test_pair2d_head_requires_declared_distance_channels():
    with pytest.raises(ValueError, match="positive.*tpl_dist"):
        Pair2DHead(
            d_pair=8,
            width=8,
            depth=0,
            rel_ch=2,
            fusion_strategy="standard",
            fusion_feature_groups={"tpl_contact": 2},
            use_tpl_dist_bins=True,
            head_type="cnn",
        )


def test_grouped_fusion_state_keys_are_unchanged_by_distance_flag():
    kwargs = {
        "d_pair": 8,
        "width": 8,
        "depth": 0,
        "rel_ch": 2,
        "fusion_strategy": "grouped",
        "fusion_feature_groups": {"tpl_contact": 2, "tpl_dist": 3},
        "head_type": "cnn",
    }
    disabled = Pair2DHead(**kwargs, use_tpl_dist_bins=False)
    enabled = Pair2DHead(**kwargs, use_tpl_dist_bins=True)

    assert isinstance(enabled.fusion, GroupedFeatureFusion)
    assert disabled.state_dict().keys() == enabled.state_dict().keys()
    for key, value in disabled.state_dict().items():
        assert value.shape == enabled.state_dict()[key].shape


def test_fusion_parity_configs_compose_with_expected_inputs_and_backbones():
    config_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        for strategy in ("standard", "trufor"):
            legacy = compose(
                config_name="train",
                overrides=[f"experiment=ablation/{strategy}_fusion"],
            )
            assert legacy.model.fusion_strategy == strategy
            assert legacy.model.use_tpl_dist_bins is False
            assert legacy.model.fusion_feature_groups is None

            parity_8m = compose(
                config_name="train",
                overrides=[
                    f"experiment=ablation/{strategy}_fusion_with_dist"
                ],
            )
            assert parity_8m.model.fusion_strategy == strategy
            assert parity_8m.model.use_tpl_dist_bins is True
            assert parity_8m.model.fusion_feature_groups.tpl_dist == 9
            assert parity_8m.data.compute_dist_bins is True
            assert parity_8m.data.topk == 4
            assert parity_8m.model.esm_model == "esm2_t6_8M_UR50D"
            # Paths are absolute and version-tagged via paths.data_version, so
            # assert the WIRING (8M config -> 8M index of the active generation)
            # rather than a literal that goes stale on every data rebuild.
            assert parity_8m.data.index_dir == parity_8m.paths.index_t6
            assert parity_8m.data.index_dir.endswith(
                f"index_t6_{parity_8m.paths.data_version}"
            )

            candidate_650m = compose(
                config_name="train",
                overrides=[f"experiment={strategy}_fusion_with_dist_650M"],
            )
            assert candidate_650m.model.fusion_strategy == strategy
            assert candidate_650m.model.use_tpl_dist_bins is True
            assert candidate_650m.model.fusion_feature_groups.tpl_dist == 9
            assert candidate_650m.data.compute_dist_bins is True
            assert candidate_650m.data.topk == 4
            assert candidate_650m.model.esm_model == "esm2_t33_650M_UR50D"
            assert candidate_650m.data.index_dir == candidate_650m.paths.index_t33
            assert candidate_650m.data.index_dir.endswith(
                f"index_t33_{candidate_650m.paths.data_version}"
            )
            assert (
                candidate_650m.data.esm_embeddings_dir
                == candidate_650m.paths.esm_t33
            )
            assert candidate_650m.data.esm_embeddings_dir.endswith(
                f"esm_t33_650M_{candidate_650m.paths.data_version}"
            )
