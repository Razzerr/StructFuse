import torch
from hydra import compose, initialize_config_dir

from src.models.components.fusion_strategies import GroupedFeatureFusion
from src.models.components.pair2d_head import Pair2DHead


def _head(use_template_features: bool) -> Pair2DHead:
    return Pair2DHead(
        d_pair=8,
        width=8,
        depth=0,
        rel_ch=2,
        fusion_strategy="grouped",
        fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 3},
        use_template_features=use_template_features,
        head_type="cnn",
    )


def _inputs():
    torch.manual_seed(7)
    shape = (2, 8, 6, 6)
    return {
        "pair_feat": torch.randn(shape),
        "prior": torch.randn(2, 1, 6, 6),
        "count": torch.randn(2, 1, 6, 6),
        "rel": torch.randn(2, 2, 6, 6),
        "esm_contacts": torch.randn(2, 1, 6, 6),
        "tpl_dist_bins": torch.randn(2, 3, 6, 6),
    }


def test_no_template_head_ignores_template_tensors():
    head = _head(use_template_features=False).eval()
    inputs = _inputs()

    with torch.no_grad():
        output_with_values = head(**inputs)
        output_with_zeros = head(
            **{
                **inputs,
                "prior": torch.zeros_like(inputs["prior"]),
                "count": torch.zeros_like(inputs["count"]),
                "tpl_dist_bins": torch.zeros_like(inputs["tpl_dist_bins"]),
            }
        )

    torch.testing.assert_close(output_with_values, output_with_zeros)


def test_no_template_encoders_are_frozen_and_receive_no_gradients():
    head = _head(use_template_features=False).train()
    inputs = _inputs()

    assert isinstance(head.fusion, GroupedFeatureFusion)
    template_params = list(head.fusion.encoders.parameters())
    assert template_params
    assert all(not param.requires_grad for param in template_params)

    head(**inputs).square().mean().backward()

    assert all(param.grad is None for param in template_params)
    assert any(
        param.grad is not None for param in head.fusion.esm_encoder.parameters()
    )


def test_frontier_head_still_uses_template_tensors():
    head = _head(use_template_features=True).eval()
    inputs = _inputs()

    with torch.no_grad():
        # The template encoders are zero-initialized, so perturb one weight to
        # verify that the enabled path consumes the modality.
        head.fusion.encoders["tpl_contact"][3].weight.fill_(0.01)
        output_with_values = head(**inputs)
        output_with_zeros = head(
            **{
                **inputs,
                "prior": torch.zeros_like(inputs["prior"]),
                "count": torch.zeros_like(inputs["count"]),
            }
        )

    assert not torch.equal(output_with_values, output_with_zeros)


def test_no_template_configs_disable_all_template_features():
    from pathlib import Path

    config_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        for experiment in (
            "ablation/no_templates",
            "baseline/esm2_650m_trained",
        ):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={experiment}"],
            )
            assert cfg.data.topk == 0
            assert cfg.data.compute_dist_bins is False
            assert cfg.model.use_template_features is False
            assert cfg.model.use_tpl_dist_bins is False

        for experiment in ("frontier_8M", "frontier"):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={experiment}"],
            )
            assert cfg.data.topk == 4
            assert cfg.model.use_template_features is True
