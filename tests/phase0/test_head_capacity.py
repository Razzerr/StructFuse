"""The larger head of `ablation/trufor_head_capacity` builds, runs and is larger.

Real ContactModel on CPU with the experiment's values, including checkpointing in
training mode, which the default 8M head does not exercise.

Plain-python: `python tests/phase0/test_head_capacity.py`.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.components.contact_model import ContactModel  # noqa: E402
from src.models.components.pair2d_head import AxialAttentionBlock  # noqa: E402
from src.models.components.triangle import TriangleMultiplicativeUpdate  # noqa: E402

BASE = dict(d_esm=32, d_pair=32, width=32, rank=8, fusion_strategy="trufor", fusion_num_heads=4,
            fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 9}, use_tpl_dist_bins=True,
            head_type="axial_tri", alternating_axial=True, triangle_c=8)
SMALL = dict(depth=8, head_num_heads=4, head_num_kv_heads=1)
LARGE = dict(depth=16, head_num_heads=8, head_num_kv_heads=4)


def _inputs(L=14, B=2):
    return dict(
        h_esm=torch.randn(B, L, 32), prior=torch.rand(B, 1, L, L), count=torch.rand(B, 1, L, L),
        rel=torch.rand(B, 14, L, L), esm_contacts=torch.rand(B, 1, L, L),
        pair_mask=torch.ones(B, 1, L, L), tpl_dist_bins=torch.rand(B, 9, L, L))


def _params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_block_pattern_repeats_and_axes_alternate():
    head = ContactModel(**BASE, **LARGE).head
    kinds = ["tri" if isinstance(b, TriangleMultiplicativeUpdate) else "ax" for b in head.blocks]
    assert kinds == ["ax", "ax", "tri", "tri"] * 4, kinds
    axes = [b.axis for b in head.blocks if isinstance(b, AxialAttentionBlock)]
    assert axes == [1, 2] * 4, axes


def test_larger_head_has_more_parameters():
    small, large = _params(ContactModel(**BASE, **SMALL)), _params(ContactModel(**BASE, **LARGE))
    assert large > 1.5 * small, (small, large)


def test_forward_and_backward_with_checkpointing():
    torch.manual_seed(0)
    model = ContactModel(**BASE, **LARGE, use_checkpoint=True)
    model.train()
    logits = model(**_inputs())
    assert logits.shape == (2, 1, 14, 14) and torch.allclose(logits, logits.transpose(-1, -2))
    logits.mean().backward()
    last = model.head.blocks[-1]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in last.parameters())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
