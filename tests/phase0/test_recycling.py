"""Recycling in Pair2DHead: real ContactModel on CPU.

Pins: off means the same parameters and the same output as before, on starts identical
because the projection is zero-initialised, evaluation always runs every pass, training
draws the count, extra passes carry no gradient, and the output does feed back once the
projection has learned something.

Plain-python: `python tests/phase0/test_recycling.py`.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.components.contact_model import ContactModel  # noqa: E402

CFG = dict(d_esm=32, d_pair=32, width=32, depth=4, rank=8, fusion_strategy="trufor", fusion_num_heads=4,
           fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 9}, use_tpl_dist_bins=True,
           head_type="axial_tri", head_num_heads=4, head_num_kv_heads=1, alternating_axial=True, triangle_c=8)


def _inputs(L=14, B=2):
    # A local generator: the global one decides how many passes training draws.
    g = torch.Generator().manual_seed(1)
    return dict(
        h_esm=torch.randn(B, L, 32, generator=g), prior=torch.rand(B, 1, L, L, generator=g),
        count=torch.rand(B, 1, L, L, generator=g), rel=torch.rand(B, 14, L, L, generator=g),
        esm_contacts=torch.rand(B, 1, L, L, generator=g), pair_mask=torch.ones(B, 1, L, L),
        tpl_dist_bins=torch.rand(B, 9, L, L, generator=g))


def _model(n, seed=0):
    torch.manual_seed(seed)
    return ContactModel(**CFG, recycling_iters=n)


def _count_passes(model, inputs):
    calls, grads = [], []
    first = model.head.blocks[0]

    def hook(_m, _in, out):
        calls.append(1)
        grads.append(out.requires_grad)

    h = first.register_forward_hook(hook)
    try:
        model(**inputs)
    finally:
        h.remove()
    return len(calls), grads


def test_off_adds_nothing():
    plain = _model(0)
    assert not any("recycle" in k for k in plain.state_dict()), "no new parameters when off"
    assert _count_passes(plain.eval(), _inputs())[0] == 1


def test_starts_identical_to_the_model_without_it():
    plain, recycled = _model(0).eval(), _model(3).eval()
    missing, unexpected = recycled.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all("recycle" in k for k in missing), (missing, unexpected)
    with torch.no_grad():
        assert torch.allclose(plain(**_inputs()), recycled(**_inputs()), atol=1e-6)


def test_evaluation_runs_every_pass_and_feeds_back():
    model = _model(3).eval()
    n, grads = _count_passes(model, _inputs())
    assert n == 4, n
    with torch.no_grad():
        before = model(**_inputs()).clone()
        torch.nn.init.normal_(model.head.recycle_proj.weight, std=0.1)
        after = model(**_inputs())
    assert not torch.allclose(before, after), "the projection must reach the output"
    assert torch.allclose(after, model(**_inputs())), "evaluation must be deterministic"


def test_training_draws_the_count_and_only_the_last_pass_has_gradient():
    model = _model(3).train()
    seen = set()
    for seed in range(40):
        torch.manual_seed(seed)
        n, grads = _count_passes(model, _inputs())
        seen.add(n)
        assert grads[-1] is True and not any(grads[:-1]), grads
    assert seen == {1, 2, 3, 4}, seen


def test_gradient_reaches_the_projection_and_stays_finite():
    model = _model(1).train()
    torch.nn.init.normal_(model.head.recycle_proj.weight, std=0.1)
    for seed in range(10):
        torch.manual_seed(seed)
        model.zero_grad()
        model(**_inputs()).mean().backward()
        grad = model.head.recycle_proj.weight.grad
        if grad is not None and grad.abs().sum() > 0:
            assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
            return
    raise AssertionError("the projection never received a gradient")


def test_checkpointing_gives_the_same_pass_count():
    model = ContactModel(**CFG, recycling_iters=2, use_checkpoint=True).eval()
    assert _count_passes(model, _inputs())[0] == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
