"""The TruFor no-template control: same architecture, no retrieval information.

Dropping the template stream would change TruFor's cross-attention and answer a
different question, so the architecture and its parameters are kept and the
structural inputs are forced to constant zeros.

That is only sound once the template encoder's BatchNorm is pinned. A BN fed
constant zeros collapses its running variance (~1e-16, measured 2026-06-12) and
then reports one thing in train mode and another in eval — the defect that
invalidated the b0cfw0w8 / kwm8qq4p controls. Statistics are fixed at mean 0 /
var 1 and never update; affine weights, convolutions and cross-attention stay
trainable, so parameterisation is preserved.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.components.pair2d_head import Pair2DHead  # noqa: E402

KW = dict(d_pair=16, width=16, depth=2, head_type="axial",
          fusion_strategy="trufor", use_tpl_dist_bins=True,
          fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 9})
B, L, REL_CH = 2, 12, 14   # rel_ch default in Pair2DHead


def _head(use_tpl: bool):
    torch.manual_seed(0)
    return Pair2DHead(use_template_features=use_tpl, **KW)


def _inputs(scale=0.0):
    torch.manual_seed(1)
    return dict(
        pair_feat=torch.randn(B, KW["d_pair"], L, L),
        prior=torch.randn(B, 1, L, L) * scale,
        count=torch.randn(B, 1, L, L) * scale,
        rel=torch.randn(B, REL_CH, L, L),
        esm_contacts=torch.randn(B, 1, L, L),
        tpl_dist_bins=torch.randn(B, 9, L, L) * scale,
    )


def test_arbitrary_template_input_does_not_change_predictions():
    h = _head(False).eval()
    with torch.no_grad():
        a = h(**_inputs(scale=0.0))
        b = h(**_inputs(scale=37.0))
    assert torch.equal(a, b), f"max |d| = {(a - b).abs().max():.3e}"


def test_missing_distance_bins_are_materialised():
    """collate emits no tpl_dist_bins when the prior builder is off."""
    h = _head(False).eval()
    kw = _inputs(); kw["tpl_dist_bins"] = None
    with torch.no_grad():
        out = h(**kw)
    assert out.shape == (B, 1, L, L)


def test_bn_statistics_never_move_during_training():
    h = _head(False)
    bns = h._frozen_template_bns
    assert bns, "no BatchNorm was pinned in the template encoder"
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    h.train()                       # Lightning does this every epoch
    assert all(not m.training for m in bns), "train() re-enabled the statistics"
    opt = torch.optim.SGD([p for p in h.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        opt.zero_grad(); h(**_inputs()).sum().backward(); opt.step()
    for m, (mu, var) in zip(bns, before):
        assert torch.equal(m.running_mean, mu) and torch.equal(m.running_var, var)
        assert int(m.num_batches_tracked) == 0


def test_encoder_agrees_between_train_and_eval_including_after_reload():
    h = _head(False)
    x = torch.zeros(B, 11, L, L)
    h.train()
    with torch.no_grad():
        a = h.fusion.template_encoder(x)
    h.eval()
    with torch.no_grad():
        b = h.fusion.template_encoder(x)
    assert torch.allclose(a, b, atol=1e-6), f"train/eval differ: {(a-b).abs().max():.3e}"

    h2 = _head(False)
    h2.load_state_dict(h.state_dict())
    h2.train()
    with torch.no_grad():
        c = h2.fusion.template_encoder(x)
    assert torch.allclose(a, c, atol=1e-6), "statistics not pinned after reload"


def test_parameters_stay_trainable():
    h = _head(False)
    names = [n for n, p in h.fusion.template_encoder.named_parameters() if p.requires_grad]
    assert names, "the control must preserve parameterisation, not freeze it"


def test_full_trufor_path_is_unchanged():
    h = _head(True).eval()
    with torch.no_grad():
        a = h(**_inputs(scale=1.0))
        b = h(**_inputs(scale=5.0))
    assert not torch.equal(a, b), "the enabled path must still use template inputs"
    assert not h._frozen_template_bns, "nothing may be pinned when templates are on"
    h.train()   # the enabled path must follow train()/eval() normally
    for m in h.fusion.template_encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert m.training, "BN must resume training on the enabled path"


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
