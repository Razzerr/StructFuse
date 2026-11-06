"""Per-template attention: maps kept apart in the data, attention over them in the model.

Real `PriorBuilder.build_one`, `collate_padded`, `template_stack`, fusion and
ContactModel on CPU; only the FAISS lookup and the parasail alignment are replaced.

Pins: each template keeps its own maps and quality, absent templates change nothing and
give no NaN, the template order cannot matter, the averaged maps are untouched by the
option, a run without the maps stops, and the no-template control ignores any input.

Plain-python: `python tests/phase0/test_template_attention.py`.
"""

import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import parasail  # noqa: F401
except ModuleNotFoundError:
    stub_p = types.ModuleType("parasail")
    stub_p.matrix_create = lambda *a, **k: None
    sys.modules["parasail"] = stub_p
try:
    import blosum  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("blosum")
    stub.BLOSUM = lambda *a, **k: defaultdict(lambda: defaultdict(int))
    sys.modules["blosum"] = stub

from src.data.components.dataset import PriorBuilder, collate_padded  # noqa: E402
from src.data.utils import align  # noqa: E402
from src.data.utils.template_selection import (  # noqa: E402
    SAME_PROTEIN_COVERAGE,
    SAME_PROTEIN_IDENTITY,
    alignment_quality,
)
from src.models.components.contact_model import ContactModel  # noqa: E402
from src.models.components.fusion_strategies import (  # noqa: E402
    PerTemplateTruForFusion,
    template_stack,
)

L = 30
D = 32
CH = 2 + 9 + 2
CFG = dict(d_esm=D, d_pair=D, width=D, depth=4, rank=8, fusion_strategy="trufor_per_template",
           fusion_num_heads=4, fusion_feature_groups={"tpl_contact": 2, "tpl_dist": 9},
           use_tpl_dist_bins=True, head_type="axial_tri", head_num_heads=4, head_num_kv_heads=1,
           alternating_axial=True, triangle_c=8)


def _write(d, stem, seq, far=False, mask=None):
    ca = np.zeros((L, 3), np.float32)
    ca[:, 0] = np.arange(L) * 100.0
    ca[2], ca[20] = (0, 0, 0), ((25, 0, 0) if far else (3, 0, 0))
    contact = np.zeros((L, L), np.uint8)
    contact[2, 20] = contact[20, 2] = 0 if far else 1
    path = d / f"{stem}.npz"
    np.savez_compressed(path, seq=np.array(seq, dtype=object), coords=ca, contact=contact,
                        mask=np.ones(L, np.uint8) if mask is None else mask, L=np.int32(L))
    return path


def _builder(paths, hits, k, per_template=True):
    b = PriorBuilder.__new__(PriorBuilder)
    b.faiss_index = types.SimpleNamespace(topk_precomputed=lambda *a, **kw: hits)
    b.topk, b.use_blosum, b.only_positive_transfer = k, False, False
    b.min_seq_sep, b.min_template_similarity, b.random_retrieval = 0, 0.0, False
    b.compute_dist_bins, b._needs_coords, b._tpl_atom = True, True, None
    b.same_protein_identity, b.same_protein_coverage = SAME_PROTEIN_IDENTITY, SAME_PROTEIN_COVERAGE
    b.alignment, b.template_weighting, b.per_template = "local", "alignment", per_template
    b._id_to_npz, b._tpl_cache, b._max_tpl_cache = paths, {}, 100
    return b


def _build(b, seq="A" * L):
    saved = align.local_alignment
    align.local_alignment = lambda *a, **k: ("", "", np.arange(L), np.arange(L))
    try:
        return b.build_one("q_A", seq, 0, L)
    finally:
        align.local_alignment = saved


def _two(tmp, k=3, per_template=True):
    d = Path(tmp)
    paths = {"1abc_A": _write(d, "1abc_A", "A" * 15 + "C" * 15),
             "2xyz_A": _write(d, "2xyz_A", "A" * 5 + "C" * 25, far=True)}
    return _builder(paths, [("1abc_A", 0.9), ("2xyz_A", 0.9)], k, per_template)


def test_each_template_keeps_its_own_maps():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(_two(tmp))
    assert out["tpl_present"].tolist() == [True, True, False]
    assert out["tpl_prior"].shape == (3, L, L) and out["tpl_dbin"].shape == (3, L, L)
    # Near in the first template (bin 1, 0-4 A), far in the second (bin 8, >= 20 A).
    assert out["tpl_dbin"][0, 2, 20] == 1 and out["tpl_dbin"][1, 2, 20] == 8, out["tpl_dbin"][:, 2, 20]
    assert out["tpl_prior"][0, 2, 20] == 1 and out["tpl_prior"][1, 2, 20] == 0
    assert (out["tpl_prior"][2] == 0).all() and (out["tpl_dbin"][2] == 0).all()
    assert np.allclose(out["tpl_quality"][0], [0.5, 1.0]) and np.allclose(out["tpl_quality"][1], [1 / 6, 1.0])
    assert (out["tpl_quality"][2] == 0).all()


def test_averaged_maps_do_not_depend_on_the_option():
    with tempfile.TemporaryDirectory() as tmp:
        on, off = _build(_two(tmp)), _build(_two(tmp, per_template=False))
    assert "tpl_prior" not in off
    for key in ("prior", "count", "dist_bins"):
        assert np.array_equal(on[key], off[key]), key


def test_no_hits_give_an_empty_stack():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(_builder({}, [], 4))
    assert out["tpl_present"].tolist() == [False] * 4 and out["tpl_prior"].shape == (4, L, L)


def test_option_needs_the_distance_bins():
    try:
        PriorBuilder("unused", compute_dist_bins=False, per_template=True)
    except ValueError as exc:
        assert "compute_dist_bins" in str(exc)
    else:
        raise AssertionError("accepted")


def test_quality_of_nothing_aligned_is_zero():
    assert alignment_quality("AAAA", "AAAA", np.full(4, -1)) == (0.0, 0.0)


def _item(pid, n):
    return {"pid": pid, "seq": "A" * n, "contact": np.zeros((n, n), np.float32),
            "mask": np.ones(n, np.float32), "coords": np.zeros((n, 3), np.float32), "L": n,
            "subset": "gold", "cluster_id": 1}


def test_collate_pads_the_stack_to_the_batch():
    class Fake:
        topk, compute_dist_bins, per_template = 2, True, True

        def build_one(self, pid, seq, s, e, filter_holdout=True):
            n = e - s
            return {"prior": np.zeros((n, n), np.float32), "count": np.zeros((n, n), np.float32),
                    "dist_bins": np.zeros((9, n, n), np.float32),
                    "n_templates_retrieved": 1, "best_tpl_sim": 0.5,
                    "tpl_prior": np.full((2, n, n), n, np.float32), "tpl_dbin": np.full((2, n, n), 3, np.int8),
                    "tpl_quality": np.array([[0.5, 1.0], [0, 0]], np.float32), "tpl_present": np.array([True, False])}

    out = collate_padded([_item("a_A", 25), _item("b_A", 20)], crop_mode="none", prior_builder=Fake())
    assert out["tpl_prior"].shape == (2, 2, 25, 25) and out["tpl_dbin"].dtype == torch.int8
    assert out["tpl_prior"][1, 0, 19, 19] == 20 and out["tpl_prior"][1, 0, 20, 20] == 0, "padding stays zero"
    assert out["tpl_dbin"][0, 0, 24, 24] == 3 and out["tpl_dbin"][1, 0, 24, 24] == 0
    assert out["tpl_present"].tolist() == [[True, False], [True, False]]


def test_stack_channels_and_what_is_zeroed():
    n = 4
    prior = torch.zeros(1, 2, n, n)
    prior[0, 0, 0, 1] = 0.7
    dbin = torch.full((1, 2, n, n), 3, dtype=torch.int8)
    quality = torch.tensor([[[0.5, 0.25], [0.9, 0.9]]])
    present = torch.tensor([[True, False]])
    mask = torch.ones(1, n, n)
    mask[0, 3, :] = 0
    s = template_stack(prior, dbin, quality, present, mask, n_dist=9)
    assert s.shape == (1, 2, 13, n, n)
    assert s[0, 0, 0, 0, 1] == 0.7 and s[0, 0, 1, 0, 1] == 1 and s[0, 0, 1, 1, 1] == 0
    assert s[0, 0, 2 + 3, 1, 1] == 1 and s[0, 0, 2 + 4, 1, 1] == 0
    assert s[0, 0, 11, 1, 1] == 0.5 and s[0, 0, 12, 1, 1] == 0.25
    assert (s[0, 0, :, 3, :] == 0).all(), "padding"
    assert (s[0, 1] == 0).all(), "absent template"


def _fusion_inputs(K=3, n=12):
    g = torch.Generator().manual_seed(0)
    return dict(
        pair_feat=torch.randn(2, D, n, n, generator=g), prior=torch.zeros(2, 1, n, n), count=torch.zeros(2, 1, n, n),
        rel=torch.rand(2, 14, n, n, generator=g), esm_contacts=torch.rand(2, 1, n, n, generator=g),
        tpl_stack=torch.rand(2, K, CH, n, n, generator=g), tpl_present=torch.ones(2, K, dtype=torch.bool))


def _fusion():
    torch.manual_seed(0)
    return PerTemplateTruForFusion(d_pair=D, d_rel=14, num_heads=4, tpl_dist_channels=9).eval()


def test_template_order_cannot_matter():
    f, x = _fusion(), _fusion_inputs()
    with torch.no_grad():
        a = f(**x)
        y = dict(x, tpl_stack=x["tpl_stack"][:, [2, 0, 1]])
        assert torch.allclose(a, f(**y), atol=1e-5)
        z = dict(x, tpl_stack=x["tpl_stack"] * 1.5)
        assert not torch.allclose(a, f(**z)), "the templates must reach the output"


def test_absent_template_changes_nothing():
    f, x = _fusion(), _fusion_inputs()
    with torch.no_grad():
        present = x["tpl_present"].clone()
        present[:, 2] = False
        ref = f(**dict(x, tpl_stack=x["tpl_stack"][:, :2], tpl_present=present[:, :2]))
        garbage = x["tpl_stack"].clone()
        garbage[:, 2] = 1e3
        assert torch.allclose(ref, f(**dict(x, tpl_stack=garbage, tpl_present=present)), atol=1e-4)


def test_no_template_present_stays_finite_with_finite_gradients():
    f = _fusion().train()
    x = _fusion_inputs()
    out = f(**dict(x, tpl_present=torch.zeros(2, 3, dtype=torch.bool)))
    assert out.isfinite().all()
    out.mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in f.parameters())


def test_run_without_the_maps_stops():
    f, x = _fusion(), _fusion_inputs()
    try:
        f(**dict(x, tpl_stack=None, tpl_present=None))
    except ValueError as exc:
        assert "per_template_features" in str(exc)
    else:
        raise AssertionError("ran without per-template maps")


def _model_inputs(n=14):
    g = torch.Generator().manual_seed(1)
    return dict(
        h_esm=torch.randn(2, n, D, generator=g), prior=torch.rand(2, 1, n, n, generator=g),
        count=torch.rand(2, 1, n, n, generator=g), rel=torch.rand(2, 14, n, n, generator=g),
        esm_contacts=torch.rand(2, 1, n, n, generator=g), pair_mask=torch.ones(2, 1, n, n),
        tpl_dist_bins=torch.rand(2, 9, n, n, generator=g), tpl_stack=torch.rand(2, 4, CH, n, n, generator=g),
        tpl_present=torch.ones(2, 4, dtype=torch.bool))


def test_full_model_trains_on_cpu():
    torch.manual_seed(0)
    model = ContactModel(**CFG).train()
    logits = model(**_model_inputs())
    assert logits.shape == (2, 1, 14, 14) and torch.allclose(logits, logits.transpose(-1, -2))
    logits.mean().backward()
    assert model.head.fusion.tpl_key.weight.grad.abs().sum() > 0
    assert model.head.fusion.tpl_query.weight.grad.abs().sum() > 0


def test_no_template_control_ignores_every_template_input():
    torch.manual_seed(0)
    model = ContactModel(**CFG, use_template_features=False).train()
    bns = [m for m in model.head.fusion.template_encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns), "BatchNorm statistics must stay pinned"
    model.eval()
    x = _model_inputs()
    with torch.no_grad():
        ref = model(**x)
        changed = dict(x, prior=x["prior"] * 9, count=x["count"] * 9, tpl_dist_bins=x["tpl_dist_bins"] * 9,
                       tpl_stack=x["tpl_stack"] * 9)
        assert torch.equal(ref, model(**changed))
        assert torch.equal(ref, model(**dict(x, tpl_stack=None, tpl_present=None)))


def test_lit_module_passes_the_stack_to_the_model():
    """Source-level: `_step` is too heavy to drive here, and this wiring is what a missing
    argument would break. `_template_stack` itself runs on the server suite."""
    src = (ROOT / "src/models/contact_lit_module.py").read_text()
    step = src[src.index("    def _step("):]
    call = step[step.index("logits = self.net("):step.index("valid_mask = valid.squeeze(1)")]
    assert "tpl_stack=tpl_stack" in call and "tpl_present=tpl_present" in call, call
    assert "self._template_stack(batch, pair_mask)" in step


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
