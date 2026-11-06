"""train.py and eval.py must build the SAME model for the same experiment.

`configs/eval.yaml` defaulted to `esm2_t33_650M` while `configs/train.yaml`
defaulted to `esm2_t6_8M`. The 25 `ablation/*` configs pin no backbone, so an
8M-trained ablation composed as 650M under eval.py and its checkpoint could not
load — a wrong-backbone eval that only surfaced as a state_dict error minutes
into a GPU job.

The divergence was invisible because every existing compose test used ONE
entrypoint. These assertions compare the two.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402

CONFIG_DIR = str((ROOT / "configs").resolve())

# One per family: unpinned ablation, pinned 8M, pinned 650M, baselines.
EXPERIMENTS = [
    "ablation/no_templates",
    "ablation/no_triangle",
    "ablation/random_retrieval",
    "ablation/bce_only",
    "ablation/trufor_no_triangle_with_dist",
    "ablation/trufor_no_dist_no_triangle",
    "ablation/trufor_bce_only_with_dist",
    "ablation/trufor_global_alignment",
    "ablation/trufor_cosine_weights",
    "ablation/trufor_head_capacity",
    "frontier_8M",
    "frontier",
    "trufor_fusion_with_dist_650M",
    "trufor_no_templates_650M",
    "trufor_fusion_with_dist_3B",
    "trufor_no_templates_3B",
    "baseline/esm2_650m_trained",
]

# Keys that must agree; anything here changes the shape of the loaded weights.
SHAPE_KEYS = ("esm_model", "d_esm", "depth", "d_pair", "head_type",
              "fusion_strategy", "use_tpl_dist_bins", "use_template_features")


def _compose(config_name: str, experiment: str):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        overrides = [f"experiment={experiment}"]
        if config_name == "eval":
            overrides.append("ckpt_path=/dev/null")
        return compose(config_name=config_name, overrides=overrides)


def test_train_and_eval_build_the_same_model():
    for exp in EXPERIMENTS:
        tr = _compose("train", exp)
        ev = _compose("eval", exp)
        for key in SHAPE_KEYS:
            a, b = tr.model.get(key), ev.model.get(key)
            assert a == b, (
                f"{exp}: model.{key} differs — train={a!r} eval={b!r}. "
                "A checkpoint trained under train.py will not load under eval.py."
            )


def test_unpinned_ablations_get_the_8m_backbone_under_both():
    for exp in ("ablation/no_templates", "ablation/no_triangle"):
        for cfg_name in ("train", "eval"):
            cfg = _compose(cfg_name, exp)
            assert cfg.model.esm_model == "esm2_t6_8M_UR50D", (
                f"{exp} under {cfg_name}.yaml resolved to "
                f"{cfg.model.esm_model!r}; the 8M panel depends on the default."
            )


def test_pinned_650m_experiments_are_unaffected_by_the_default():
    for exp in ("frontier", "trufor_fusion_with_dist_650M",
                "trufor_no_templates_650M", "baseline/esm2_650m_trained"):
        for cfg_name in ("train", "eval"):
            cfg = _compose(cfg_name, exp)
            assert cfg.model.esm_model == "esm2_t33_650M_UR50D", (
                f"{exp} under {cfg_name}.yaml resolved to {cfg.model.esm_model!r} — "
                "these pin `override /model` and must not follow the default."
            )


def test_data_paths_agree_between_entrypoints():
    """A mismatched index or embedding cache is the same class of defect."""
    for exp in ("frontier_8M", "frontier"):
        tr, ev = _compose("train", exp), _compose("eval", exp)
        for key in ("index_dir", "esm_embeddings_dir", "topk", "max_chains_per_cluster"):
            assert tr.data.get(key) == ev.data.get(key), (
                f"{exp}: data.{key} differs — train={tr.data.get(key)!r} "
                f"eval={ev.data.get(key)!r}"
            )


def test_3b_stack_differs_from_650m_only_in_the_backbone_and_its_cache():
    """Retrieval stays on the 650M index, and the 3B control is matched like the 650M one."""
    expected = {"model.esm_model", "model.d_esm", "data.esm_embeddings_dir"}
    assert _config_diff("trufor_fusion_with_dist_650M", "trufor_fusion_with_dist_3B") == expected
    assert _config_diff("trufor_no_templates_650M", "trufor_no_templates_3B") == expected
    assert _config_diff("trufor_fusion_with_dist_3B", "trufor_no_templates_3B") == _config_diff(
        "trufor_fusion_with_dist_650M", "trufor_no_templates_650M")


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

def test_650m_no_template_control_is_matched_to_the_headline():
    """The Stage-G control must differ from the headline ONLY in retrieval.

    A control that also drifts in fusion, backbone, batch size or index path
    stops measuring "templates removed" and starts measuring a mixture — the
    exact confound that made `baseline/esm2_650m_trained` (grouped) unusable as
    a control for a TruFor headline.
    """
    head = _compose("train", "trufor_fusion_with_dist_650M")
    ctrl = _compose("train", "trufor_no_templates_650M")

    must_match_model = ("esm_model", "d_esm", "depth", "d_pair", "head_type",
                        "fusion_strategy", "use_tpl_dist_bins", "use_checkpoint")
    for key in must_match_model:
        assert head.model.get(key) == ctrl.model.get(key), (
            f"model.{key} differs between the 650M headline and its control: "
            f"{head.model.get(key)!r} vs {ctrl.model.get(key)!r}"
        )
    for key in ("batch_size", "index_dir", "esm_embeddings_dir", "crop_size", "min_seq_sep"):
        assert head.data.get(key) == ctrl.data.get(key), (
            f"data.{key} differs between the 650M headline and its control: "
            f"{head.data.get(key)!r} vs {ctrl.data.get(key)!r}"
        )

    # ...and it must actually switch retrieval off, on both sides of the boundary.
    assert head.data.topk > 0 and ctrl.data.topk == 0
    assert head.model.use_template_features is True
    assert ctrl.model.use_template_features is False
    assert ctrl.data.compute_dist_bins is False
    # Width parity: the encoder still sees 2 + 9 channels, zeros materialised in
    # the forward. Flipping this to False would silently change the architecture.
    assert ctrl.model.use_tpl_dist_bins is True


# Keys that name a run rather than configure it.
_NAMING_PREFIXES = ("task_name", "tags", "logger.", "hydra.", "extras.",
                    "paths.output_dir", "paths.log_dir")


def _flat(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_flat(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def _config_diff(a: str, b: str) -> set:
    """Every configuring key whose (unresolved) value differs between two experiments."""
    from omegaconf import OmegaConf
    fa = _flat(OmegaConf.to_container(_compose("train", a), resolve=False))
    fb = _flat(OmegaConf.to_container(_compose("train", b), resolve=False))
    return {k for k in set(fa) | set(fb)
            if not k.startswith(_NAMING_PREFIXES)
            and fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def test_trufor_panel_cells_differ_from_their_reference_only_as_intended():
    """The three TruFor cells added on 2026-10-14 close the selected stack's panel.

    Each must be its grouped counterpart's intervention applied to TruFor, and
    nothing else: a full-config diff, not a list of keys to compare, so a stray
    key (the `fusion_feature_groups: null` the 2025 `trufor_no_dist_no_triangle`
    carried) fails here instead of reaching an audit manifest.
    """
    ref = "ablation/trufor_fusion_with_dist"
    expected = {
        "ablation/trufor_no_triangle_with_dist": {"model.head_type"},
        "ablation/trufor_no_dist_no_triangle": {
            "model.head_type", "model.use_tpl_dist_bins", "data.compute_dist_bins"},
        "ablation/trufor_bce_only_with_dist": {"model.use_tversky", "model.tversky_weight"},
        "ablation/trufor_no_dist": {"model.use_tpl_dist_bins", "data.compute_dist_bins"},
        "ablation/trufor_global_alignment": {"data.alignment"},
        "ablation/trufor_cosine_weights": {"data.template_weighting"},
        "ablation/trufor_head_capacity": {
            "model.depth", "model.head_num_heads", "model.head_num_kv_heads", "model.use_checkpoint"},
    }
    for cell, keys in expected.items():
        got = _config_diff(ref, cell)
        assert got == keys, f"{cell} vs {ref}: differs in {sorted(got)}, expected {sorted(keys)}"
    # The 2x2 edge that only the new cell closes.
    got = _config_diff("ablation/trufor_no_dist", "ablation/trufor_no_dist_no_triangle")
    assert got == {"model.head_type"}, sorted(got)
    # Same interventions as the grouped panel they are compared with.
    grouped = {
        "ablation/no_triangle": "ablation/trufor_no_triangle_with_dist",
        "ablation/tpl_contact_only": "ablation/trufor_no_dist_no_triangle",
        "ablation/bce_only": "ablation/trufor_bce_only_with_dist",
    }
    for g_cell, t_cell in grouped.items():
        assert _config_diff("frontier_8M", g_cell) == _config_diff(ref, t_cell), (g_cell, t_cell)


if __name__ == "__main__":
    sys.exit(_run_all())
