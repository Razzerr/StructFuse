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
    "frontier_8M",
    "frontier",
    "trufor_fusion_with_dist_650M",
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
    for exp in ("frontier", "trufor_fusion_with_dist_650M", "baseline/esm2_650m_trained"):
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
