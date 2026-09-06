"""Evaluation batch size is separate from the training one.

The model is padding-dependent: `PairFeatures` normalises with `InstanceNorm2d`
over the whole L x L map (padding included, in eval() too) and axial attention
calls SDPA without `attn_mask`. So a chain's prediction depends on which chains
share its batch, and two runs over different dataset compositions are not
comparable. `eval_batch_size=1` removes padding from val/test — but it must NOT
change the training batch, or the effective batch after gradient accumulation
moves with it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402

CONFIG_DIR = str((ROOT / "configs").resolve())


def _cfg(**overrides):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        ov = ["experiment=frontier_8M"] + [f"{k}={v}" for k, v in overrides.items()]
        return compose(config_name="train", overrides=ov)


def test_default_is_none_so_behaviour_is_unchanged():
    cfg = _cfg()
    assert cfg.data.eval_batch_size is None, (
        "default must stay null; anything else silently changes every existing run"
    )


def test_eval_override_does_not_touch_the_training_batch():
    cfg = _cfg(**{"data.eval_batch_size": 1})
    assert cfg.data.eval_batch_size == 1
    assert cfg.data.batch_size == 12, (
        f"training batch moved to {cfg.data.batch_size} — the effective batch after "
        "accumulate_grad_batches would change with it"
    )
    assert cfg.trainer.accumulate_grad_batches == 4


def test_datamodule_falls_back_to_batch_size_when_unset():
    import types
    src = (ROOT / "src/data/contact_lit_datamodule.py").read_text()
    # The fallback is one line; assert on it directly rather than constructing a
    # DataModule (which would need the real dataset on disk).
    assert "int(eval_batch_size) if eval_batch_size else self.batch_size" in src


def test_only_val_and_test_loaders_use_the_eval_batch():
    src = (ROOT / "src/data/contact_lit_datamodule.py").read_text()
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "self.eval_batch_size" not in line or "self.eval_batch_size =" in line:
            continue
        window = "\n".join(lines[max(0, i - 25):i])
        assert ("def val_dataloader" in window or "def test_dataloader" in window), (
            f"line {i+1} uses eval_batch_size outside val/test_dataloader:\n{line}"
        )
    # and the train path must still use the training batch
    train_start = src.index("def train_dataloader")
    train_body = src[train_start:train_start + 1200]
    assert "self.batch_size" in train_body
    assert "self.eval_batch_size" not in train_body


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
