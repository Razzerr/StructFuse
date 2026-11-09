"""src/train.py must seed the global generators for seed 0 too.

The guard was `if cfg.get("seed"):`, so seed 0 skipped `L.seed_everything` and one
2026-09 650M run trained with unseeded generators. This test runs the guard from
train.py on a config with seed 0 and fails if the call is skipped.

Plain-python: `python tests/phase0/test_seed_guard.py`.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _guard_seeds(seed) -> bool:
    """Evaluate the `if` that wraps `L.seed_everything` in train.py for one seed value."""
    tree = ast.parse((ROOT / "src" / "train.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "seed_everything" in ast.unparse(node.body[0]):
            cfg = {"seed": seed}
            return bool(eval(compile(ast.Expression(node.test), "train.py", "eval"), {"cfg": cfg}))
    raise AssertionError("no `if` around L.seed_everything found in src/train.py")


def test_seed_zero_is_seeded():
    assert _guard_seeds(0), "seed 0 skips L.seed_everything"


def test_other_seeds_are_seeded_and_missing_seed_is_not():
    assert _guard_seeds(42) and _guard_seeds(1337)
    assert not _guard_seeds(None)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
