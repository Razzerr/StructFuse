"""Bit-reproducibility smoke test for StructFuse training pipeline.

Runs `src/train.py` TWICE with an identical short configuration (frontier_8M,
seed=42, ltb=0.005, max_epochs=1, deterministic=true, num_workers=0, W&B
disabled) and diffs the resulting `train_metrics.json` files. Any divergence
above tolerance is reported as a leaking-randomness bug, with the offending
keys printed.

Required to pass before launching the 3-seed headline H1 — without bit
reproducibility the seed-stratified mean ± std is meaningless.

Usage:
    # Full run (fires two ~1-min training jobs, requires GPU on server):
    python scripts/smoke_repro_test.py

    # Compare two existing runs without re-launching:
    python scripts/smoke_repro_test.py --skip-runs \\
        --run1 .temp/repro/run1 --run2 .temp/repro/run2

    # Custom tolerance (default 1e-6):
    python scripts/smoke_repro_test.py --tolerance 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPRO_ROOT = Path(".temp/repro")
DEFAULT_TOLERANCE = 1e-6
TRAIN_OVERRIDES = (
    "experiment=frontier_8M",
    "seed=42",
    "trainer.limit_train_batches=0.005",
    "trainer.max_epochs=1",
    "trainer.deterministic=true",
    "data.num_workers=0",
    "test=false",
)


def _run_training(out_dir: Path) -> int:
    """Launch one training run with `paths.output_dir=<out_dir>`. Returns exit code.

    W&B is disabled via WANDB_MODE=disabled env var (Hydra `logger=null` is rejected
    by config group override validation, and the project ships only `logger=wandb`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "src/train.py",
        f"paths.output_dir={out_dir}",
        f"task_name=smoke_repro_{out_dir.name}",
        *TRAIN_OVERRIDES,
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    print(f"\n>>> Launching: WANDB_MODE=disabled {' '.join(cmd)}")
    return subprocess.run(cmd, env=env, check=False).returncode


def _load_metrics(run_dir: Path) -> dict:
    f = run_dir / "train_metrics.json"
    if not f.exists():
        raise SystemExit(f"Missing train_metrics.json in {run_dir} — did training crash?")
    return json.loads(f.read_text())


def _diff(m1: dict, m2: dict, tolerance: float) -> list[tuple[str, Optional[float], Optional[float], float]]:
    """Return list of (key, val1, val2, abs_delta) for every divergent key."""
    keys = sorted(set(m1) | set(m2))
    bad = []
    for k in keys:
        v1, v2 = m1.get(k), m2.get(k)
        if v1 is None or v2 is None:
            if v1 != v2:
                bad.append((k, v1, v2, float("inf")))
            continue
        try:
            delta = abs(float(v1) - float(v2))
        except (TypeError, ValueError):
            if v1 != v2:
                bad.append((k, v1, v2, float("inf")))
            continue
        if delta > tolerance:
            bad.append((k, v1, v2, delta))
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--skip-runs", action="store_true", help="Skip launching training; only diff existing runs")
    p.add_argument("--run1", type=Path, default=REPRO_ROOT / "run1")
    p.add_argument("--run2", type=Path, default=REPRO_ROOT / "run2")
    p.add_argument("--keep-old", action="store_true", help="Don't wipe run1/run2 dirs before launching")
    args = p.parse_args()

    if not args.skip_runs:
        if not args.keep_old:
            for d in (args.run1, args.run2):
                if d.exists():
                    shutil.rmtree(d)
        rc1 = _run_training(args.run1)
        rc2 = _run_training(args.run2)
        if rc1 != 0 or rc2 != 0:
            raise SystemExit(f"Training failed (rc1={rc1}, rc2={rc2})")

    m1 = _load_metrics(args.run1)
    m2 = _load_metrics(args.run2)

    bad = _diff(m1, m2, args.tolerance)
    if not bad:
        print(f"\n✓ PASS — all {len(m1)} metrics match within tolerance {args.tolerance:g}")
        sys.exit(0)

    print(f"\n✗ FAIL — {len(bad)} metric(s) diverge above tolerance {args.tolerance:g}:")
    print(f"  {'metric':<40} {'run1':>16} {'run2':>16} {'|delta|':>16}")
    print(f"  {'-'*40} {'-'*16} {'-'*16} {'-'*16}")
    for k, v1, v2, delta in bad[:30]:
        v1_s = f"{v1:.8g}" if isinstance(v1, (int, float)) else str(v1)
        v2_s = f"{v2:.8g}" if isinstance(v2, (int, float)) else str(v2)
        d_s = f"{delta:.2e}" if delta != float("inf") else "inf"
        print(f"  {k:<40} {v1_s:>16} {v2_s:>16} {d_s:>16}")
    if len(bad) > 30:
        print(f"  ... ({len(bad) - 30} more)")
    print("\nLikely causes:")
    print("  - dataloader worker RNG drift (num_workers != 0 in repro test)")
    print("  - non-deterministic CUDA op (check trainer.deterministic=true is honoured)")
    print("  - timing-dependent code (e.g. wall-clock seeding somewhere)")
    print("  - 3rd-party library RNG not re-seeded by L.seed_everything")
    sys.exit(1)


if __name__ == "__main__":
    main()
