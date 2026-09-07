#!/usr/bin/env python3
"""Is a wrong contact prediction wrong, or just displaced?

Motivation: eyeballing validation figures suggests the model often puts its mass
one or two residues away from the true contact. If that is systematic rather than
anecdotal it points somewhere specific — most likely the NW alignment behind the
template prior, where a one-residue shift transfers the contact one residue off —
and it would be a different fix from "the model is inaccurate".

Method. For each chain take the top-K ranked pairs (K = nominal length, the same
K as P@L) and score them three ways:
  exact   a predicted pair is a true contact
  tol=1   a true contact exists within Chebyshev distance 1
  tol=2   ... within distance 2

**Tolerance inflates any predictor**, so the numbers alone prove nothing: with
tolerance d each true contact covers up to (2d+1)^2 cells. The script therefore
computes a matched NULL by scoring K pairs drawn uniformly from the same
admissible set, and reports the lift over that null. A displacement effect is
real only if the observed gain clears the null's gain.

Also reported: the median Chebyshev distance from each MISSED true contact to the
nearest of the top-K predictions. If the model were simply wrong, that distance
would be large; if it is displaced, it sits at 1-2.

Usage:
  python scripts/near_miss_analysis.py --ckpt <path> --experiment frontier_8M \\
      --n-chains 400 --out .temp/near_miss.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _top_k_pairs(prob: np.ndarray, mask: np.ndarray, k: int):
    """Indices of the k highest-probability admissible pairs, i<j only."""
    iu = np.triu_indices_from(prob, k=1)
    valid = mask[iu] > 0
    scores = prob[iu][valid]
    rows, cols = iu[0][valid], iu[1][valid]
    if scores.size == 0:
        return np.empty(0, int), np.empty(0, int)
    k = min(k, scores.size)
    sel = np.argpartition(-scores, k - 1)[:k]
    return rows[sel], cols[sel]


def _within(pred_r, pred_c, true_r, true_c, tol: int) -> np.ndarray:
    """For each predicted pair, is some true contact within Chebyshev tol?"""
    if true_r.size == 0 or pred_r.size == 0:
        return np.zeros(pred_r.size, bool)
    dr = np.abs(pred_r[:, None] - true_r[None, :])
    dc = np.abs(pred_c[:, None] - true_c[None, :])
    return (np.maximum(dr, dc) <= tol).any(axis=1)


def analyse_chain(prob, contact, mask, k, rng):
    iu = np.triu_indices_from(prob, k=1)
    admissible = mask[iu] > 0
    n_adm = int(admissible.sum())
    tr, tc = np.where(np.triu(contact * (mask > 0), k=1) > 0)
    if n_adm == 0 or tr.size == 0:
        return None

    pr, pc = _top_k_pairs(prob, mask, k)
    rows_a, cols_a = iu[0][admissible], iu[1][admissible]
    pick = rng.choice(n_adm, size=min(k, n_adm), replace=False)
    nr, nc = rows_a[pick], cols_a[pick]

    out = {"n_admissible": n_adm, "n_true": int(tr.size), "k": int(pr.size)}
    for tol in (0, 1, 2):
        out[f"hit_tol{tol}"] = float(_within(pr, pc, tr, tc, tol).mean())
        out[f"null_tol{tol}"] = float(_within(nr, nc, tr, tc, tol).mean())

    # distance from each missed true contact to the nearest top-K prediction
    if pr.size:
        d = np.maximum(np.abs(tr[:, None] - pr[None, :]),
                       np.abs(tc[:, None] - pc[None, :])).min(axis=1)
        missed = d > 0
        out["median_dist_missed"] = float(np.median(d[missed])) if missed.any() else 0.0
        out["frac_missed_within_2"] = float((d[missed] <= 2).mean()) if missed.any() else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--experiment", default="frontier_8M")
    ap.add_argument("--n-chains", type=int, default=400)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".temp/near_miss.tsv")
    args = ap.parse_args()

    import hydra
    import pandas as pd
    import torch
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3",
                               config_dir=str(Path.cwd() / "configs")):
        cfg = compose(config_name="eval",
                      overrides=[f"experiment={args.experiment}",
                                 f"ckpt_path={args.ckpt}",
                                 "data.eval_batch_size=1"])

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("test" if args.split == "test" else "validate")
    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval().cuda()

    rng = np.random.default_rng(args.seed)
    rows = []
    # ContactLitModule has no forward(); predictions come from _step, which
    # assembles the ESM/template inputs itself. This is the same path
    # validation_step and test_step use, so the probabilities here are exactly
    # the ones the reported metrics are computed from.
    with torch.no_grad():
        for batch in loader:
            _, viz = model._step(batch, stage="test", return_visualization=True)
            prob = viz["prob"].detach().squeeze().float().cpu().numpy()
            contact = viz["contact"].detach().squeeze().cpu().numpy()
            mask = viz["valid_mask"].detach().squeeze().cpu().numpy()
            k = int(batch["seq_len"][0].item())
            r = analyse_chain(prob, contact, mask, k, rng)
            if r:
                r["sample_id"] = batch["pid"][0]
                rows.append(r)
            if len(rows) >= args.n_chains:
                break

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.6f")

    print(f"chains analysed: {len(df)}   (K = nominal length, pairs i<j, long-range mask)\n")
    print(f"{'tol':>4} {'observed':>9} {'null':>9} {'lift':>9}")
    for tol in (0, 1, 2):
        o, n = df[f"hit_tol{tol}"].mean(), df[f"null_tol{tol}"].mean()
        print(f"{tol:>4} {o:9.4f} {n:9.4f} {o - n:+9.4f}")
    print(f"\ngain from tolerance 0 -> 1: observed {df['hit_tol1'].mean() - df['hit_tol0'].mean():+.4f}"
          f"   null {df['null_tol1'].mean() - df['null_tol0'].mean():+.4f}")
    print("If the observed gain does not clear the null gain, the near misses are "
          "the arithmetic of tolerance, not displacement.")
    print(f"\nmedian Chebyshev distance, missed true contact -> nearest top-K pred: "
          f"{df['median_dist_missed'].median():.1f}")
    print(f"fraction of missed true contacts within distance 2: "
          f"{df['frac_missed_within_2'].mean():.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
