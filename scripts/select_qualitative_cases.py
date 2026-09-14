"""Deterministic, pre-registered selection of qualitative contact-map cases.

The paper's Figure 4 shows a handful of chains as maps (ground truth, template
prior, matched no-template prediction, StructFuse prediction). A hand-picked
set invites the cherry-picking objection; this script replaces the hand with a
rule fixed BEFORE any map is looked at, and it runs on the same per-chain
tables the paper's statistics come from.

Unit of selection is the sequence FAMILY, not the chain — the same estimand
as every headline number:

1. For a category, take its population: all long-defined evaluated chains of
   the seed-42 pair (StructFuse vs matched no-template), restricted to the
   category's identity regime when it has one. Regimes are Table 12's coarse
   strata of the TOP-RANKED template's crop identity, left-closed
   (`<0.30`, `[0.30, 0.90)`, `>=0.90`) — the same closure as analysis H.
2. Average the paired gain (StructFuse - control, `--metric`) per family over
   the family's chains in that population. That per-family mean is the quantity
   Table 12c averages; the per-family list is what the target is taken from.
3. The target is a statistic of the per-family means: the maximum
   ("largest gain"), the minimum ("failure"), or the MEAN ("near-mean-gain
   case of the regime") — the mean of per-family means is exactly the regime's
   reported cluster-balanced gain (Table 12c), so such a case is the family
   that sits at the number the table prints. It represents the regime's MEAN
   DELTA, not necessarily its typical difficulty: a near-mean family can have
   both models near-perfect, or both poor. A median would not be wrong — it
   would answer a different question the paper does not report.
4. Families are ranked by |family mean - target|, ties by cluster id. The
   first family that holds at least one ELIGIBLE chain is the case; its
   representative chain is the eligible chain whose own gain is nearest the
   family mean, ties by `sample_id`. Families already used by an earlier
   category (primary or backup) are skipped, so no family appears twice.

Eligibility keeps the extremes honest and the maps legible, and is applied to
chains only: the evaluated crop is the whole chain (`query_len <= --max-len`,
so the map has no cropped-out contacts), length `>= --min-len`, and at least
`--min-pos-frac * L` true long-range contacts (a top-L ranking over a handful of
positives is a coin flip). The "failure" category additionally requires a
non-zero long-range contact prior (`has_prior_long == 1` from
`template_coverage.py`): a case labelled "retrieved geometry misleads" must at
least have had geometry.

Each category also records `--n-backups` further families in rank order. A
backup replaces the primary only if the primary cannot be rendered, and the
reason must be written into the figure's provenance — the rule does not permit
choosing the prettier map.

What the selection does NOT do: it does not estimate anything (Tables 1 and 12
do), it does not establish that a "failure" prior was wrong (neither coverage
nor analysis H measures prior correctness), and its identity regime is the
top-1 template by retrieval score (rank 2-4 may be closer).

Inputs must be one population: `sample_id` sets of the reference, control,
identity and coverage tables must be identical, and the paired gain recomputed
from reference/control must reproduce the identity table's `delta_*` column
exactly — otherwise the tables belong to different runs and the script refuses.

Usage (local, CPU, seconds):
    python scripts/select_qualitative_cases.py \
        --reference logs/.../paper_650m_trufor_s42_bs1/.../per_sample_metrics.tsv \
        --control   logs/.../paper_650m_trufor_no_templates_s42_bs1/.../per_sample_metrics.tsv \
        --identity  paper/retrieval_identity/artifacts/650m_trufor_s42_2026/per_sample_best_template_identity.tsv \
        --coverage  paper/template_coverage/artifacts_650m_2026/coverage_per_chain.tsv \
        --out-dir   paper/qualitative_cases/artifacts_650m_s42_2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_identity_audit import IDENTITY_BINS  # noqa: E402

# Coarse regimes of Table 12 — left-closed, like analysis H. The two edges must
# be edges of the audit's fine bins, otherwise the regimes would not nest.
REMOTE_MAX = 0.30
NEAR_DUP_MIN = 0.90
_edges = {low for _, low, _ in IDENTITY_BINS} | {high for _, _, high in IDENTITY_BINS}
assert REMOTE_MAX in _edges and NEAR_DUP_MIN in _edges, "regimes must nest in IDENTITY_BINS"

REGIME_REMOTE = "<0.30"
REGIME_HOMOLOGOUS = "0.30-0.90"
REGIME_NEAR_DUP = ">=0.90"
REGIMES = (REGIME_REMOTE, REGIME_HOMOLOGOUS, REGIME_NEAR_DUP)


def identity_regime(identity: float) -> str:
    """Left-closed coarse stratum of a top-1 crop identity (NaN -> 'missing')."""
    if identity is None or not math.isfinite(identity):
        return "missing"
    if identity < REMOTE_MAX:
        return REGIME_REMOTE
    if identity < NEAR_DUP_MIN:
        return REGIME_HOMOLOGOUS
    return REGIME_NEAR_DUP


@dataclass(frozen=True)
class Category:
    name: str
    effect: str              # helps | no_effect | hurts — the reader's label
    regime: Optional[str]    # identity regime restricting the population, or None = all
    quantile: str            # max | min | mean | median — target on the per-family means
    require_prior_long: bool
    plan_row: Optional[int]  # row of figures_tables_plan.md 2.7 it fills, None = supplementary
    description: str


# Order matters: earlier categories claim families first. Extremes go first so
# that a near-mean pick cannot accidentally consume the maximal family.
CATEGORIES: Sequence[Category] = (
    Category("largest_gain", "helps", None, "max", False, 1,
             "family with the largest mean gain over the whole evaluated population"),
    Category("failure", "hurts", None, "min", True, 4,
             "family with the most negative mean gain; representative chain must "
             "have a non-zero long-range contact prior"),
    Category("near_mean_gain_homologous", "helps", REGIME_HOMOLOGOUS, "mean", False, 2,
             "family nearest the reported family-balanced gain (Table 12c) among chains "
             "whose top-ranked template has identity in [0.30, 0.90)"),
    Category("near_mean_gain_remote", "no_effect", REGIME_REMOTE, "mean", False, 3,
             "family nearest the reported family-balanced gain (Table 12c) among chains "
             "whose top-ranked template has identity below 0.30"),
    Category("near_mean_gain_near_dup", "helps", REGIME_NEAR_DUP, "mean", False, None,
             "family nearest the reported family-balanced gain (Table 12c) among chains "
             "whose top-ranked template has identity >= 0.90 (supplementary)"),
)


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_population(reference: Path, control: Path, identity: Path, coverage: Path,
                    metric: str) -> pd.DataFrame:
    """Join the four per-chain tables 1:1 and verify they describe one run pair."""
    ref = pd.read_csv(reference, sep="\t")
    ctl = pd.read_csv(control, sep="\t")
    idt = pd.read_csv(identity, sep="\t")
    cov = pd.read_csv(coverage, sep="\t")
    if "split" in cov.columns:
        cov = cov[cov["split"] == "test"]

    for name, df in (("reference", ref), ("control", ctl), ("identity", idt), ("coverage", cov)):
        if df["sample_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate sample_id")
    sets = {n: set(d["sample_id"]) for n, d in
            (("reference", ref), ("control", ctl), ("identity", idt), ("coverage", cov))}
    base = sets["reference"]
    for name, s in sets.items():
        if s != base:
            raise ValueError(
                f"population mismatch: {name} has {len(s)} ids, reference {len(base)}; "
                f"only in {name}: {len(s - base)}, only in reference: {len(base - s)}")

    if metric not in ref.columns or metric not in ctl.columns:
        raise ValueError(f"metric {metric!r} missing from reference/control")
    for col in ("cluster_id", "seq_len", "n_valid_long_pairs", "n_pos_long"):
        if col not in ref.columns:
            raise ValueError(f"reference lacks {col!r}")

    r = ref[["sample_id", "pdb_id", "chain_id", "subset", "cluster_id", "seq_len",
             "n_valid_long_pairs", "n_pos_long", "best_tpl_sim", metric]].rename(
        columns={metric: "ref"})
    c = ctl[["sample_id", metric, "n_valid_long_pairs"]].rename(
        columns={metric: "ctl", "n_valid_long_pairs": "n_valid_long_pairs_ctl"})
    id_cols = ["sample_id", "query_len", "crop_len", "crop_start", "crop_end",
               "best_template_id", "best_score", "crop_seq_identity_aligned",
               "crop_query_coverage"]
    delta_col = f"delta_{metric}"
    if delta_col in idt.columns:
        id_cols.append(delta_col)
    i = idt[id_cols]
    v = cov[["sample_id", "has_prior_long", "prior_nz_frac_long"]]

    pop = r.merge(c, on="sample_id").merge(i, on="sample_id").merge(v, on="sample_id")
    if (pop["n_valid_long_pairs"] != pop["n_valid_long_pairs_ctl"]).any():
        raise ValueError("reference and control disagree on n_valid_long_pairs — "
                         "not the same evaluation universe")
    pop = pop.drop(columns=["n_valid_long_pairs_ctl"])
    pop["delta"] = pop["ref"] - pop["ctl"]

    # Long-defined only: chains with no long-range pair carry NaN metrics and are
    # excluded from every headline aggregate; excluding them here keeps the
    # family means identical to the paper's.
    pop = pop[pop["n_valid_long_pairs"] > 0].copy()
    if pop["delta"].isna().any() or pop["ref"].isna().any() or pop["ctl"].isna().any():
        raise ValueError("NaN metric on a long-defined chain")

    if delta_col in pop.columns:
        gap = (pop["delta"] - pop[delta_col]).abs().max()
        if not (gap < 1e-9):
            raise ValueError(
                f"recomputed paired gain differs from the identity table's {delta_col} "
                f"(max |diff| = {gap:.3g}); the identity audit was run on a different pair")
        pop = pop.drop(columns=[delta_col])

    if (pop["cluster_id"] < 0).any():
        raise ValueError("unknown cluster (-1) in a long-defined chain; refusing to pool")
    pop["regime"] = pop["crop_seq_identity_aligned"].map(identity_regime)
    pop["whole_chain"] = pop["crop_len"] == pop["query_len"]
    return pop.sort_values("sample_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# rule
# ----------------------------------------------------------------------------

def eligible_mask(pop: pd.DataFrame, min_len: int, max_len: int, min_pos_frac: float,
                  require_prior_long: bool) -> pd.Series:
    m = (pop["whole_chain"]
         & (pop["query_len"] <= max_len)
         & (pop["seq_len"] >= min_len)
         & (pop["n_pos_long"] >= min_pos_frac * pop["seq_len"]))
    if require_prior_long:
        m = m & (pop["has_prior_long"] == 1)
    return m


def family_means(pop: pd.DataFrame, regime: Optional[str]) -> pd.DataFrame:
    """Per-family mean gain over the family's chains in the regime (or all)."""
    sub = pop if regime is None else pop[pop["regime"] == regime]
    g = sub.groupby("cluster_id")["delta"].agg(["mean", "count"]).reset_index()
    return g.rename(columns={"mean": "family_mean", "count": "family_n"})


def quantile_target(values: np.ndarray, quantile: str) -> float:
    if len(values) == 0:
        raise ValueError("empty family list")
    if quantile == "max":
        return float(np.max(values))
    if quantile == "min":
        return float(np.min(values))
    if quantile == "mean":
        return float(np.mean(values))
    if quantile == "median":
        return float(np.median(values))
    raise ValueError(quantile)


def select_category(pop: pd.DataFrame, cat: Category, used: set, *, min_len: int,
                    max_len: int, min_pos_frac: float, n_backups: int) -> List[Dict[str, object]]:
    fam = family_means(pop, cat.regime)
    target = quantile_target(fam["family_mean"].to_numpy(), cat.quantile)
    fam["dist"] = (fam["family_mean"] - target).abs()
    fam = fam.sort_values(["dist", "cluster_id"], kind="mergesort").reset_index(drop=True)

    sub = pop if cat.regime is None else pop[pop["regime"] == cat.regime]
    elig = sub[eligible_mask(sub, min_len, max_len, min_pos_frac, cat.require_prior_long)]

    picks: List[Dict[str, object]] = []
    for rank_all, frow in enumerate(fam.itertuples(index=False), start=1):
        cid = int(frow.cluster_id)
        if cid in used:
            continue
        cand = elig[elig["cluster_id"] == cid]
        if cand.empty:
            continue
        cand = cand.assign(_d=(cand["delta"] - frow.family_mean).abs())
        cand = cand.sort_values(["_d", "sample_id"], kind="mergesort")
        row = cand.iloc[0]
        used.add(cid)
        picks.append({
            "category": cat.name, "effect": cat.effect, "role": "primary" if not picks else
            f"backup_{len(picks)}", "plan_row": cat.plan_row if cat.plan_row is not None else "",
            "regime": cat.regime or "all", "quantile": cat.quantile, "target": target,
            "family_rank_incl_skipped": rank_all,
            "sample_id": row["sample_id"], "pdb_id": row["pdb_id"], "chain_id": row["chain_id"],
            "subset": row["subset"], "cluster_id": cid,
            "family_mean_gain": float(frow.family_mean), "family_n_in_regime": int(frow.family_n),
            "chain_gain": float(row["delta"]), "ref": float(row["ref"]), "ctl": float(row["ctl"]),
            "seq_len": int(row["seq_len"]), "n_pos_long": int(row["n_pos_long"]),
            "n_valid_long_pairs": int(row["n_valid_long_pairs"]),
            "top1_identity": float(row["crop_seq_identity_aligned"]),
            "top1_template_id": row["best_template_id"], "top1_score": float(row["best_score"]),
            "top1_query_coverage": float(row["crop_query_coverage"]),
            "has_prior_long": int(row["has_prior_long"]),
            "prior_nz_frac_long": float(row["prior_nz_frac_long"]),
            "crop_start": int(row["crop_start"]), "crop_end": int(row["crop_end"]),
        })
        if len(picks) > n_backups:
            break
    if not picks:
        raise ValueError(f"{cat.name}: no family with an eligible chain")
    return picks


def strata_summary(pop: pd.DataFrame, *, min_len: int, max_len: int,
                   min_pos_frac: float) -> pd.DataFrame:
    rows = []
    elig = eligible_mask(pop, min_len, max_len, min_pos_frac, False)
    for regime in (None, *REGIMES):
        sub = pop if regime is None else pop[pop["regime"] == regime]
        fam = family_means(pop, regime)
        rows.append({
            "regime": regime or "all", "chains": int(len(sub)),
            "families": int(len(fam)),
            "eligible_chains": int((elig & (pop["regime"] == regime if regime else True)).sum()),
            "eligible_families": int(pop.loc[elig & ((pop["regime"] == regime) if regime else True),
                                             "cluster_id"].nunique()),
            "family_mean_gain_median": float(np.median(fam["family_mean"])),
            "family_mean_gain_mean": float(fam["family_mean"].mean()),
            "family_mean_gain_min": float(fam["family_mean"].min()),
            "family_mean_gain_max": float(fam["family_mean"].max()),
            "chain_gain_mean": float(sub["delta"].mean()),
            "chain_gain_median": float(sub["delta"].median()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> pd.DataFrame:
    pop = load_population(args.reference, args.control, args.identity, args.coverage, args.metric)
    used: set = set()
    picks: List[Dict[str, object]] = []
    for cat in CATEGORIES:
        picks.extend(select_category(pop, cat, used, min_len=args.min_len, max_len=args.max_len,
                                     min_pos_frac=args.min_pos_frac, n_backups=args.n_backups))
    sel = pd.DataFrame(picks)
    summary = strata_summary(pop, min_len=args.min_len, max_len=args.max_len,
                             min_pos_frac=args.min_pos_frac)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "selected_cases.tsv", sep="\t", index=False, float_format="%.6g")
    summary.to_csv(out / "strata_summary.tsv", sep="\t", index=False, float_format="%.6g")
    # What the server export job needs, and nothing else.
    sel[["category", "role", "sample_id", "pdb_id", "chain_id", "crop_start", "crop_end",
         "top1_template_id"]].to_csv(out / "export_requests.tsv", sep="\t", index=False)

    manifest = {
        "rule": {
            "unit": "sequence family (cluster); one family per case, never reused",
            "population": "long-defined evaluated test chains of the seed-42 pair, restricted to "
                          "the category's identity regime when it has one",
            "regimes": {"boundaries": [REMOTE_MAX, NEAR_DUP_MIN], "closure": "left-closed, as "
                        "analysis H / Table 12", "attribute": "top-1 template by retrieval score"},
            "target": "statistic (max | min | mean) of per-family mean paired gains; the mean "
                      "of per-family means is the regime's reported cluster-balanced gain",
            "family_choice": "nearest family mean to the target, ties by cluster_id, first with an "
                             "eligible chain",
            "chain_choice": "eligible chain nearest the family mean, ties by sample_id",
            "eligibility": {"whole_chain": True, "min_len": args.min_len, "max_len": args.max_len,
                            "min_pos_frac": args.min_pos_frac,
                            "failure_requires_has_prior_long": True},
            "backups": args.n_backups,
            "backup_policy": "a backup replaces the primary only if the primary cannot be rendered; "
                             "record the reason in the figure provenance",
            "metric": args.metric,
            "categories": [asdict(c) for c in CATEGORIES],
        },
        "inputs": {k: {"path": str(p), "md5": md5(p)} for k, p in
                   (("reference", args.reference), ("control", args.control),
                    ("identity", args.identity), ("coverage", args.coverage))},
        "population": {"chains": int(len(pop)), "families": int(pop["cluster_id"].nunique())},
        "limits": [
            "Cases illustrate; they estimate nothing. The estimates are Tables 1 and 12.",
            "'failure' = negative paired gain with a non-zero long-range contact prior. It does "
            "not establish that the transferred contacts were wrong: neither coverage nor "
            "analysis H measures prior correctness.",
            "The identity regime is the top-1 template by retrieval score; ranks 2-4 may be "
            "closer.",
            "'near_mean_gain_*' = the family whose mean gain is nearest the regime's reported "
            "family-balanced gain (Table 12c); the representative chain is nearest its family "
            "mean. It illustrates the reported mean delta — not necessarily the regime's typical "
            "difficulty, and not a median.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return sel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metric", default="P@L_long")
    p.add_argument("--min-len", type=int, default=80)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--min-pos-frac", type=float, default=0.5)
    p.add_argument("--n-backups", type=int, default=1)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sel = run(args)
    cols = ["category", "role", "sample_id", "cluster_id", "regime", "seq_len", "top1_identity",
            "family_mean_gain", "chain_gain", "ref", "ctl"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(sel[cols].to_string(index=False))
    print(f"\nwrote {args.out_dir}/selected_cases.tsv, strata_summary.tsv, export_requests.tsv, "
          f"manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
