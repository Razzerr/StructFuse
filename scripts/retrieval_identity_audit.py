#!/usr/bin/env python3
"""Audit sequence identity between each query and its best retrieved template.

This script is intentionally data-side and model-free. It replays FAISS
retrieval with the same admissibility filters used at validation/test time,
loads the query/template NPZ sequences, computes global-alignment sequence
identity, and aggregates the results by retrieval-score bin and by
sequence-identity bin.

Aggregation unit (2026-09-11): ``--bootstrap cluster`` (default) averages every
bin per query sequence cluster first, then unweighted over clusters, and
bootstraps clusters — the same estimator as the headline metrics and
``paired_significance.py``. Per-chain values are kept beside it as ``*_chain``
columns. ``--bootstrap chain`` reproduces the pre-2026 per-chain summaries and
is not valid for new claims. The cluster comes from the reference TSV's
``cluster_id`` (what the headline metric used) and falls back to the index.

Run it on the server that has `data/processed` and `data/index_t33_2026` or
`data/index_t6`; those files are not part of the lightweight local paper
checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_SECTION_DIR = ROOT / "paper" / "retrieval_identity"
DEFAULT_OUTDIR = PAPER_SECTION_DIR / "artifacts"

SIM_BINS: tuple[tuple[str, float, float], ...] = (
    ("sim<0.95", -math.inf, 0.95),
    ("0.95-0.98", 0.95, 0.98),
    ("0.98-0.99", 0.98, 0.99),
    ("0.99-0.995", 0.99, 0.995),
    ("0.995-0.999", 0.995, 0.999),
    ("sim>=0.999", 0.999, math.inf),
)

IDENTITY_BINS: tuple[tuple[str, float, float], ...] = (
    ("id<0.20", -math.inf, 0.20),
    ("0.20-0.30", 0.20, 0.30),
    ("0.30-0.40", 0.30, 0.40),
    ("0.40-0.50", 0.40, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.90", 0.70, 0.90),
    ("0.90-0.99", 0.90, 0.99),
    ("id>=0.99", 0.99, math.inf),
)

METRIC_COLUMNS = (
    "P@L_long",
    "P@L/2_long",
    "P@L/5_long",
    "AUC-PR_long",
    "f1_long",
)


def _get_protein_id(chain_id: str) -> str:
    """Extract protein-level ID consistently with src.models.utils.faiss."""
    return chain_id.rsplit("_", 1)[0].lower()


class LightweightFaissIndex:
    """FAISS retrieval helper that avoids loading embeddings.npy.

    The runtime FaissIndex loads both `faiss.index` and `embeddings.npy` because
    training needs O(1) access to precomputed query vectors. For this audit the
    vectors are already stored inside IndexFlatIP, so reconstructing the query
    vector from the FAISS index avoids a second 4.4 GB array for index_t33.
    """

    def __init__(self, index_dir: Path):
        import faiss

        self.index_dir = index_dir
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with (index_dir / "ids.json").open() as handle:
            self.meta = json.load(handle)
        self.row2id = [m["id"] for m in self.meta]
        self.id2npz = {m["id"]: path_from_root(m["npz"]) for m in self.meta}
        self.id2cluster = {m["id"]: int(m.get("cluster_id", -1)) for m in self.meta}
        self.row2cluster = [int(m.get("cluster_id", -1)) for m in self.meta]
        self.chain2cluster = {m["id"]: int(m.get("cluster_id", -1)) for m in self.meta}
        self._id2row = {chain_id: i for i, chain_id in enumerate(self.row2id)}
        self.d = int(self.index.d)

        self.cluster2size: dict[int, int] = {}
        for cid in self.row2cluster:
            cid = int(cid)
            if cid != -1:
                self.cluster2size[cid] = self.cluster2size.get(cid, 0) + 1

    def clusters_for_chain(self, chain_id: str, prot_id: str | None = None) -> set[int]:
        """Mirror of ``FaissIndex._clusters_for_chain`` — keep the two in sync."""
        exact = int(self.chain2cluster.get(chain_id, -1))
        return {exact} if exact != -1 else set()

    def same_cluster(self, query_clusters: set[int], tpl_cluster: int) -> bool:
        """Mirror of ``FaissIndex._same_cluster`` — keep the two in sync.

        True ⇒ blocked. An unknown cluster on either side blocks; it is never
        read as "different cluster, therefore admissible".
        """
        if not query_clusters:
            return True
        tpl_cluster = int(tpl_cluster)
        return tpl_cluster == -1 or tpl_cluster in query_clusters

    def _reconstruct_query(self, query_name: str) -> np.ndarray | None:
        row = self._id2row.get(query_name)
        if row is None:
            return None
        try:
            x = self.index.reconstruct(int(row))
        except TypeError:
            x = np.empty(self.d, dtype=np.float32)
            self.index.reconstruct(int(row), x)
        return np.asarray(x, dtype=np.float32).reshape(1, -1)

    def topk_precomputed(
        self,
        query_name: str,
        k: int,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float]]:
        query_prot_id = _get_protein_id(query_name)
        query_clusters = self.clusters_for_chain(query_name, query_prot_id)
        x = self._reconstruct_query(query_name)
        if x is None:
            return []

        extra_cluster = sum(self.cluster2size.get(cid, 0) for cid in query_clusters)
        search_k = max(k * 3, k + 500 + extra_cluster)
        search_k = min(search_k, self.index.ntotal)
        sims, idxs = self.index.search(x.astype(np.float32), search_k)

        out: list[tuple[str, float]] = []
        for sim, row_idx in zip(sims[0].tolist(), idxs[0].tolist()):
            if row_idx < 0:
                continue
            tpl_id = self.row2id[row_idx]
            tpl_prot_id = _get_protein_id(tpl_id)
            if tpl_prot_id == query_prot_id:
                continue
            if self.same_cluster(query_clusters, int(self.row2cluster[row_idx])):
                continue
            if sim < min_similarity:
                continue
            out.append((tpl_id, float(sim)))
            if len(out) >= k:
                break
        return out


def path_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_float(value: object) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_per_sample(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path} is missing a sample_id column")
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"{path} contains duplicate sample_id={sample_id}")
            rows[sample_id] = row
    return rows


def load_seq(npz_path: Path) -> str:
    try:
        data = np.load(npz_path, allow_pickle=True)
        try:
            seq_arr = data["seq"]
            if isinstance(seq_arr, np.ndarray) and seq_arr.shape == ():
                return str(seq_arr.item())
            return str(seq_arr)
        finally:
            data.close()
    except (EOFError, OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RuntimeError(f"Could not load sequence from {npz_path}: {exc}") from exc


def center_crop_bounds(length: int, crop_size: int | None) -> tuple[int, int]:
    if crop_size is None or crop_size <= 0 or crop_size >= length:
        return 0, length
    start = max(0, (length - crop_size) // 2)
    return start, start + crop_size


def alignment_identity(query_seq: str, template_seq: str) -> dict[str, float | int]:
    from src.data.utils.align import needleman_wunsch

    q_aln, t_aln, _, _ = needleman_wunsch(query_seq, template_seq)
    aligned_pairs = 0
    matches = 0
    query_residues = 0
    template_residues = 0
    for q, t in zip(q_aln, t_aln):
        has_q = q != "-"
        has_t = t != "-"
        query_residues += int(has_q)
        template_residues += int(has_t)
        if has_q and has_t:
            aligned_pairs += 1
            matches += int(q == t)

    identity_aligned = matches / aligned_pairs if aligned_pairs else float("nan")
    identity_query = matches / len(query_seq) if query_seq else float("nan")
    query_coverage = aligned_pairs / len(query_seq) if query_seq else float("nan")
    template_coverage = aligned_pairs / len(template_seq) if template_seq else float("nan")
    return {
        "matches": matches,
        "aligned_pairs": aligned_pairs,
        "query_residues": query_residues,
        "template_residues": template_residues,
        "seq_identity_aligned": identity_aligned,
        "seq_identity_query_len": identity_query,
        "query_coverage": query_coverage,
        "template_coverage": template_coverage,
    }


def bin_label(score: float) -> str:
    if not math.isfinite(score):
        return "missing"
    # FAISS/IP scores can exceed 1 by tiny float error after normalization.
    score = min(max(score, 0.0), 1.0)
    for label, low, high in SIM_BINS:
        if score >= low and score < high:
            return label
    return "missing"


def bootstrap_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` — one entry per resampling unit."""
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, len(values), size=len(values))
        means[i] = float(values[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def finite_values(rows: list[dict[str, object]], column: str) -> np.ndarray:
    vals = np.array([parse_float(row.get(column)) for row in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def quantile_or_nan(values: Iterable[float], q: float) -> float:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def frac_ge(values: Iterable[float], threshold: float) -> float:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float((arr >= threshold).mean()) if len(arr) else float("nan")


def frac_lt(values: Iterable[float], threshold: float) -> float:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float((arr < threshold).mean()) if len(arr) else float("nan")


def cluster_of(row: dict[str, object]) -> int:
    """Query cluster used for aggregation. Prefers the reference TSV's
    ``cluster_id`` (the id the headline metric aggregated over); falls back to
    the index-derived ``query_cluster_id``. -1 means unknown."""
    cid = parse_float(row.get("ref_cluster_id"))
    if math.isfinite(cid) and cid >= 0:
        return int(cid)
    raw = str(row.get("query_cluster_id", "")).strip()
    if raw and ";" not in raw:
        try:
            return int(raw)
        except ValueError:
            return -1
    return -1


def cluster_means(rows: list[dict[str, object]], column: str) -> np.ndarray:
    """Per-cluster mean of ``column`` over ``rows``; rows with unknown cluster or
    non-finite value are dropped. Returns one value per cluster."""
    acc: dict[int, list[float]] = {}
    for row in rows:
        cid = cluster_of(row)
        if cid < 0:
            continue
        val = parse_float(row.get(column))
        if not math.isfinite(val):
            continue
        acc.setdefault(cid, []).append(val)
    return np.array([float(np.mean(v)) for v in acc.values()], dtype=np.float64)


def cluster_frac(rows: list[dict[str, object]], column: str, threshold: float, below: bool = False) -> float:
    """Cluster-balanced fraction: per-cluster fraction of chains meeting the
    threshold, then unweighted mean over clusters."""
    acc: dict[int, list[float]] = {}
    for row in rows:
        cid = cluster_of(row)
        if cid < 0:
            continue
        val = parse_float(row.get(column))
        if not math.isfinite(val):
            continue
        acc.setdefault(cid, []).append(float(val < threshold) if below else float(val >= threshold))
    if not acc:
        return float("nan")
    return float(np.mean([np.mean(v) for v in acc.values()]))


def n_clusters(rows: list[dict[str, object]]) -> int:
    return len({cluster_of(r) for r in rows if cluster_of(r) >= 0})


def identity_bin_label(identity: float) -> str:
    if not math.isfinite(identity):
        return "missing"
    for label, low, high in IDENTITY_BINS:
        if identity >= low and identity < high:
            return label
    return "missing"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return repr(float(value))
    return value


def formatted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: format_value(v) for k, v in row.items()} for row in rows]


def _identity_stats(group: list[dict[str, object]], unit: str, prefix: str) -> dict[str, object]:
    """Identity / coverage summaries for one bin. Chain-level distribution
    statistics plus the cluster-balanced mean and fractions."""
    col_al = f"{prefix}_seq_identity_aligned"
    col_ql = f"{prefix}_seq_identity_query_len"
    col_cov = f"{prefix}_query_coverage"
    ident = [parse_float(r.get(col_al)) for r in group]
    out: dict[str, object] = {
        f"mean_{col_al}_chain": mean_or_nan(ident),
        f"median_{col_al}_chain": quantile_or_nan(ident, 0.5),
        f"p25_{col_al}_chain": quantile_or_nan(ident, 0.25),
        f"p75_{col_al}_chain": quantile_or_nan(ident, 0.75),
        f"frac_{prefix}_identity_ge_30pct_chain": frac_ge(ident, 0.30),
        f"frac_{prefix}_identity_ge_50pct_chain": frac_ge(ident, 0.50),
        f"frac_{prefix}_identity_ge_90pct_chain": frac_ge(ident, 0.90),
        f"frac_{prefix}_identity_ge_99pct_chain": frac_ge(ident, 0.99),
        f"frac_{prefix}_identity_lt_30pct_chain": frac_lt(ident, 0.30),
        f"mean_{col_ql}_chain": mean_or_nan(parse_float(r.get(col_ql)) for r in group),
        f"mean_{col_cov}_chain": mean_or_nan(parse_float(r.get(col_cov)) for r in group),
    }
    if unit == "cluster":
        cm = cluster_means(group, col_al)
        out.update(
            {
                f"mean_{col_al}": float(cm.mean()) if len(cm) else float("nan"),
                f"median_{col_al}": float(np.median(cm)) if len(cm) else float("nan"),
                f"frac_{prefix}_identity_ge_30pct": cluster_frac(group, col_al, 0.30),
                f"frac_{prefix}_identity_ge_50pct": cluster_frac(group, col_al, 0.50),
                f"frac_{prefix}_identity_ge_90pct": cluster_frac(group, col_al, 0.90),
                f"frac_{prefix}_identity_ge_99pct": cluster_frac(group, col_al, 0.99),
                f"frac_{prefix}_identity_lt_30pct": cluster_frac(group, col_al, 0.30, below=True),
                f"mean_{col_ql}": mean_or_nan(cluster_means(group, col_ql)),
                f"mean_{col_cov}": mean_or_nan(cluster_means(group, col_cov)),
            }
        )
    else:
        for k in list(out):
            if k.endswith("_chain"):
                out[k[: -len("_chain")]] = out[k]
    return out


def _metric_stats(
    group: list[dict[str, object]], unit: str, n_bootstrap: int, seed: int
) -> dict[str, object]:
    out: dict[str, object] = {}
    for m_idx, metric in enumerate(METRIC_COLUMNS):
        ref_col = f"reference_{metric}"
        ctrl_col = f"control_{metric}"
        delta_col = f"delta_{metric}"
        chain_delta = finite_values(group, delta_col)
        out[f"n_{delta_col}_chain"] = int(len(chain_delta))
        out[f"mean_{delta_col}_chain"] = float(chain_delta.mean()) if len(chain_delta) else float("nan")
        if unit == "cluster":
            ref_vals = cluster_means(group, ref_col)
            ctrl_vals = cluster_means(group, ctrl_col)
            delta_vals = cluster_means(group, delta_col)
            out[f"n_clusters_{delta_col}"] = int(len(delta_vals))
        else:
            ref_vals = finite_values(group, ref_col)
            ctrl_vals = finite_values(group, ctrl_col)
            delta_vals = chain_delta
        out[f"mean_{ref_col}"] = float(ref_vals.mean()) if len(ref_vals) else float("nan")
        out[f"mean_{ctrl_col}"] = float(ctrl_vals.mean()) if len(ctrl_vals) else float("nan")
        out[f"mean_{delta_col}"] = float(delta_vals.mean()) if len(delta_vals) else float("nan")
        ci_low, ci_high = bootstrap_ci(delta_vals, n_bootstrap, seed + 1009 * (m_idx + 1))
        out[f"ci95_lo_{delta_col}"] = ci_low
        out[f"ci95_hi_{delta_col}"] = ci_high
    return out


def summarize_by(
    rows: list[dict[str, object]],
    bin_column: str,
    bin_labels: Iterable[str],
    identity_prefixes: Iterable[str],
    unit: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_bin: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_bin.setdefault(str(row.get(bin_column, "missing")), []).append(row)
    for idx, label in enumerate((*bin_labels, "missing")):
        group = by_bin.get(label, [])
        if not group:
            continue
        row: dict[str, object] = {
            bin_column: label,
            "unit": unit,
            "n": len(group),
            "n_clusters": n_clusters(group) if unit == "cluster" else len(group),
            "n_with_template": sum(int(r.get("n_hits", 0)) > 0 for r in group),
            "mean_best_score_chain": mean_or_nan(parse_float(r.get("score_for_bin")) for r in group),
        }
        if unit == "cluster":
            row["mean_best_score"] = mean_or_nan(cluster_means(group, "score_for_bin"))
        else:
            row["mean_best_score"] = row["mean_best_score_chain"]
        for prefix in identity_prefixes:
            row.update(_identity_stats(group, unit, prefix))
        row.update(_metric_stats(group, unit, n_bootstrap, seed + 7919 * (idx + 1)))
        out.append(row)
    return out


def summarize(rows: list[dict[str, object]], n_bootstrap: int, seed: int, unit: str = "cluster") -> list[dict[str, object]]:
    """Per retrieval-score bin (kept for backward compatibility of the output name)."""
    prefixes = [p for p in ("crop", "full") if any(f"{p}_seq_identity_aligned" in r for r in rows)]
    return summarize_by(rows, "score_bin", [l for l, _, _ in SIM_BINS], prefixes, unit, n_bootstrap, seed)


def global_summary(rows: list[dict[str, object]], unit: str, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    """One row over all queries: the whole-test redundancy and gain figures."""
    prefixes = [p for p in ("crop", "full") if any(f"{p}_seq_identity_aligned" in r for r in rows)]
    row: dict[str, object] = {
        "scope": "all",
        "unit": unit,
        "n": len(rows),
        "n_clusters": n_clusters(rows),
        "n_with_template": sum(int(r.get("n_hits", 0)) > 0 for r in rows),
        "n_unknown_query_cluster": sum(cluster_of(r) < 0 for r in rows),
        "frac_best_template_same_cluster_as_query": mean_or_nan(
            float(str(r.get("best_template_cluster_id", "")) != "" and cluster_of(r) >= 0
                  and parse_float(r.get("best_template_cluster_id")) == cluster_of(r))
            for r in rows if int(r.get("n_hits", 0)) > 0
        ),
        "n_best_template_unknown_cluster": sum(
            parse_float(r.get("best_template_cluster_id")) == -1 for r in rows if int(r.get("n_hits", 0)) > 0
        ),
        "mean_best_score_chain": mean_or_nan(parse_float(r.get("score_for_bin")) for r in rows),
        "mean_best_score": mean_or_nan(cluster_means(rows, "score_for_bin")) if unit == "cluster"
        else mean_or_nan(parse_float(r.get("score_for_bin")) for r in rows),
    }
    for prefix in prefixes:
        row.update(_identity_stats(rows, unit, prefix))
    row.update(_metric_stats(rows, unit, n_bootstrap, seed))
    out = [row]
    casp = [r for r in rows if str(r.get("subset", "")) == "casp16"]
    if casp:
        crow: dict[str, object] = {
            "scope": "casp16", "unit": unit, "n": len(casp), "n_clusters": n_clusters(casp),
            "n_with_template": sum(int(r.get("n_hits", 0)) > 0 for r in casp),
            "n_unknown_query_cluster": sum(cluster_of(r) < 0 for r in casp),
            "frac_best_template_same_cluster_as_query": float("nan"),
            "n_best_template_unknown_cluster": sum(
                parse_float(r.get("best_template_cluster_id")) == -1 for r in casp if int(r.get("n_hits", 0)) > 0
            ),
            "mean_best_score_chain": mean_or_nan(parse_float(r.get("score_for_bin")) for r in casp),
            "mean_best_score": mean_or_nan(cluster_means(casp, "score_for_bin")) if unit == "cluster"
            else mean_or_nan(parse_float(r.get("score_for_bin")) for r in casp),
        }
        for prefix in prefixes:
            crow.update(_identity_stats(casp, unit, prefix))
        crow.update(_metric_stats(casp, unit, n_bootstrap, seed + 31))
        out.append(crow)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="650m_trufor_k4")
    parser.add_argument("--index-dir", default="data/index_t33_2026")
    parser.add_argument("--data-root", default="data/processed_2026")
    parser.add_argument("--id-list", default="data/output_splits_2026/test_ids.txt")
    parser.add_argument("--splits-json", default="data/output_splits_2026/mmcif_final_splits.json")
    parser.add_argument("--skip-ids-file", default="data/corrupt_ids.txt")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--min-template-similarity", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--identity-scope",
        choices=("crop", "full", "both"),
        default="crop",
        help="Crop identity matches the center-cropped contact-evaluation input.",
    )
    parser.add_argument(
        "--per-sample-tsv",
        default=(
            "logs/paper_650m_trufor_s42_bs1/runs/"
            "2026-09-11_08-45-58/per_sample_metrics.tsv"
        ),
        help="Reference run TSV (bs=1 final evaluation). Restricts the audit to the "
        "evaluated (C=8-capped) proteins and carries their metrics and cluster ids.",
    )
    parser.add_argument(
        "--control-per-sample-tsv",
        default=(
            "logs/paper_650m_trufor_no_templates_s42_bs1/runs/"
            "2026-09-11_08-47-11/per_sample_metrics.tsv"
        ),
        help="Paired control TSV (bs=1), the matched no-template model.",
    )
    parser.add_argument(
        "--bootstrap",
        choices=("cluster", "chain"),
        default="cluster",
        help="Aggregation and resampling unit for every summary. 'cluster' (default) "
        "averages per query sequence cluster first; 'chain' reproduces the pre-2026 "
        "per-chain summaries and is not valid for new claims.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--allow-score-mismatch",
        action="store_true",
        help="Do not fail if recomputed FAISS scores differ from the reference TSV.",
    )
    parser.add_argument("--score-tolerance", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.time()

    out_dir = path_from_root(args.out_dir) if args.out_dir else DEFAULT_OUTDIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    index_dir = path_from_root(args.index_dir)
    data_root = path_from_root(args.data_root)
    id_list = path_from_root(args.id_list)
    splits_json = path_from_root(args.splits_json)
    skip_ids = path_from_root(args.skip_ids_file) if args.skip_ids_file else None
    per_sample_path = path_from_root(args.per_sample_tsv) if args.per_sample_tsv else None
    control_path = path_from_root(args.control_per_sample_tsv) if args.control_per_sample_tsv else None

    reference_rows = read_per_sample(per_sample_path)
    if per_sample_path and not reference_rows:
        raise ValueError(f"No rows loaded from {per_sample_path}")

    # Input contract — the audit must not complete on incomplete inputs.
    # (1) A named control must exist; a missing file used to degrade silently
    #     to an empty dict and NaN deltas. Pass --control-per-sample-tsv "" for a
    #     descriptive audit without a control.
    # (2) Reference and control must cover the same chains: no silent narrowing
    #     to the intersection.
    control_rows: dict[str, dict[str, str]] = {}
    if control_path is not None:
        if not control_path.exists():
            raise FileNotFoundError(
                f"--control-per-sample-tsv {rel(control_path)} does not exist. "
                "Pass an empty string to run without a paired control."
            )
        control_rows = read_per_sample(control_path)
        if not control_rows:
            raise ValueError(f"No rows loaded from {control_path}")
        if reference_rows:
            ref_ids, ctl_ids = set(reference_rows), set(control_rows)
            if ref_ids != ctl_ids:
                only_ref = sorted(ref_ids - ctl_ids)
                only_ctl = sorted(ctl_ids - ref_ids)
                raise ValueError(
                    "Reference and control populations differ: "
                    f"{len(ref_ids)} vs {len(ctl_ids)} chains; "
                    f"{len(only_ref)} only in reference (e.g. {only_ref[:5]}), "
                    f"{len(only_ctl)} only in control (e.g. {only_ctl[:5]}). "
                    "The audit does not narrow to the intersection."
                )

    required = [index_dir / "faiss.index", index_dir / "ids.json"]
    if not reference_rows and not id_list.exists():
        required.append(id_list)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required server-side inputs:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    print(f"Loading FAISS index from {rel(index_dir)}", flush=True)
    faiss_index = LightweightFaissIndex(index_dir)
    id_to_npz = faiss_index.id2npz
    id_to_cluster = faiss_index.id2cluster

    rows: list[dict[str, object]] = []
    score_mismatches: list[dict[str, object]] = []
    if reference_rows:
        # (3) Every reference chain must be in the index; dropping absentees
        #     here used to hide a mismatched --index-dir behind a smaller n.
        absent = [pid for pid in reference_rows if pid not in id_to_npz]
        if absent:
            raise KeyError(
                f"{len(absent)} of {len(reference_rows)} reference chains are absent from "
                f"{rel(index_dir)}/ids.json (e.g. {absent[:5]}). Check that --index-dir is the "
                "index the reference run retrieved from; the audit does not drop them."
            )
        query_ids = list(reference_rows)
    else:
        raw_ids = {line.strip() for line in id_list.read_text().splitlines() if line.strip()}
        query_ids = [pid for pid in faiss_index.row2id if pid.split("_")[0] in raw_ids]
    if args.max_samples > 0:
        query_ids = query_ids[: args.max_samples]

    print(f"Scanning {len(query_ids)} query chains (topk={args.topk})", flush=True)
    for n_done, pid in enumerate(query_ids, start=1):
        query_npz = id_to_npz.get(pid)
        if query_npz is None:
            raise KeyError(f"Query {pid} missing from ids.json metadata")
        full_seq = load_seq(query_npz)
        crop_start, crop_end = center_crop_bounds(len(full_seq), args.crop_size)
        crop_seq = full_seq[crop_start:crop_end]
        hits = faiss_index.topk_precomputed(
            pid,
            args.topk,
            min_similarity=args.min_template_similarity,
        )

        ref = reference_rows.get(pid, {})
        ctrl = control_rows.get(pid, {})
        logged_score = parse_float(ref.get("best_tpl_sim"))
        best_tpl_id = ""
        best_score = float("nan")
        tpl_seq = ""
        tpl_cluster = ""
        row: dict[str, object] = {
            "sample_id": pid,
            "pdb_id": str(ref.get("pdb_id", pid.split("_")[0])),
            "chain_id": str(ref.get("chain_id", pid.rsplit("_", 1)[-1] if "_" in pid else "")),
            "subset": str(ref.get("subset", "")),
            "query_cluster_id": ";".join(
                str(cid) for cid in sorted(faiss_index.clusters_for_chain(pid))
            ),
            "ref_cluster_id": ref.get("cluster_id", ""),
            "query_len": len(full_seq),
            "crop_start": crop_start,
            "crop_end": crop_end,
            "crop_len": len(crop_seq),
            "topk": args.topk,
            "n_hits": len(hits),
            "best_template_id": "",
            "best_template_protein_id": "",
            "best_template_cluster_id": "",
            "best_score": best_score,
            "logged_best_tpl_sim": logged_score,
            "score_abs_diff": float("nan"),
            "score_for_bin": logged_score if math.isfinite(logged_score) else best_score,
            "score_bin": "missing",
            "crop_identity_bin": "missing",
            "full_identity_bin": "missing",
        }

        if hits:
            best_tpl_id, best_score = hits[0]
            tpl_npz = id_to_npz.get(best_tpl_id)
            if tpl_npz is None:
                raise KeyError(f"Template {best_tpl_id} missing from ids.json metadata")
            tpl_seq = load_seq(tpl_npz)
            tpl_cluster = id_to_cluster.get(best_tpl_id, "")
            row.update(
                {
                    "best_template_id": best_tpl_id,
                    "best_template_protein_id": _get_protein_id(best_tpl_id),
                    "best_template_cluster_id": tpl_cluster,
                    "best_template_len": len(tpl_seq),
                    "best_score": best_score,
                    "score_for_bin": logged_score if math.isfinite(logged_score) else best_score,
                }
            )
            if math.isfinite(logged_score):
                diff = abs(logged_score - best_score)
                row["score_abs_diff"] = diff
                if diff > args.score_tolerance and len(score_mismatches) < 10:
                    score_mismatches.append(
                        {
                            "sample_id": pid,
                            "logged_best_tpl_sim": logged_score,
                            "recomputed_best_score": best_score,
                            "best_template_id": best_tpl_id,
                            "score_abs_diff": diff,
                        }
                    )
            score_for_bin = parse_float(row["score_for_bin"])
            row["score_bin"] = bin_label(score_for_bin)

            if args.identity_scope in ("crop", "both"):
                ident = alignment_identity(crop_seq, tpl_seq)
                row.update({f"crop_{k}": v for k, v in ident.items()})
                row["crop_identity_bin"] = identity_bin_label(parse_float(row.get("crop_seq_identity_aligned")))
            if args.identity_scope in ("full", "both"):
                ident = alignment_identity(full_seq, tpl_seq)
                row.update({f"full_{k}": v for k, v in ident.items()})
                row["full_identity_bin"] = identity_bin_label(parse_float(row.get("full_seq_identity_aligned")))

        for metric in METRIC_COLUMNS:
            ref_val = parse_float(ref.get(metric))
            ctrl_val = parse_float(ctrl.get(metric))
            row[f"reference_{metric}"] = ref_val
            row[f"control_{metric}"] = ctrl_val
            row[f"delta_{metric}"] = ref_val - ctrl_val if math.isfinite(ref_val) and math.isfinite(ctrl_val) else float("nan")

        rows.append(row)
        if n_done % 1000 == 0:
            print(f"  scanned {n_done}/{len(query_ids)}", flush=True)

    if score_mismatches and not args.allow_score_mismatch:
        mismatch_path = out_dir / "score_mismatch_examples.tsv"
        write_tsv(mismatch_path, formatted_rows(score_mismatches))
        raise RuntimeError(
            f"Recomputed FAISS scores differ from {rel(per_sample_path)} for at least "
            f"{len(score_mismatches)} examples. Wrote {rel(mismatch_path)}. "
            "Check that --index-dir, --topk and --per-sample-tsv come from the same run, "
            "or rerun with --allow-score-mismatch if this is intentional."
        )

    per_sample_out = out_dir / "per_sample_best_template_identity.tsv"
    summary_out = out_dir / "identity_by_retrieval_score_bin.tsv"
    crop_bin_out = out_dir / "gain_by_crop_identity_bin.tsv"
    full_bin_out = out_dir / "gain_by_full_identity_bin.tsv"
    global_out = out_dir / "global_summary.tsv"
    manifest_out = out_dir / "manifest.json"
    write_tsv(per_sample_out, formatted_rows(rows))
    unit = args.bootstrap
    n_unknown = sum(cluster_of(r) < 0 for r in rows)
    if unit == "cluster" and n_unknown:
        print(f"[audit] {n_unknown} queries have no known cluster and are excluded from cluster-balanced summaries", flush=True)
    prefixes = [p for p in ("crop", "full") if p == "crop" and args.identity_scope in ("crop", "both")
                or p == "full" and args.identity_scope in ("full", "both")]
    summary_rows = summarize(rows, args.n_bootstrap, args.seed, unit)
    write_tsv(summary_out, formatted_rows(summary_rows))
    id_labels = [l for l, _, _ in IDENTITY_BINS]
    written_bins = {}
    if "crop" in prefixes:
        write_tsv(crop_bin_out, formatted_rows(summarize_by(rows, "crop_identity_bin", id_labels, prefixes, unit, args.n_bootstrap, args.seed + 101)))
        written_bins["gain_by_crop_identity_bin"] = rel(crop_bin_out)
    if "full" in prefixes:
        write_tsv(full_bin_out, formatted_rows(summarize_by(rows, "full_identity_bin", id_labels, prefixes, unit, args.n_bootstrap, args.seed + 202)))
        written_bins["gain_by_full_identity_bin"] = rel(full_bin_out)
    write_tsv(global_out, formatted_rows(global_summary(rows, unit, args.n_bootstrap, args.seed + 303)))

    manifest = {
        "label": args.label,
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "index_dir": rel(index_dir),
            "data_root": rel(data_root),
            "id_list": rel(id_list),
            "splits_json": rel(splits_json),
            "skip_ids_file": rel(skip_ids) if skip_ids else "",
            "reference_per_sample_tsv": rel(per_sample_path) if per_sample_path else "",
            "control_per_sample_tsv": rel(control_path) if control_path and control_path.exists() else "",
        },
        "parameters": {
            "topk": args.topk,
            "crop_size": args.crop_size,
            "min_len": args.min_len,
            "min_template_similarity": args.min_template_similarity,
            "identity_scope": args.identity_scope,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "score_tolerance": args.score_tolerance,
            "allow_score_mismatch": args.allow_score_mismatch,
            "max_samples": args.max_samples,
        },
        "outputs": {
            "per_sample": rel(per_sample_out),
            "summary_by_score_bin": rel(summary_out),
            "global_summary": rel(global_out),
            **written_bins,
        },
        "aggregation_unit": unit,
        "populations": {
            "reference_chains": len(reference_rows),
            "control_chains": len(control_rows),
            "audited_chains": len(rows),
            "unknown_query_cluster": n_unknown,
        },
        "notes": [
            "Retrieval is replayed with filter_holdout=False, matching validation/test inference.",
            "Same-protein and same-cluster filtering remain active inside FaissIndex.topk_precomputed.",
            "Retrieval score is FAISS inner product/cosine similarity between normalized mean-pooled ESM2 embeddings.",
            "Default sequence identity is computed between the center-cropped query sequence and the best full-length template sequence, matching the contact-evaluation crop.",
            "seq_identity_aligned uses matches divided by aligned residue pairs; seq_identity_query_len uses matches divided by query length.",
            "Summaries are per query sequence cluster first (mean within cluster), then unweighted over clusters; *_chain columns are the per-chain values. A cluster contributes to every bin one of its chains falls in.",
            "Identity is computed for the single top-1 template by retrieval score, not for the most sequence-similar of the K templates; a closer homologue may sit at rank 2-4. Sufficient for best-template stratification; not sufficient for a claim that the model gains without any close homologue among its templates. Read high identity together with alignment coverage.",
            "n_clusters in a bin is the number of families represented in that bin; bins are a per-chain attribute of the reference run and are not a partition of families, so n_clusters sums to more than the family count across bins.",
            "The audit is restricted to the chains in the reference TSV, i.e. the C=8-capped evaluation set including the cap-exempt CASP16 chains.",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {rel(per_sample_out)}", flush=True)
    print(f"Wrote {rel(summary_out)}", flush=True)
    for name, path in written_bins.items():
        print(f"Wrote {path}", flush=True)
    print(f"Wrote {rel(global_out)}", flush=True)
    print(f"Wrote {rel(manifest_out)}", flush=True)


if __name__ == "__main__":
    main()
