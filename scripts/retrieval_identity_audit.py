#!/usr/bin/env python3
"""Audit sequence identity between each query and its best retrieved template.

This script is intentionally data-side and model-free. It replays FAISS
retrieval with the same admissibility filters used at validation/test time,
loads the query/template NPZ sequences, computes global-alignment sequence
identity, and aggregates the results by retrieval-score bin.

Run it on the server that has `data/processed` and `data/index_t33` or
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

        self.prot2clusters: dict[str, set[int]] = {}
        self.prot2cluster: dict[str, int] = {}
        for m in self.meta:
            prot_id = _get_protein_id(m["id"])
            cid = int(m.get("cluster_id", -1))
            if cid != -1:
                self.prot2clusters.setdefault(prot_id, set()).add(cid)
                self.prot2cluster.setdefault(prot_id, cid)

        self.cluster2size: dict[int, int] = {}
        for cid in self.row2cluster:
            cid = int(cid)
            if cid != -1:
                self.cluster2size[cid] = self.cluster2size.get(cid, 0) + 1

    def clusters_for_chain(self, chain_id: str, prot_id: str | None = None) -> set[int]:
        """Mirror of ``FaissIndex._clusters_for_chain`` — keep the two in sync."""
        exact = int(self.chain2cluster.get(chain_id, -1))
        if exact != -1:
            return {exact}
        prot_id = prot_id or _get_protein_id(chain_id)
        return set(self.prot2clusters.get(prot_id, set()))

    def same_cluster(self, query_clusters: set[int], tpl_id: str, tpl_cluster: int) -> bool:
        """Mirror of ``FaissIndex._same_cluster`` — keep the two in sync."""
        if not query_clusters:
            return False
        tpl_cluster = int(tpl_cluster)
        if tpl_cluster != -1:
            return tpl_cluster in query_clusters
        tpl_clusters = self.prot2clusters.get(_get_protein_id(tpl_id), set())
        return bool(query_clusters.intersection(tpl_clusters))

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
            if self.same_cluster(query_clusters, tpl_id, int(self.row2cluster[row_idx])):
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


def summarize(rows: list[dict[str, object]], n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_bin: dict[str, list[dict[str, object]]] = {label: [] for label, _, _ in SIM_BINS}
    by_bin["missing"] = []
    for row in rows:
        by_bin.setdefault(str(row["score_bin"]), []).append(row)

    for idx, (label, _, _) in enumerate((*SIM_BINS, ("missing", 0.0, 0.0))):
        group = by_bin.get(label, [])
        if not group:
            continue
        crop_identity = [parse_float(r.get("crop_seq_identity_aligned")) for r in group]
        crop_query_identity = [parse_float(r.get("crop_seq_identity_query_len")) for r in group]
        crop_query_coverage = [parse_float(r.get("crop_query_coverage")) for r in group]
        full_identity = [parse_float(r.get("full_seq_identity_aligned")) for r in group]
        row: dict[str, object] = {
            "score_bin": label,
            "n": len(group),
            "n_with_template": sum(int(r.get("n_hits", 0)) > 0 for r in group),
            "mean_best_score": mean_or_nan(parse_float(r.get("score_for_bin")) for r in group),
            "mean_crop_seq_identity_aligned": mean_or_nan(crop_identity),
            "median_crop_seq_identity_aligned": quantile_or_nan(crop_identity, 0.5),
            "p25_crop_seq_identity_aligned": quantile_or_nan(crop_identity, 0.25),
            "p75_crop_seq_identity_aligned": quantile_or_nan(crop_identity, 0.75),
            "frac_crop_identity_ge_30pct": frac_ge(crop_identity, 0.30),
            "frac_crop_identity_ge_50pct": frac_ge(crop_identity, 0.50),
            "frac_crop_identity_ge_90pct": frac_ge(crop_identity, 0.90),
            "mean_crop_seq_identity_query_len": mean_or_nan(crop_query_identity),
            "mean_crop_query_coverage": mean_or_nan(crop_query_coverage),
            "mean_full_seq_identity_aligned": mean_or_nan(full_identity),
        }
        for metric in METRIC_COLUMNS:
            ref_col = f"reference_{metric}"
            ctrl_col = f"control_{metric}"
            delta_col = f"delta_{metric}"
            ref_vals = finite_values(group, ref_col)
            ctrl_vals = finite_values(group, ctrl_col)
            delta_vals = finite_values(group, delta_col)
            row[f"mean_{ref_col}"] = float(ref_vals.mean()) if len(ref_vals) else float("nan")
            row[f"n_{ref_col}"] = int(len(ref_vals))
            row[f"mean_{ctrl_col}"] = float(ctrl_vals.mean()) if len(ctrl_vals) else float("nan")
            row[f"n_{ctrl_col}"] = int(len(ctrl_vals))
            row[f"mean_{delta_col}"] = float(delta_vals.mean()) if len(delta_vals) else float("nan")
            row[f"n_{delta_col}"] = int(len(delta_vals))
            ci_low, ci_high = bootstrap_ci(delta_vals, n_bootstrap, seed + 1009 * (idx + 1))
            row[f"ci95_lo_{delta_col}"] = ci_low
            row[f"ci95_hi_{delta_col}"] = ci_high
        out.append(row)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="650m_trufor_k4")
    parser.add_argument("--index-dir", default="data/index_t33")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--id-list", default="data/output_splits/test_ids.txt")
    parser.add_argument("--splits-json", default="data/output_splits/mmcif_final_splits.json")
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
            "logs/paper_650m_trufor_fusion_dist_s42/runs/"
            "2026-07-07_09-10-24/per_sample_metrics.tsv"
        ),
        help="Reference run TSV. Used to restrict to evaluated proteins and carry metrics.",
    )
    parser.add_argument(
        "--control-per-sample-tsv",
        default=(
            "logs/paper_650m_trained_no_templates_fixed/runs/"
            "2026-06-12_17-06-05/per_sample_metrics.tsv"
        ),
        help="Optional paired control TSV, usually trained no-template.",
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
    control_rows = read_per_sample(control_path) if control_path and control_path.exists() else {}
    if per_sample_path and not reference_rows:
        raise ValueError(f"No rows loaded from {per_sample_path}")

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
        query_ids = [pid for pid in reference_rows if pid in id_to_npz]
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
            if args.identity_scope in ("full", "both"):
                ident = alignment_identity(full_seq, tpl_seq)
                row.update({f"full_{k}": v for k, v in ident.items()})

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
    manifest_out = out_dir / "manifest.json"
    write_tsv(per_sample_out, formatted_rows(rows))
    summary_rows = summarize(rows, args.n_bootstrap, args.seed)
    write_tsv(summary_out, formatted_rows(summary_rows))

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
        },
        "notes": [
            "Retrieval is replayed with filter_holdout=False, matching validation/test inference.",
            "Same-protein and same-cluster filtering remain active inside FaissIndex.topk_precomputed.",
            "Retrieval score is FAISS inner product/cosine similarity between normalized mean-pooled ESM2 embeddings.",
            "Default sequence identity is computed between the center-cropped query sequence and the best full-length template sequence, matching the contact-evaluation crop.",
            "seq_identity_aligned uses matches divided by aligned residue pairs; seq_identity_query_len uses matches divided by query length.",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {rel(per_sample_out)}", flush=True)
    print(f"Wrote {rel(summary_out)}", flush=True)
    print(f"Wrote {rel(manifest_out)}", flush=True)


if __name__ == "__main__":
    main()
