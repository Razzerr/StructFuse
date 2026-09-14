"""Export the arrays behind Figure 4 for the pre-registered qualitative cases.

One invocation = one trained model. Run it twice — once for the headline
StructFuse checkpoint and once for the matched no-template checkpoint — on the
ten chains in `export_requests.tsv` (from `select_qualitative_cases.py`). Each
run writes one NPZ per chain with everything the figure needs and nothing the
figure could be tempted to re-derive:

    contact        (L, L) uint8   ground-truth contact map (crop = whole chain)
    pair_mask      (L, L) uint8   residues with coordinates
    valid_mask     (L, L) uint8   pair_mask AND |i-j| >= min_seq_sep (the metric mask)
    prior          (L, L) float32 projected template contact prior (zeros for
                                  the no-template run — by construction)
    count          (L, L) float32 template count per pair
    tpl_dist_bins  (9, L, L) float32 SOFT distance histogram — per-pair mass
                                  over bins after template aggregation, NOT
                                  integer bin ids; must never be cast to an
                                  integer dtype (only if the run has it)
    prob           (L, L) float32 sigmoid(logits) of THIS checkpoint, computed
                                  in the evaluation's dtype (bf16 under
                                  bf16-mixed) and upcast losslessly afterwards
    plus scalars: sample_id, seq, subset, cluster_id, crop_start/end, seq_len,
    threshold (the evaluation's `test/threshold_used`, full precision),
    prob_dtype, n_templates_retrieved, best_tpl_sim, tag.

Same forward pass as the evaluation. The prediction is `predict_only()` from
`cost_benchmark.py` — the exact code path Table 13 timed — under the trainer
precision's autocast, on the eval collate at batch size 1 (the reported
policy), with `trainer.deterministic` mirrored. The sigmoid is taken on the
logits AS THE NET EMITS THEM (`torch.sigmoid(logits)`, exactly as
`ContactLitModule._step`), not on `logits.float()`: under bf16-mixed the
probabilities are bf16-quantised and the top-L ranking's ties depend on that
quantisation. Upcasting first would produce a different ranking from the one
the tables report.

Provenance is PINNED, not discovered. The launcher names the evaluation run
(`+export.eval_run_dir`, the hydra run dir of the bs = 1 evaluation) and its
audit manifest (`+export.eval_audit_manifest`, `.temp/audit/<run_id>/manifest.json`).
From these the script takes the checkpoint path, the per-sample table and the
threshold (`eval_metrics.json` `test/threshold_used`, full precision — never a
hand-typed rounding). The checkpoint actually loaded must reproduce the audit
manifest's `checkpoint_identity` fingerprint and size, or the script refuses:
agreement of a discrete metric is a sanity check, not a proof of checkpoint
identity — the fingerprint is.

Gate. For every chain the script recomputes the per-sample metrics with the
production `per_sample_metric_rows` and refuses unless EVERY metric present in
the evaluation's `per_sample_metrics.tsv` and in the recomputed row agrees
within `+export.metric_tol` (default 0). P@L checks the top-L set only;
`AUC-PR_long` checks the ranking over all long-range pairs and `f1_long`
checks the threshold, which P@L cannot see. A map whose metrics do not
reproduce the table is not the map the table describes.

Crop contract. `select_qualitative_cases.py` only admits chains whose
evaluated crop is the whole chain; the request carries `crop_start/crop_end`
and the script refuses if the collate produces anything else.

Hydra: `export` is not a declared group, so every key is appended with `+`:

    python scripts/export_contact_maps.py experiment=trufor_fusion_with_dist_650M \
        data.eval_batch_size=1 data.num_workers=0 \
        +export.requests=paper/qualitative_cases/artifacts_650m_s42_2026/export_requests.tsv \
        +export.tag=structfuse \
        +export.eval_run_dir=logs/paper_650m_trufor_s42_bs1/runs/2026-09-11_08-45-58 \
        +export.eval_audit_manifest=.temp/audit/n87663ed/manifest.json

    python scripts/export_contact_maps.py experiment=trufor_no_templates_650M ... \
        +export.tag=no_templates \
        +export.eval_run_dir=logs/paper_650m_trufor_no_templates_s42_bs1/runs/2026-09-11_08-47-11 \
        +export.eval_audit_manifest=.temp/audit/qux75yw4/manifest.json

`ckpt_path` may be given to override the manifest's path (same file moved); the
fingerprint check still applies.

Both are wired as `launch_preflight.sh --only export-maps` / `--only export-maps-notpl`.
Output: `<hydra run dir>/maps_<tag>/<sample_id>.npz`, `exported_cases.tsv`, `manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import hydra.core.hydra_config
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cost_benchmark import _nullcontext, predict_only, resolve_precision  # noqa: E402

REQUIRED_REQUEST_COLUMNS = ("sample_id", "crop_start", "crop_end")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_requests(path: Path) -> List[Dict[str, object]]:
    import csv
    with path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no requests")
    missing = [c for c in REQUIRED_REQUEST_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate sample_id")
    return rows


# Every per-sample metric the gate compares when the evaluation's table has it.
GATE_METRICS = ("P@L", "P@L/2", "P@L/5", "P@L_short", "P@L_medium", "P@L_long",
                "P@L/2_long", "P@L/5_long", "AUC-PR_long", "f1_long", "precision_long",
                "recall_long", "f1", "precision", "recall")


def read_eval_run(run_dir: Path) -> Dict[str, object]:
    """The bs=1 evaluation this export must reproduce: its per-sample table and
    its threshold, read at full precision from `eval_metrics.json`."""
    import csv
    import math
    tsv = run_dir / "per_sample_metrics.tsv"
    metrics_json = run_dir / "eval_metrics.json"
    for p in (tsv, metrics_json):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — eval_run_dir must be the hydra run dir "
                                    f"of the bs=1 evaluation")
    em = json.loads(metrics_json.read_text())
    if "test/threshold_used" not in em:
        raise ValueError(f"{metrics_json}: no test/threshold_used")
    threshold = float(em["test/threshold_used"])
    reported: Dict[str, Dict[str, float]] = {}
    with tsv.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = [c for c in GATE_METRICS if c in (reader.fieldnames or [])]
        for r in reader:
            reported[r["sample_id"]] = {c: float(r[c]) for c in cols}
    return {"run_dir": str(run_dir), "threshold": threshold, "reported": reported,
            "gate_metrics": cols, "per_sample_tsv": str(tsv), "eval_metrics": str(metrics_json),
            "n_rows": len(reported)}


def verify_checkpoint_identity(ckpt: Path, audit_manifest: Path) -> Dict[str, object]:
    """The loaded checkpoint must be the one the evaluation's audit manifest
    fingerprinted (`src.utils.audit._file_identity`, same method: full sha256
    for small files, sampled head/mid/tail sha256 for large ones)."""
    from src.utils.audit import _file_identity
    man = json.loads(audit_manifest.read_text())
    expected = man.get("checkpoint_identity") or {}
    if not expected.get("fingerprint"):
        raise ValueError(f"{audit_manifest}: no checkpoint_identity fingerprint")
    actual = _file_identity(ckpt) or {}
    same = (actual.get("fingerprint_type") == expected.get("fingerprint_type")
            and actual.get("fingerprint") == expected.get("fingerprint")
            and actual.get("size_bytes") == expected.get("size_bytes"))
    if not same:
        raise ValueError(
            f"checkpoint identity mismatch: loaded {ckpt} -> "
            f"{actual.get('fingerprint_type')}:{actual.get('fingerprint')} "
            f"({actual.get('size_bytes')} B) vs evaluation {man.get('run_id')} "
            f"{expected.get('fingerprint_type')}:{expected.get('fingerprint')} "
            f"({expected.get('size_bytes')} B)")
    return {"audit_manifest": str(audit_manifest), "eval_run_id": man.get("run_id"),
            "eval_task_name": man.get("task_name"), "eval_git_commit": man.get("git_commit"),
            "manifest_ckpt_path": man.get("ckpt_path"), "loaded_ckpt_path": str(ckpt),
            "fingerprint_type": actual.get("fingerprint_type"),
            "fingerprint": actual.get("fingerprint"), "size_bytes": actual.get("size_bytes"),
            "status": "MATCH"}


def load_checkpoint_into(model, ckpt: Path) -> Dict[str, object]:
    """Strict load of the Lightning state_dict; returns provenance for the manifest.

    The saved state IS the EMA snapshot (ModelCheckpoint fires after the EMA
    swap-in), so loading `state_dict` gives the evaluated weights. The module's
    own `load_state_dict` pops its stored `pred_threshold`; that value is the
    threshold at save time under the TRAINING validation batch, which is why the
    reported threshold is passed explicitly and only recorded here.
    """
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = dict(payload["state_dict"] if "state_dict" in payload else payload)
    # ContactLitModule.state_dict() stores the threshold beside the weights; pop
    # it here rather than rely on the module's own override to do it, so a
    # strict load is strict about WEIGHTS only.
    stored_threshold = state.pop("pred_threshold", payload.get("pred_threshold"))
    missing, unexpected = model.load_state_dict(state, strict=True) or ([], [])
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={list(missing)[:5]} "
                         f"unexpected={list(unexpected)[:5]}")
    return {"path": str(ckpt), "sha256": sha256(ckpt),
            "stored_pred_threshold": (float(stored_threshold)
                                      if stored_threshold is not None else None),
            "epoch": payload.get("epoch"), "global_step": payload.get("global_step")}


def apply_determinism(cfg) -> Optional[str]:
    """Mirror what Lightning does for `trainer.deterministic` so the gate can ask
    for bit-exact agreement with the evaluation. Without this, a non-deterministic
    kernel could move a probability by 1e-6 and flip a tie in the top-L set."""
    import os
    det = cfg.trainer.get("deterministic") if "trainer" in cfg else None
    if det in (True, "warn"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=(det == "warn"))
        return str(det)
    return None


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _np(t: torch.Tensor, dtype) -> "np.ndarray":
    import numpy as np
    return t.detach().cpu().to(torch.float32).numpy().astype(dtype)


def export_one(model, datamodule, dataset, pid: str, req: Dict[str, object], *,
               device: torch.device, autocast_dtype, threshold: float, tag: str,
               out_dir: Path) -> Dict[str, object]:
    import numpy as np
    from src.models.utils.metrics import per_sample_metric_rows

    id_pos = {p: i for i, p in enumerate(dataset.ids)}
    item = dataset[id_pos[pid]]
    batch = datamodule._collate_eval([item])
    L = int(batch["seq_len"][0])
    s, e = (int(x) for x in batch["crop_bounds"][0])
    want = (int(req["crop_start"]), int(req["crop_end"]))
    if (s, e) != want or L != want[1] - want[0]:
        raise ValueError(f"{pid}: collate crop {(s, e)} (L={L}) != requested {want}; the "
                         f"selection admits whole-chain crops only")
    full_len = int(item.get("L", L))
    if L != full_len:
        raise ValueError(f"{pid}: crop {L} != chain length {full_len}")

    batch = _to_device(batch, device)
    ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
           if autocast_dtype is not None else _nullcontext())
    with torch.no_grad(), ctx:
        logits = predict_only(model, batch, device, autocast_dtype)
        # As ContactLitModule._step: sigmoid on the logits the net emits, in
        # their dtype. NOT logits.float() — that changes the quantisation of
        # the probabilities and therefore the ties in the top-L ranking.
        prob = torch.sigmoid(logits)
    if prob.dim() == 4:
        prob = prob.squeeze(1)
    valid = batch["pair_mask"] * batch["long_mask"]
    rows = per_sample_metric_rows(
        prob, batch["contact"], valid, batch["pid"], batch.get("subset", ["all"]),
        threshold, seq_lens=batch.get("seq_len"),
        n_templates=batch.get("n_templates_retrieved"), best_sims=batch.get("best_tpl_sim"),
        cluster_ids=batch.get("cluster_id"))
    row = rows[0]

    n_tpl = batch.get("n_templates_retrieved")
    best = batch.get("best_tpl_sim")
    arrays = {
        "contact": _np(batch["contact"][0], np.uint8),
        "pair_mask": _np(batch["pair_mask"][0], np.uint8),
        "valid_mask": _np(valid[0], np.uint8),
        "prior": _np(batch["prior"][0, 0], np.float32),
        "count": _np(batch["count"][0, 0], np.float32),
        "prob": _np(prob[0], np.float32),
    }
    if batch.get("tpl_dist_bins") is not None:
        # Soft per-pair mass over 9 bins (template-aggregated), values in [0,1].
        # An integer dtype here would zero every entry below 1.
        arrays["tpl_dist_bins"] = _np(batch["tpl_dist_bins"][0], np.float32)
    scalars = {
        "sample_id": pid, "seq": batch["seq"][0], "subset": str(batch.get("subset", ["all"])[0]),
        "cluster_id": int(batch["cluster_id"][0]) if "cluster_id" in batch else -1,
        "crop_start": s, "crop_end": e, "seq_len": L, "threshold": float(threshold),
        "prob_dtype": str(prob.dtype), "logits_dtype": str(logits.dtype),
        "n_templates_retrieved": int(n_tpl[0]) if n_tpl is not None else 0,
        "best_tpl_sim": float(best[0]) if best is not None else 0.0, "tag": tag,
        "category": str(req.get("category", "")), "role": str(req.get("role", "")),
        "top1_template_id": str(req.get("top1_template_id", "")),
    }
    np.savez_compressed(out_dir / f"{pid}.npz", **arrays,
                        **{k: np.asarray(v) for k, v in scalars.items()})
    out = {**scalars, "file": f"maps_{tag}/{pid}.npz",
           "n_pos_long": int(row["n_pos_long"]),
           "prior_nonzero_frac": float((arrays["prior"] > 0).mean())}
    for m in GATE_METRICS:
        if m in row:
            out[m] = float(row[m])
    return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    import math

    ex = cfg.get("export", {}) or {}
    for key in ("requests", "tag", "eval_run_dir", "eval_audit_manifest"):
        if ex.get(key) is None:
            raise SystemExit(f"+export.{key} is required")
    for stale in ("threshold", "per_sample_tsv"):
        if ex.get(stale) is not None:
            raise SystemExit(f"+export.{stale} is no longer accepted: the threshold and the "
                             f"per-sample table are read from +export.eval_run_dir at full "
                             f"precision, not typed by hand")
    requests_path = Path(str(ex["requests"]))
    tag = str(ex["tag"])
    tol = float(ex.get("metric_tol", 0.0))
    allow_mismatch = bool(ex.get("allow_metric_mismatch", False))
    if int(cfg.data.get("eval_batch_size") or 0) != 1:
        raise SystemExit("data.eval_batch_size=1 is the reported policy; set it explicitly")

    eval_run = read_eval_run(Path(str(ex["eval_run_dir"])))
    threshold = float(eval_run["threshold"])
    reported: Dict[str, Dict[str, float]] = eval_run["reported"]
    audit_path = Path(str(ex["eval_audit_manifest"]))
    audit = json.loads(audit_path.read_text())
    ckpt = cfg.get("ckpt_path") or audit.get("ckpt_path")
    if not ckpt:
        raise SystemExit("no checkpoint: pass ckpt_path or an audit manifest that records one")
    ckpt = Path(str(ckpt))
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prec_label, autocast_dtype = resolve_precision(cfg)
    deterministic = apply_determinism(cfg)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    maps_dir = out_dir / f"maps_{tag}"
    maps_dir.mkdir(parents=True, exist_ok=True)

    requests = read_requests(requests_path)
    print(f"tag={tag} device={device} precision={prec_label} threshold={threshold!r} "
          f"requests={len(requests)} ckpt={ckpt} eval_run={eval_run['run_dir']}", flush=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    dataset = datamodule.dset_test
    known = set(dataset.ids)
    missing = [r["sample_id"] for r in requests if r["sample_id"] not in known]
    if missing:
        raise SystemExit(f"requested chains not in the evaluated test set: {missing}")
    absent = [r["sample_id"] for r in requests if r["sample_id"] not in reported]
    if absent:
        raise SystemExit(f"requested chains absent from {eval_run['per_sample_tsv']}: {absent}")

    model = hydra.utils.instantiate(cfg.model)
    ckpt_info = load_checkpoint_into(model, ckpt)
    identity = verify_checkpoint_identity(ckpt, audit_path)
    model = model.to(device).eval()

    rows: List[Dict[str, object]] = []
    mismatches: List[str] = []
    for req in requests:
        pid = req["sample_id"]
        row = export_one(model, datamodule, dataset, pid, req, device=device,
                         autocast_dtype=autocast_dtype, threshold=threshold, tag=tag,
                         out_dir=maps_dir)
        worst = 0.0
        for m in eval_run["gate_metrics"]:
            if m not in row:
                continue
            got, want = float(row[m]), float(reported[pid][m])
            if math.isnan(got) and math.isnan(want):
                continue
            diff = abs(got - want) if not (math.isnan(got) or math.isnan(want)) else math.inf
            row[f"{m}_reported"] = want
            worst = max(worst, diff)
            if diff > tol:
                mismatches.append(f"{pid} {m}: recomputed {got!r} vs reported {want!r}")
        row["max_abs_diff_vs_reported"] = worst
        rows.append(row)
        print(f"  {pid}: L={row['seq_len']} P@L_long={row.get('P@L_long', float('nan')):.4f} "
              f"(reported {reported[pid].get('P@L_long', float('nan')):.4f}) "
              f"max|diff|={worst:.3g} prior_nz={row['prior_nonzero_frac']:.4f} "
              f"prob_dtype={row['prob_dtype']}", flush=True)

    fields = sorted({k for r in rows for k in r})
    with (out_dir / "exported_cases.tsv").open("w") as fh:
        fh.write("\t".join(fields) + "\n")
        for r in rows:
            fh.write("\t".join(repr(r[k]) if isinstance(r.get(k), float) else str(r.get(k, ""))
                               for k in fields) + "\n")

    manifest = {
        "tag": tag,
        "checkpoint": ckpt_info,
        "checkpoint_identity_vs_evaluation": identity,
        "evaluation": {k: eval_run[k] for k in ("run_dir", "per_sample_tsv", "eval_metrics",
                                                 "n_rows", "gate_metrics")},
        "threshold_used": threshold,
        "threshold_source": "eval_metrics.json test/threshold_used of the pinned bs=1 "
                            "evaluation, full precision; the checkpoint's stored value is "
                            "recorded under checkpoint.stored_pred_threshold and not used",
        "forward": "scripts.cost_benchmark.predict_only — the same code path Table 13 timed; "
                   "sigmoid taken on the net's logits in their own dtype, as _step does",
        "precision": prec_label, "autocast": autocast_dtype is not None, "device": str(device),
        "deterministic": deterministic, "eval_batch_size": 1,
        "config": {"experiment": str(cfg.get("task_name", "")), "topk": int(cfg.data.get("topk") or 0),
                   "use_template_features": bool(cfg.model.get("use_template_features", True)),
                   "index_dir": str(cfg.data.get("index_dir")),
                   "crop_size": int(cfg.data.get("crop_size") or 0),
                   "crop_mode": "center (eval collate)"},
        "requests": {"path": str(requests_path), "n": len(requests)},
        "gate": {"metrics": eval_run["gate_metrics"], "tolerance": tol,
                 "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL",
                 "note": "P@L checks the top-L set; AUC-PR_long the whole long-range ranking; "
                         "f1_long the threshold"},
        "arrays": {"contact": "uint8", "pair_mask": "uint8", "valid_mask": "uint8",
                   "prior": "float32", "count": "float32", "prob": "float32 (lossless upcast "
                   "of the evaluation-dtype sigmoid; see prob_dtype)",
                   "tpl_dist_bins": "float32 soft histogram mass (if present)"},
        "limits": [
            "prior/count come from the eval collate of THIS run; the no-template run carries "
            "zeros by construction — take the prior from the StructFuse export.",
            "prob is this checkpoint's prediction on the whole-chain crop at batch size 1.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nWrote {maps_dir} ({len(rows)} NPZ), exported_cases.tsv, manifest.json; "
          f"checkpoint identity {identity['status']}; gate {manifest['gate']['status']}",
          flush=True)
    if mismatches and not allow_mismatch:
        raise SystemExit("metric gate FAILED — the exported maps do not reproduce the reported "
                         "per-sample metrics:\n  " + "\n  ".join(mismatches)
                         + "\nSet +export.allow_metric_mismatch=true only with a recorded reason.")


if __name__ == "__main__":
    main()
