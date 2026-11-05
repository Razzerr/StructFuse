"""ESMFold on evaluated test chains, scored on OUR labels (external reference).

Run on the server through `launch_preflight.sh --only esmfold-casp16[-smoke]`, with
the interpreter of the venv made by `scripts/setup_esmfold_env.sh` (structfuse
packages + transformers; weights pre-downloaded, the job runs offline).

Per chain (the subset rows of a pinned bs=1 per-sample table):
1. Labels from the evaluation collate itself (`datamodule._collate_eval`, batch of
   one): the same centre window, contact map and pair x long-range mask StructFuse
   was scored on. Written to labels/<pid>.npz, so other external methods (the CASP16
   AF2/AF3 models, scored locally) use exactly the same labels.
2. ESMFold on the FULL chain sequence the dataset holds (the sequence StructFuse saw),
   final structure-module Calpha coordinates -> esmfold/<pid>.npz.
3. Score with scripts.external_contacts (p = 8/(8+d), per_sample_metric_rows).

Gate: each chain's seq_len, n_valid_long_pairs and n_pos_long must equal the pinned
table — i.e. the labels are the ones the reported metrics were computed on.

Run requires a no-retrieval experiment (data.topk=0) so no FAISS index is loaded,
data.esm_embeddings_dir=null and data.eval_batch_size=1.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import hydra
import hydra.core.hydra_config
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.external_contacts import THRESHOLD, load_labels, save_labels, score_coords  # noqa: E402

GATE_COLUMNS = ("seq_len", "n_valid_long_pairs", "n_pos_long")


def read_reference(path: Path, subset: str) -> dict:
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    picked = {r["sample_id"]: r for r in rows if r.get("subset") == subset}
    if not picked:
        raise SystemExit(f"no rows with subset={subset!r} in {path}")
    return picked


def gate_value(row: dict, key: str) -> int:
    v = row[key]
    return int(float(v))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {}) or {}
    if ext.get("reference_tsv") is None:
        raise SystemExit("+ext.reference_tsv is required (the pinned bs=1 per_sample_metrics.tsv)")
    if int(cfg.data.get("eval_batch_size") or 0) != 1:
        raise SystemExit("data.eval_batch_size=1 is the reported policy; set it explicitly")
    if int(cfg.data.get("topk") or 0) != 0:
        raise SystemExit("use a no-retrieval experiment (data.topk=0): labels only, no FAISS index")
    reference_tsv = Path(str(ext["reference_tsv"]))
    subset = str(ext.get("subset", "casp16"))
    model_id = str(ext.get("model", "facebook/esmfold_v1"))
    chunk = int(ext.get("chunk_size", 64))
    max_chains = ext.get("max_chains")

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    labels_dir, coords_dir = out_dir / "labels", out_dir / "esmfold"
    labels_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)

    reference = read_reference(reference_tsv, subset)
    pids = sorted(reference)
    if max_chains is not None:
        pids = pids[: int(max_chains)]

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    dataset = datamodule.dset_test
    id_pos = {p: i for i, p in enumerate(dataset.ids)}
    missing = [p for p in pids if p not in id_pos]
    if missing:
        raise SystemExit(f"chains of {reference_tsv} not in the evaluated test set: {missing}")

    import transformers
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(model_id)
    model = EsmForProteinFolding.from_pretrained(model_id)
    model.esm = model.esm.half()  # as in the reference ESMFold usage; trunk stays fp32
    model.trunk.set_chunk_size(chunk)
    model = model.to(device).eval()
    revision = getattr(model.config, "_commit_hash", None)
    print(f"ESMFold {model_id}@{revision} transformers={transformers.__version__} "
          f"torch={torch.__version__} device={device} chains={len(pids)} subset={subset}", flush=True)

    rows, mismatches = [], []
    for k, pid in enumerate(pids, 1):
        item = dataset[id_pos[pid]]
        batch = datamodule._collate_eval([item])
        seq = batch["seq"][0]
        if len(seq) != int(item["L"]):
            raise SystemExit(f"{pid}: sequence length {len(seq)} != chain length {item['L']}")
        save_labels(labels_dir / f"{pid}.npz", pid, batch)

        input_ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        ca = out["positions"][-1, 0, :, 1, :].float().cpu().numpy()  # atom14 index 1 = CA
        plddt = out["plddt"][0, :, 1].float().cpu().numpy()          # atom37 index 1 = CA
        if ca.shape[0] != len(seq):
            raise SystemExit(f"{pid}: ESMFold returned {ca.shape[0]} residues for {len(seq)}")
        np.savez_compressed(coords_dir / f"{pid}.npz", ca=ca.astype(np.float32),
                            plddt=plddt.astype(np.float32), seq=np.asarray(seq))

        labels = load_labels(labels_dir / f"{pid}.npz")
        row = score_coords(labels, ca, {"method": "esmfold", "fold_seconds": secs,
                                        "full_len": len(seq), "mean_plddt_ca": float(plddt.mean())})
        for key in GATE_COLUMNS:
            got, want = gate_value(row, key), gate_value(reference[pid], key)
            if got != want:
                mismatches.append(f"{pid} {key}: {got} vs reference {want}")
        rows.append(row)
        print(f"[{k}/{len(pids)}] {pid} L={len(seq)} window={labels['seq_len']} "
              f"P@L_long={row['P@L_long']:.4f} (StructFuse {float(reference[pid]['P@L_long']):.4f}) "
              f"{secs:.1f}s", flush=True)

    fields = sorted({k for r in rows for k in r})
    with (out_dir / "per_sample_metrics_esmfold.tsv").open("w") as fh:
        fh.write("\t".join(fields) + "\n")
        for r in rows:
            fh.write("\t".join(repr(r[k]) if isinstance(r.get(k), float) else str(r.get(k, ""))
                               for k in fields) + "\n")
    manifest = {
        "method": "ESMFold", "model": model_id, "revision": revision,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": "ESM-2 stem fp16, folding trunk fp32 (TF32 matmul allowed)",
        "chunk_size": chunk, "subset": subset, "n_chains": len(rows),
        "reference_tsv": str(reference_tsv), "experiment_for_labels": str(cfg.get("task_name", "")),
        "input_sequence": "full chain sequence held by the dataset (retained coordinate-file residues)",
        "coordinates": "positions[-1] (final structure-module block), atom14 index 1 = CA",
        "score": f"p = 8/(8+d_CA); threshold {THRESHOLD} <=> d <= 8 A",
        "gate": {"columns": list(GATE_COLUMNS), "mismatches": mismatches,
                 "status": "PASS" if not mismatches else "FAIL"},
        "fold_seconds_total": float(sum(r["fold_seconds"] for r in rows)),
        "argv": sys.argv,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nWrote {len(rows)} chains to {out_dir}; gate {manifest['gate']['status']}", flush=True)
    if mismatches:
        raise SystemExit("label gate FAILED:\n  " + "\n  ".join(mismatches))


if __name__ == "__main__":
    main()
