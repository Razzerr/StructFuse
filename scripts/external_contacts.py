"""Score contact maps derived from EXTERNAL structure predictions on our labels.

Shared by `scripts/esmfold_predict.py` (server, ESMFold) and the CASP16 baseline
scoring in `paper/external_baselines/casp16/` (local, AF2 / AF3 models from the
CASP16 prediction archive), so every external method is scored by one code path.

Conventions, chosen so that an external method is scored exactly like StructFuse:
- Labels are OUR labels: Calpha-Calpha < 8 A, the evaluation centre window, the
  pair x long-range mask of the evaluation collate (exported per chain by
  `esmfold_predict.py`). Nothing is recomputed from the external model.
- A predicted structure gives a distance d per residue pair. The score is
  p = 8 / (8 + d): monotone in -d (so P@L / AUC-PR rank pairs by predicted
  distance), never saturates, and p >= 0.5 exactly when d <= 8 A, so F1 at the
  threshold 0.5 is the predicted-structure contact call.
- Residues the external model does not cover get no coordinate; every pair
  touching them scores 0 (predicted non-contact), as an unpredicted residue
  would in CASP.
- Metrics come from `per_sample_metric_rows`, the function behind every
  per-sample table in the paper.
"""

from __future__ import annotations

import difflib
from typing import Dict, Optional, Tuple

import numpy as np

CONTACT_A = 8.0
THRESHOLD = 0.5  # p >= 0.5  <=>  d <= CONTACT_A

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}


def prob_from_coords(ca: np.ndarray, start: int, end: int) -> np.ndarray:
    """(L,3) Calpha coordinates of the FULL chain (NaN rows = not predicted) ->
    (end-start, end-start) float32 scores on the evaluation window."""
    sub = np.asarray(ca, dtype=np.float64)[start:end]
    if sub.shape[0] != end - start:
        raise ValueError(f"coordinates cover {sub.shape[0]} residues, window needs {end - start}")
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    p = CONTACT_A / (CONTACT_A + d)
    p[~np.isfinite(p)] = 0.0
    return p.astype(np.float32)


def parse_ca_model(text: str) -> Dict[str, Tuple[str, np.ndarray]]:
    """PDB/CASP-TS text -> {chain_id: (one-letter sequence, (n,3) Calpha coords)}.

    Only ATOM records (plus HETATM MSE) with atom name CA and an amino-acid residue
    name are read, so calcium ions and nucleic acids are ignored. The first
    alternate location wins; parsing stops at the first ENDMDL.
    """
    chains: Dict[str, Dict[Tuple[int, str], Tuple[str, Tuple[float, float, float]]]] = {}
    order: Dict[str, list] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        rec = line[:6]
        if rec != "ATOM  " and not (rec == "HETATM" and line[17:20] == "MSE"):
            continue
        if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
            continue
        resn = line[17:20].strip()
        if resn not in AA3:
            continue
        ch = line[21]
        key = (int(line[22:26]), line[26])
        res = chains.setdefault(ch, {})
        if key in res:
            continue
        res[key] = (AA3[resn], (float(line[30:38]), float(line[38:46]), float(line[46:54])))
        order.setdefault(ch, []).append(key)
    out = {}
    for ch, keys in order.items():
        seq = "".join(chains[ch][k][0] for k in keys)
        xyz = np.array([chains[ch][k][1] for k in keys], dtype=np.float64)
        out[ch] = (seq, xyz)
    return out


def map_sequence(ours: str, theirs: str) -> Tuple[np.ndarray, int]:
    """Index map ours -> theirs from identical matching blocks (difflib, no junk
    heuristic). Returns (idx with -1 where unmatched, number of matched residues).
    Adequate for the near-identical sequences compared here (deposited chain vs
    CASP target construct); a short spurious match is rejected by the caller's
    coverage rule, not here."""
    sm = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    idx = np.full(len(ours), -1, dtype=np.int64)
    n = 0
    for a, b, size in sm.get_matching_blocks():
        if size:
            idx[a:a + size] = np.arange(b, b + size)
            n += size
    return idx, n


def coords_on_our_chain(our_seq: str, model_seq: str, model_xyz: np.ndarray) -> Tuple[np.ndarray, int]:
    idx, n = map_sequence(our_seq, model_seq)
    ca = np.full((len(our_seq), 3), np.nan)
    ok = idx >= 0
    ca[ok] = model_xyz[idx[ok]]
    return ca, n


def score_chain(prob: np.ndarray, contact: np.ndarray, valid: np.ndarray, *, pid: str,
                subset: str, cluster_id: int, seq_len: int) -> dict:
    """One per-sample row, computed by the paper's own per_sample_metric_rows."""
    import torch

    from src.models.utils.metrics import per_sample_metric_rows

    rows = per_sample_metric_rows(
        torch.from_numpy(np.asarray(prob, dtype=np.float32))[None],
        torch.from_numpy(np.asarray(contact, dtype=np.float32))[None],
        torch.from_numpy(np.asarray(valid, dtype=np.float32))[None],
        [pid], [subset], THRESHOLD, seq_lens=torch.tensor([int(seq_len)]),
        cluster_ids=[int(cluster_id)])
    return rows[0]


def save_labels(path, pid: str, batch: dict) -> None:
    """Labels of one chain from an evaluation-collate batch of size one (the window,
    contact map and pair x long-range mask StructFuse is scored on)."""
    L = int(batch["seq_len"][0])
    s, e = (int(x) for x in batch["crop_bounds"][0])
    valid = (batch["pair_mask"] * batch["long_mask"])[0, :L, :L]
    np.savez_compressed(
        path, pid=np.asarray(pid), seq=np.asarray(batch["seq"][0]),
        subset=np.asarray(str(batch["subset"][0])), cluster_id=np.asarray(int(batch["cluster_id"][0])),
        crop_start=np.asarray(s), crop_end=np.asarray(e), seq_len=np.asarray(L),
        contact=batch["contact"][0, :L, :L].numpy().astype(np.uint8),
        valid=valid.numpy().astype(np.uint8))


def load_labels(path) -> dict:
    """Per-chain label NPZ written by esmfold_predict.py."""
    z = np.load(path, allow_pickle=False)
    return {
        "pid": str(z["pid"]), "seq": str(z["seq"]), "subset": str(z["subset"]),
        "cluster_id": int(z["cluster_id"]), "crop_start": int(z["crop_start"]),
        "crop_end": int(z["crop_end"]), "seq_len": int(z["seq_len"]),
        "contact": z["contact"].astype(np.float32), "valid": z["valid"].astype(np.float32),
    }


def score_coords(labels: dict, ca_full: np.ndarray, extra: Optional[dict] = None) -> dict:
    prob = prob_from_coords(ca_full, labels["crop_start"], labels["crop_end"])
    row = score_chain(prob, labels["contact"], labels["valid"], pid=labels["pid"],
                      subset=labels["subset"], cluster_id=labels["cluster_id"],
                      seq_len=labels["seq_len"])
    win = ca_full[labels["crop_start"]:labels["crop_end"]]
    row["window_coverage"] = float(np.isfinite(win).all(axis=1).mean()) if len(win) else float("nan")
    if extra:
        row.update(extra)
    return row
