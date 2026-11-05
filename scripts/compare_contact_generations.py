#!/usr/bin/env python3
"""Gate for a contact-definition rebuild: everything except the labels must be unchanged.

Compares two `build_contacts.py` outputs chain by chain (e.g. the Cα generation
`data/processed_2026` and a Cβ rebuild). PASS requires:
  * the same set of chain stems (from each dir's npz_lengths.json);
  * per chain, identical `seq`, `L`, `mask` and Cα `coords` - the residue set and
    sequence are what the ESM cache and the FAISS embeddings are keyed on, so they
    stay valid only if these match exactly;
  * the new files carry `contact_atom`, `coords_cb` and `cb_source`, with a
    symmetric, zero-diagonal contact map.
It also reports how the labels moved (positives before/after, changed pairs) and
where Cβ came from (CB atom / Gly Cα / virtual / Cα fallback).

Usage (server, inside an allocation):
    python scripts/compare_contact_generations.py --old-dir data/processed_2026 \
        --new-dir data/processed_2026cb --out-dir .temp/cb_rebuild_gate --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SAME_FIELDS = ("seq", "L", "mask", "coords")
CB_NAMES = {0: "no_ca", 1: "cb_atom", 2: "gly_ca", 3: "virtual", 4: "ca_fallback"}


def _seq(z) -> str:
    s = z["seq"]
    return str(s.item()) if s.shape == () else str(s)


def compare_one(item: tuple) -> dict:
    stem, old_dir, new_dir = item
    row = {"stem": stem}
    try:
        with np.load(Path(old_dir) / f"{stem}.npz", allow_pickle=True) as o, \
                np.load(Path(new_dir) / f"{stem}.npz", allow_pickle=True) as n:
            row["seq_ok"] = _seq(o) == _seq(n)
            row["L_ok"] = int(o["L"]) == int(n["L"])
            row["mask_ok"] = np.array_equal(o["mask"], n["mask"])
            row["coords_ok"] = np.array_equal(o["coords"], n["coords"])
            missing = [k for k in ("contact_atom", "coords_cb", "cb_source") if k not in n.files]
            row["new_fields_ok"] = not missing
            if missing:
                return row
            row["contact_atom"] = str(n["contact_atom"])
            c_old = o["contact"].astype(bool)
            c_new = n["contact"].astype(bool)
            row["sym_ok"] = bool(np.array_equal(c_new, c_new.T) and not c_new.diagonal().any())
            iu = np.triu_indices(c_new.shape[0], 1)
            row["L"] = int(n["L"])
            row["pos_old"] = int(c_old[iu].sum())
            row["pos_new"] = int(c_new[iu].sum())
            row["pairs_changed"] = int((c_old[iu] != c_new[iu]).sum())
            src = Counter(int(x) for x in n["cb_source"])
            for code, name in CB_NAMES.items():
                row[f"cb_{name}"] = src.get(code, 0)
            mask = n["mask"].astype(bool)
            row["cb_mask_ok"] = bool(np.array_equal(n["cb_source"] > 0, mask))
    except Exception as exc:  # noqa: BLE001 - reported per chain, fails the gate
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-chains", type=int, default=0, help="0 = all (smoke runs only)")
    ap.add_argument("--partial-new", action="store_true",
                    help="smoke: the new dir holds a few entries; chains missing from it are not failures")
    args = ap.parse_args()

    old = json.loads((args.old_dir / "npz_lengths.json").read_text())
    new = json.loads((args.new_dir / "npz_lengths.json").read_text())
    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    stems = sorted(set(old) & set(new))
    if args.max_chains:
        stems = stems[: args.max_chains]
    len_mismatch = sum(old[s] != new[s] for s in stems)

    items = [(s, str(args.old_dir), str(args.new_dir)) for s in stems]
    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        rows = list(pool.imap_unordered(compare_one, items, chunksize=256))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (args.out_dir / "per_chain.tsv").open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: r["stem"]):
            fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")

    def count(flag):
        return sum(1 for r in rows if r.get(flag) is False)

    failures = {f"{f}_mismatch": count(f"{f}_ok") for f in SAME_FIELDS}
    failures.update({
        "new_fields_missing": count("new_fields_ok"), "contact_not_symmetric": count("sym_ok"),
        "cb_source_vs_mask": count("cb_mask_ok"), "errors": sum("error" in r for r in rows),
        "only_in_old": 0 if args.partial_new else len(only_old), "only_in_new": len(only_new),
        "npz_lengths_mismatch": len_mismatch,
    })
    done = [r for r in rows if "pos_new" in r]
    summary = {
        "old_dir": str(args.old_dir), "new_dir": str(args.new_dir), "chains_compared": len(rows),
        "partial_new": bool(args.partial_new),
        "failures": failures, "status": "PASS" if not any(failures.values()) else "FAIL",
        "contact_atom_new": dict(Counter(r.get("contact_atom") for r in done)),
        "positives_old": sum(r["pos_old"] for r in done), "positives_new": sum(r["pos_new"] for r in done),
        "pairs_changed": sum(r["pairs_changed"] for r in done),
        "cb_source_totals": {name: sum(r.get(f"cb_{name}", 0) for r in done) for name in CB_NAMES.values()},
        "chains_with_any_ca_fallback": sum(r.get("cb_ca_fallback", 0) > 0 for r in done),
        "examples": {"only_in_old": only_old[:10], "only_in_new": only_new[:10],
                     "errors": [(r["stem"], r["error"]) for r in rows if "error" in r][:10]},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("chains_compared", "status", "failures", "positives_old",
                                              "positives_new", "cb_source_totals")}, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
