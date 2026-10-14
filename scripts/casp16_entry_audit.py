#!/usr/bin/env python3
"""Why do only 26 of the 42 CASP16-flagged entries contribute evaluated chains?

Read-only. For every CASP16-flagged entry in the 2026 split it reports whether
the entry is in test_ids.txt, which of its chains reached npz_lengths.json, which
survived the production chain rule (ContactDataset: length >= 20, not in a
skip-id file), and, for entries with no surviving chain, what the mmCIF
actually contains (polymer type and residue counts per chain, first model).

Self-check: the surviving chains must be exactly 145 chains from 26 entries,
the CASP16 population of every bs=1 evaluation (CASP16 is cap-exempt, so every
surviving chain is evaluated). If not, the chain rule here is wrong and the
per-entry reasons must not be trusted.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SPLITS = Path("data/output_splits_2026/mmcif_final_splits.json")
TEST_IDS = Path("data/output_splits_2026/test_ids.txt")
LENGTHS = Path("data/processed_2026/npz_lengths.json")
SKIP = [Path("data/corrupt_ids.txt"), Path("data/output_splits_2026/no_cluster_ids.txt")]
OUT = Path(".temp/provenance_2026/casp16_entry_audit.tsv")
MIN_LEN, MAX_LEN = 20, 1280  # ContactDataset min_len; build_contacts.py --max_len default


def stems(path):
    return {(s[:-4] if s.endswith(".npz") else s) for s in (l.strip() for l in path.read_text().split()) if s}


def mmcif_chains(path):
    """Per chain of model 1: entity polymer type, amino-acid residues, residues with CA."""
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    rows = []
    for ch in st[0]:
        ent = st.get_entity_of(ch.get_polymer())
        ptype = str(ent.polymer_type).split(".")[-1] if ent else "none"
        aa = [r for r in ch if gemmi.find_tabulated_residue(r.name) is not None
              and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        n_ca = sum(1 for r in aa if r.find_atom("CA", "*") is not None)
        rows.append((ch.name, ptype, len(aa), n_ca))
    return rows


def main():
    for p in (SPLITS, TEST_IDS, LENGTHS, *SKIP):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    splits = json.loads(SPLITS.read_text())
    test_ids = {l.strip() for l in TEST_IDS.read_text().split() if l.strip()}
    lengths = json.loads(LENGTHS.read_text())
    skip = {p.name: stems(p) for p in SKIP}

    by_entry = {}
    for stem, L in lengths.items():
        by_entry.setdefault(stem.split("_")[0].lower(), []).append((stem, int(L)))

    flagged = sorted(p for p, info in splits.items() if info.get("casp16_test_set"))
    out, kept_total, kept_entries, reasons = [], 0, 0, Counter()
    for pid in flagged:
        chains = sorted(by_entry.get(pid, []))
        kept, dropped, filters_hit = [], [], set()
        for stem, L in chains:
            why = [n for n, s in skip.items() if stem in s]
            if L < MIN_LEN:
                why.append(f"len<{MIN_LEN}")
            filters_hit.update(why)
            (dropped if why else kept).append(f"{stem}:{L}" + (f"[{','.join(why)}]" if why else ""))
        kept_total += len(kept)
        kept_entries += bool(kept)

        detail = ""
        if kept:
            reason = "evaluated"
        elif dropped:
            # Chains reached npz_lengths.json; the production chain rule removed them.
            reason = "all_npz_chains_filtered:" + "+".join(sorted(filters_hit))
        else:
            # No chain was written at all: ask the structure why.
            path = Path(splits[pid].get("path", ""))
            if not path.exists():
                reason = "mmcif_missing"
                detail = str(path)
            else:
                try:
                    cc = mmcif_chains(path)
                except Exception as exc:  # report, never swallow
                    reason, detail = "mmcif_parse_error", repr(exc)
                else:
                    detail = "; ".join(f"{n}:{t}:aa{a}:ca{c}" for n, t, a, c in cc)
                    protein = [(a, c) for _, t, a, c in cc if a > 0]
                    if not protein:
                        reason = "no_protein_chain"
                    elif all(c == 0 for _, c in protein):
                        reason = "protein_without_CA"
                    elif all(a > MAX_LEN for a, _ in protein):
                        reason = f"all_protein_chains_>{MAX_LEN}"
                    elif all(a < MIN_LEN for a, _ in protein):
                        reason = f"all_protein_chains_<{MIN_LEN}"
                    else:
                        reason = "UNEXPLAINED"
        if reason != "evaluated":
            reasons[reason] += 1
        out.append({
            "pdb_id": pid, "in_test_ids": pid in test_ids, "reason": reason,
            "n_npz_chains": len(chains), "n_kept": len(kept),
            "kept": " ".join(kept), "dropped": " ".join(dropped), "mmcif_chains": detail,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader(); w.writerows(out)

    print(f"CASP16-flagged entries: {len(flagged)}  (in test_ids.txt: {sum(r['in_test_ids'] for r in out)})")
    print(f"entries with >=1 evaluated chain: {kept_entries}   evaluated chains: {kept_total}")
    ok = (kept_entries, kept_total) == (26, 145)
    print("SELF-CHECK", "MATCH (26 entries / 145 chains)" if ok else
          f"MISMATCH — expected 26 / 145; the reasons below are NOT reliable")
    print("entries without an evaluated chain, by reason:")
    for r, n in reasons.most_common():
        print(f"  {n:3d}  {r}")
    print(f"per-entry table: {OUT}")


if __name__ == "__main__":
    main()
