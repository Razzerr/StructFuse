"""External structure predictions are scored like StructFuse, on our labels.

Pins the conventions of scripts/external_contacts.py: the distance score ranks pairs
by predicted Calpha distance and its 0.5 threshold is the d <= 8 A contact call;
uncovered residues are predicted non-contacts; the PDB/CASP-TS reader ignores ions
and nucleic acids; the sequence map follows identical blocks across gaps.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.external_contacts import (  # noqa: E402
    THRESHOLD,
    coords_on_our_chain,
    map_sequence,
    parse_ca_model,
    prob_from_coords,
    score_chain,
)


def helix_like(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(3.8 * steps, axis=0)


def labels_from(ca: np.ndarray):
    d = np.sqrt(((ca[:, None] - ca[None]) ** 2).sum(-1))
    contact = (d < 8.0).astype(np.float32)
    valid = np.ones_like(contact)
    np.fill_diagonal(valid, 0)
    return contact, valid, d


def test_threshold_is_the_8A_contact_call():
    ca = helix_like(60)
    _, _, d = labels_from(ca)
    p = prob_from_coords(ca, 0, 60)
    assert np.array_equal(p >= THRESHOLD, d <= 8.0)


def test_score_ranks_by_distance():
    ca = helix_like(40, seed=1)
    _, _, d = labels_from(ca)
    p = prob_from_coords(ca, 0, 40)
    iu = np.triu_indices(40, 1)
    # Along decreasing score the distance never decreases (float32 ties allowed).
    order = np.argsort(-p[iu], kind="stable")
    assert np.diff(d[iu][order]).min() > -1e-6


def test_uncovered_residues_are_predicted_non_contacts():
    ca = helix_like(30, seed=2)
    ca[5] = np.nan
    p = prob_from_coords(ca, 0, 30)
    assert (p[5] == 0).all() and (p[:, 5] == 0).all()
    assert (p[np.arange(30) != 5][:, np.arange(30) != 5] > 0).all()


def test_window_is_cut_from_the_full_chain():
    ca = helix_like(500, seed=3)
    p = prob_from_coords(ca, 58, 442)
    assert p.shape == (384, 384)
    assert np.allclose(p, prob_from_coords(ca[58:442], 0, 384))


def test_perfect_structure_scores_like_the_oracle():
    ca = helix_like(120, seed=4)
    contact, valid, _ = labels_from(ca)
    row = score_chain(prob_from_coords(ca, 0, 120), contact, valid, pid="x_A", subset="casp16",
                      cluster_id=1, seq_len=120)
    iu = np.triu_indices(120, 24)
    n_pos = int(contact[iu].sum())
    k = min(120, len(iu[0]))
    assert row["n_pos_long"] == n_pos
    assert abs(row["P@L_long"] - min(n_pos, k) / k) < 1e-6
    assert abs(row["f1_long"] - 1.0) < 1e-9 or n_pos == 0


def test_reader_keeps_protein_calpha_only():
    pdb = "\n".join([
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  CA AALA A   2       2.000   0.000   0.000  0.50  0.00           C",
        "ATOM      4  CA BALA A   2       9.000   0.000   0.000  0.50  0.00           C",
        "HETATM    5  CA  MSE A   3       3.000   0.000   0.000  1.00  0.00           C",
        "HETATM    6 CA    CA A 101       5.000   5.000   5.000  1.00  0.00          CA",
        "ATOM      7  P     A R   1       7.000   0.000   0.000  1.00  0.00           P",
        "ATOM      8  CA  LYS B   1       4.000   0.000   0.000  1.00  0.00           C",
        "ENDMDL",
        "ATOM      9  CA  TRP B   2       4.000   0.000   0.000  1.00  0.00           C",
    ])
    out = parse_ca_model(pdb)
    assert set(out) == {"A", "B"}
    seq, xyz = out["A"]
    assert seq == "GAM" and xyz[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out["B"][0] == "K"


def test_sequence_map_follows_blocks_across_gaps():
    ours = "ACDEFGHIKLMNPQRST"            # deposited chain, residues 7-9 unresolved
    theirs = "MSS" + "ACDEFG" + "WWW" + "HIKLMNPQRST" + "HHHHHH"
    idx, n = map_sequence(ours, theirs)
    assert n == len(ours)
    assert theirs[idx[0]] == "A" and theirs[idx[6]] == "H"
    xyz = np.arange(len(theirs) * 3, dtype=float).reshape(-1, 3)
    ca, n2 = coords_on_our_chain(ours, theirs, xyz)
    assert n2 == n and np.isfinite(ca).all() and np.array_equal(ca[6], xyz[idx[6]])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
