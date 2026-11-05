"""Cβ contact labels (CASP definition) in src/data/utils/pdb_utils.py.

Pins: the residue set, sequence, mask and Cα coords do not depend on the contact
atom (the ESM cache and FAISS embeddings rely on that); Cβ follows the CASP rule
(CB atom, Cα for Gly, virtual Cβ when CB is missing, Cα when N or C is missing
too); the contact map is computed from the requested atom; the virtual-Cβ
constants reproduce real CB positions.

Plain-python: `python tests/phase0/test_cb_contacts.py`. gemmi/Biopython are
stubbed when absent (local env); the functions under test use duck typing.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for name in ("gemmi", "Bio", "Bio.PDB", "Bio.PDB.Chain"):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.Chain = mod.Structure = mod.PDBParser = object
        sys.modules[name] = mod

from src.data.utils import pdb_utils as pu  # noqa: E402


class Pos:
    def __init__(self, xyz):
        self.x, self.y, self.z = (float(v) for v in xyz)


class Atom:
    def __init__(self, xyz):
        self.pos = Pos(xyz)


class Residue:
    def __init__(self, name, atoms, het="A"):
        self.name, self.het_flag, self._atoms = name, het, atoms

    def find_atom(self, name, altloc):
        xyz = self._atoms.get(name)
        return Atom(xyz) if xyz is not None else None


class Chain(list):
    def __init__(self, name, residues):
        super().__init__(residues)
        self.name = name


# Ideal backbone: |N-CA| 1.458 Å, |CA-C| 1.525 Å, N-CA-C 111°.
_t = np.deg2rad(111.0)
N0, CA0, C0 = 1.458 * np.array([np.cos(_t), np.sin(_t), 0.0]), np.zeros(3), np.array([1.525, 0.0, 0.0])


def _chain():
    return Chain("A", [
        Residue("ALA", {"N": N0, "CA": CA0, "C": C0, "CB": [0.1, 0.2, 1.5]}),     # CB atom
        Residue("GLY", {"N": N0 + 5, "CA": CA0 + 5, "C": C0 + 5}),                # Gly -> Cα
        Residue("SER", {"N": N0 + 10, "CA": CA0 + 10, "C": C0 + 10}),             # virtual
        Residue("VAL", {"CA": CA0 + 15}),                                         # Cα fallback
        Residue("LEU", {"N": N0 + 20, "C": C0 + 20}),                             # no Cα
        Residue("HOH", {"O": [0, 0, 0]}, het="H"),                                # skipped
    ])


def test_cb_follows_the_casp_rule_and_leaves_the_residue_set_alone():
    seq, coords, mask, cb, src = pu.parse_chain_gemmi(_chain())
    assert seq == "AGSVL", seq
    assert mask.tolist() == [1, 1, 1, 1, 0]
    assert np.allclose(coords[3], CA0 + 15) and np.allclose(coords[4], 0)
    assert src.tolist() == [pu.CB_ATOM, pu.CB_GLY_CA, pu.CB_VIRTUAL, pu.CB_CA_FALLBACK, pu.CB_NONE], src
    assert np.allclose(cb[0], [0.1, 0.2, 1.5])
    assert np.allclose(cb[1], CA0 + 5) and np.allclose(cb[3], CA0 + 15) and np.allclose(cb[4], 0)
    assert np.allclose(cb[2], pu.virtual_cb(N0 + 10, CA0 + 10, C0 + 10), atol=1e-5)
    assert abs(np.linalg.norm(cb[2] - (CA0 + 10)) - 1.52) < 0.05  # a Cβ-Cα bond, not Cα itself


def _two_residue_chain(far_ca: bool):
    """Residues 0 and 7 (|i-j| = 7 >= min_sep 6): Cα 9 Å apart, side chains pointing
    at each other so the Cβ are 6 Å apart."""
    filler = [Residue("ALA", {"N": N0 + 40 + k, "CA": CA0 + 40 + k, "C": C0 + 40 + k, "CB": CA0 + 41 + k})
              for k in range(6)]
    r0 = Residue("ALA", {"N": [0, 1, -1], "CA": [0, 0, 0], "C": [0, -1, -1], "CB": [1.5, 0, 0]})
    r7 = Residue("ALA", {"N": [9, 1, -1], "CA": [9, 0, 0], "C": [9, -1, -1], "CB": [7.5, 0, 0]})
    return Chain("B", [r0, *filler, r7])


def test_contact_map_uses_the_requested_atom():
    chain = _two_residue_chain(True)
    seq, coords, mask, cb, _ = pu.parse_chain_gemmi(chain)
    by_ca = pu.contact_map_from_coords(coords, mask, cutoff=8.0, min_sep=6)
    by_cb = pu.contact_map_from_coords(cb, mask, cutoff=8.0, min_sep=6)
    assert by_ca[0, 7] == 0 and by_cb[0, 7] == 1 and by_cb[7, 0] == 1

    original = pu.read_mmcif_structure
    pu.read_mmcif_structure = lambda path: types.SimpleNamespace(name="1ABC", __len__=None)
    try:
        class S(list):
            name = "1ABC"
        pu.read_mmcif_structure = lambda path: S([[chain]])
        out_cb = pu.process_mmcif_file("x.cif", min_len=1, contact_atom="CB")[0]
        out_ca = pu.process_mmcif_file("x.cif", min_len=1, contact_atom="CA")[0]
        assert out_cb["contact"][0, 7] == 1 and out_ca["contact"][0, 7] == 0
        assert out_cb["contact_atom"] == "CB" and out_cb["coords_cb"].shape == (8, 3)
        for key in ("seq", "coords", "mask", "L"):  # labels change, nothing else
            assert np.array_equal(out_cb[key], out_ca[key]), key
        try:
            pu.process_mmcif_file("x.cif", min_len=1, contact_atom="CG")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown contact atom accepted")
    finally:
        pu.read_mmcif_structure = original


def test_virtual_cb_reproduces_real_cb_on_a_full_atom_model():
    """AF2 CASP16 model (full atom): virtual Cβ from N/Cα/C vs the modelled CB."""
    models = sorted((ROOT / "paper/external_baselines/casp16/models/145").glob("*.pdb"))
    if not models:
        raise AssertionError("fixture models missing: paper/external_baselines/casp16/models/145")
    res = {}
    for line in models[0].read_text().splitlines():
        if line.startswith("ATOM") and line[17:20] != "GLY":
            key = (line[21], int(line[22:26]))
            res.setdefault(key, {})[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    d = [np.linalg.norm(pu.virtual_cb(a["N"], a["CA"], a["C"]) - a["CB"])
         for a in res.values() if {"N", "CA", "C", "CB"} <= set(a)]
    assert len(d) > 100, len(d)
    assert np.median(d) < 0.2 and np.percentile(d, 99) < 0.5, (np.median(d), np.percentile(d, 99))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
