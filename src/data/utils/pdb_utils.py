from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain
import numpy as np

AMINOACID_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "ASN": "N",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_chain(chain: Chain) -> tuple:
    """
    Extract sequence, coordinates, and mask from a protein chain.

    Args:
        chain: Bio.PDB.Chain.Chain object

    Returns:
        tuple: (seq, coords, mask)
            - seq: str, 1-letter sequence
            - coords: np.ndarray (L, 3), C_alpha coordinates (0.0 if missing)
            - mask: np.ndarray (L,), 1 if C_alpha present, 0 otherwise
    """
    seq, coords, mask = [], [], []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

        name = res.get_resname()
        if name not in AMINOACID_MAP:
            continue
        seq.append(AMINOACID_MAP[name])

        atom = res["CA"] if "CA" in res else None
        if atom is None:
            # Zeroes to eliminate `nan` problems. controlled by mask anyway.
            coords.append([0.0, 0.0, 0.0])
            mask.append(0)
        else:
            coords.append(atom.coord.astype(np.float32))
            mask.append(1)

    seq = "".join(seq)
    coords = np.array(coords, dtype=np.float32)
    mask = np.array(mask, dtype=np.uint8)
    return seq, coords, mask


def contact_map_from_coords(
    coords: np.ndarray, mask: np.ndarray, cutoff: float = 8.0, min_sep: int = 6
) -> np.ndarray:
    """
    Build a binary contact map from C_alpha coordinates.

    Args:
        coords: np.ndarray (L, 3), C_alpha coordinates
        mask: np.ndarray (L,), 1 if C_alpha present, 0 otherwise
        cutoff: float, distance threshold in Angstroms (default: 8.0)
        min_sep: int, minimum sequence separation (default: 6)

    Returns:
        np.ndarray (L, L), binary contact map (uint8)
    """
    mask_bool = mask.astype(bool)
    coords_masked = coords[mask_bool]
    L = len(mask)
    C = np.zeros((L, L), dtype=np.uint8)
    if coords_masked.shape[0] == 0:
        return C

    idx = np.where(mask_bool)[0]
    dist = np.linalg.norm(
        coords_masked[:, None, :] - coords_masked[None, :, :], axis=-1
    )
    contacts = (dist < cutoff).astype(np.uint8)

    seps = np.abs(idx[:, None] - idx[None, :])
    contacts_filtered = contacts * (seps >= min_sep)

    C[np.ix_(idx, idx)] = contacts_filtered

    # Ensure symmetry & zero diagonal
    C = np.triu(C, 1)
    C = C + C.T
    return C
