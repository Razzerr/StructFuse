from typing import List, Tuple, Optional
import gzip

from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain
import numpy as np
import gemmi

# Standard amino acids
AMINOACID_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
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
    # Modified amino acids (common in PDB)
    "MSE": "M",  # Selenomethionine
    "CYX": "C",  # Cysteine involved in disulfide bond
    "HIE": "H",  # Histidine (epsilon tautomer)
    "HID": "H",  # Histidine (delta tautomer)
    "HIP": "H",  # Protonated histidine
    "HSE": "H",  # Histidine (selenocysteine)
    "HSD": "H",  # Histidine (CHARMM naming)
    "HSP": "H",  # Protonated histidine (CHARMM)
    "PTR": "Y",  # Phosphotyrosine
    "SEP": "S",  # Phosphoserine
    "TPO": "T",  # Phosphothreonine
    "CSO": "C",  # S-hydroxycysteine
    "CSS": "C",  # Disulfide bridge cysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "MLY": "K",  # N-dimethyl-lysine
    "M3L": "K",  # N-trimethyl-lysine
    "ALY": "K",  # N-acetyl-lysine
    "PCA": "E",  # Pyroglutamic acid
    "CGU": "E",  # Gamma-carboxy-glutamic acid
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


def read_mmcif_structure(path: str) -> gemmi.Structure:
    """
    Read mmCIF file (plain or gzipped) using gemmi.

    Args:
        path: Path to mmCIF file (.cif or .cif.gz)

    Returns:
        gemmi.Structure object
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="latin-1") as f:
            content = f.read()
        doc = gemmi.cif.read_string(content)
    else:
        doc = gemmi.cif.read(path)
    
    block = doc.sole_block()
    structure = gemmi.make_structure_from_block(block)
    return structure


def parse_chain_gemmi(chain: gemmi.Chain) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Extract sequence, C-alpha coordinates, and mask from a gemmi Chain.

    Args:
        chain: gemmi.Chain object

    Returns:
        tuple: (seq, coords, mask)
            - seq: str, 1-letter sequence
            - coords: np.ndarray (L, 3), C_alpha coordinates (0.0 if missing)
            - mask: np.ndarray (L,), 1 if C_alpha present, 0 otherwise
    """
    seq_list: List[str] = []
    coords_list: List[List[float]] = []
    mask_list: List[int] = []

    for residue in chain:
        # Skip non-polymer residues (water, ligands, etc.)
        if residue.het_flag != "A":  # 'A' = ATOM (polymer), 'H' = HETATM
            # But allow modified residues that are in our map
            if residue.het_flag == "H" and residue.name not in AMINOACID_MAP:
                continue

        res_name = residue.name
        if res_name not in AMINOACID_MAP:
            continue

        seq_list.append(AMINOACID_MAP[res_name])

        # Find C-alpha atom
        ca_atom = residue.find_atom("CA", "*")  # "*" = any altloc
        if ca_atom is None:
            coords_list.append([0.0, 0.0, 0.0])
            mask_list.append(0)
        else:
            pos = ca_atom.pos
            coords_list.append([pos.x, pos.y, pos.z])
            mask_list.append(1)

    seq = "".join(seq_list)
    coords = np.array(coords_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.uint8)
    return seq, coords, mask


def process_mmcif_file(
    path: str,
    cutoff: float = 8.0,
    min_sep: int = 6,
    min_len: int = 20,
    max_len: int = 0,
) -> List[dict]:
    """
    Parse an mmCIF file and extract per-chain contact maps.

    Args:
        path: Path to mmCIF file (.cif or .cif.gz)
        cutoff: C_alpha distance threshold in Angstroms
        min_sep: Minimum sequence separation |i-j|
        min_len: Minimum chain length (residues)
        max_len: Maximum chain length (0 = no limit)

    Returns:
        List[dict]: One dict per valid chain with keys:
            - seq: 1-letter sequence
            - coords: C_alpha coordinates
            - mask: residue validity mask
            - contact: binary contact map
            - L: sequence length
            - pdb_id: PDB identifier
            - chain_id: chain identifier
            - source_path: original file path
            - cutoff, min_sep: preprocessing parameters
    """
    structure = read_mmcif_structure(path)
    
    # Extract PDB ID from structure or filename
    pdb_id = structure.name.lower() if structure.name else ""
    if not pdb_id:
        # Fallback: extract from filename
        from pathlib import Path
        fname = Path(path).name
        if fname.endswith(".cif.gz"):
            pdb_id = fname[:-7].lower()
        elif fname.endswith(".cif"):
            pdb_id = fname[:-4].lower()

    results = []

    # Use first model (most structures have only one)
    if len(structure) == 0:
        return results
    model = structure[0]

    for chain in model:
        seq, coords, mask = parse_chain_gemmi(chain)
        L = len(seq)

        if L < min_len:
            continue
        if max_len > 0 and L > max_len:
            continue

        # Skip chains with no valid CA atoms
        if mask.sum() == 0:
            continue

        contact = contact_map_from_coords(coords, mask, cutoff=cutoff, min_sep=min_sep)

        results.append(
            dict(
                seq=np.array(seq, dtype=object),
                coords=coords.astype(np.float32),
                mask=mask.astype(np.uint8),
                contact=contact.astype(np.uint8),
                L=np.int32(L),
                pdb_id=pdb_id,
                chain_id=str(chain.name),
                source_path=str(path),
                cutoff=np.float32(cutoff),
                min_sep=np.int32(min_sep),
            )
        )

    return results
