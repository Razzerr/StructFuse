import parasail
import numpy as np
import blosum as bl
from typing import Tuple

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_PARASAIL_MATRIX = parasail.matrix_create("ACDEFGHIKLMNPQRSTVWY", 1, -1)
_BLOSUM62 = bl.BLOSUM(62, default=0)


def needleman_wunsch(
    query_seq: str,
    template_seq: str,
    gap: int = -2,
) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """
    Perform global sequence alignment using Needleman-Wunsch algorithm.

    Args:
        query_seq (str): Query sequence (length Lq)
        template_seq (str): Template sequence (length Lt)
        gap (int): Linear gap penalty (default: -2)

    Returns:
        Tuple containing:
            - query_aligned (str): Aligned query with gaps ('-')
            - template_aligned (str): Aligned template with gaps ('-')
            - query_to_template (np.ndarray): Shape (Lq,), maps query index to template index (-1 for gaps)
            - template_to_query (np.ndarray): Shape (Lt,), maps template index to query index (-1 for gaps)
    """
    query_len = len(query_seq)
    template_len = len(template_seq)

    gap_open = abs(gap)
    gap_extend = abs(gap)

    # Run Needleman-Wunsch alignment with traceback
    result = parasail.nw_trace_scan_16(
        query_seq, template_seq, gap_open, gap_extend, _PARASAIL_MATRIX
    )

    query_aligned = result.traceback.query
    template_aligned = result.traceback.ref

    # Fallback: reconstruct from CIGAR if traceback is incomplete
    if len(query_aligned) < min(query_len, template_len):
        log.warning(
            f"Parasail traceback incomplete (got {len(query_aligned)}, expected >={min(query_len, template_len)}). Using CIGAR fallback."
        )
        cigar = result.cigar
        if cigar:
            query_aln_chars = []
            template_aln_chars = []
            query_idx = 0
            template_idx = 0

            for op_len, op_char in cigar.decode:
                if op_char in ("M", "=", "X"):
                    # Match or mismatch
                    for _ in range(op_len):
                        query_aln_chars.append(query_seq[query_idx])
                        template_aln_chars.append(template_seq[template_idx])
                        query_idx += 1
                        template_idx += 1
                elif op_char == "I":
                    # Insertion in query
                    for _ in range(op_len):
                        query_aln_chars.append(query_seq[query_idx])
                        template_aln_chars.append("-")
                        query_idx += 1
                elif op_char == "D":
                    # Deletion in query
                    for _ in range(op_len):
                        query_aln_chars.append("-")
                        template_aln_chars.append(template_seq[template_idx])
                        template_idx += 1

            query_aligned = "".join(query_aln_chars)
            template_aligned = "".join(template_aln_chars)

    # Build residue index mappings
    query_to_template = np.full(query_len, -1, dtype=np.int32)
    template_to_query = np.full(template_len, -1, dtype=np.int32)

    query_idx = 0
    template_idx = 0

    for query_char, template_char in zip(query_aligned, template_aligned):
        has_query = query_char != "-"
        has_template = template_char != "-"

        if has_query and has_template:
            # Both residues present: record the mapping
            query_to_template[query_idx] = template_idx
            template_to_query[template_idx] = query_idx

        # Increment ungapped indices
        if has_query:
            query_idx += 1
        if has_template:
            template_idx += 1

    return query_aligned, template_aligned, query_to_template, template_to_query


def project_prior(
    query_seq: str,
    template_seq: str,
    template_contact: np.ndarray,
    min_seq_sep: int = 0,
    symmetrize: bool = True,
    use_blosum: bool = True,
) -> np.ndarray:
    """
    Project template contact map onto query sequence via sequence alignment.

    Args:
        query_seq (str): Query sequence (length Lq)
        template_seq (str): Template sequence (length Lt)
        template_contact (np.ndarray): Shape (Lt, Lt), binary contact map (0/1)
        min_seq_sep (int): Minimum sequence separation, pairs |i-j| < min_seq_sep set to -1 (or 0 if use_blosum=True)
        symmetrize (bool): Ensure output is symmetric
        use_blosum (bool): If True, weight contacts by BLOSUM62 scores instead of binary 0/1

    Returns:
        np.ndarray: Shape (Lq, Lq), projected contact prior
            - If use_blosum=False (default): values are -1 (unknown), 0 (non-contact), 1 (contact)
            - If use_blosum=True: values are BLOSUM62 scores (-4 to 11) for contacts, 0 for non-contacts/gaps

    Notes:
        - Diagonal is always set to 0 (residues don't contact themselves)
        - Positions where either residue maps to a gap remain -1 (binary) or 0 (BLOSUM)
        - use_blosum=True weights contacts by sequence similarity, giving higher confidence
          to contacts between biochemically similar residues
    """
    if (
        template_contact.ndim != 2
        or template_contact.shape[0] != template_contact.shape[1]
    ):
        raise ValueError("template_contact must be a square matrix")

    template_len = len(template_seq)
    if template_contact.shape[0] != template_len:
        raise ValueError(
            f"template_contact shape {template_contact.shape} does not match "
            f"template_seq length {template_len}"
        )

    _, _, query_to_template, _ = needleman_wunsch(query_seq, template_seq)

    query_len = len(query_seq)
    
    # Use different default values based on mode
    if use_blosum:
        prior = np.zeros((query_len, query_len), dtype=np.float32)
    else:
        prior = np.full((query_len, query_len), -1, dtype=np.int8)

    for i in range(query_len):
        template_i = query_to_template[i]
        if template_i == -1:
            continue  # gap

        for j in range(query_len):
            template_j = query_to_template[j]
            if template_j == -1:
                continue  # gap

            contact = template_contact[template_i, template_j]
            
            if use_blosum:
                # Store BLOSUM score for all aligned positions (contacts and non-contacts)
                blosum_score = _BLOSUM62[template_seq[template_i]][template_seq[template_j]]
                normalized_score = (blosum_score + 4) / 15.0  # Range: [0, 1]
                if contact > 0:
                    # Contact: use BLOSUM score as confidence
                    prior[i, j] = normalized_score
                else:
                    # Non-contact: use negative of BLOSUM score to indicate "confident non-contact"
                    # High BLOSUM = similar residues but no contact = confident negative
                    prior[i, j] = -normalized_score
            else:
                prior[i, j] = contact

    # Apply sequence separation filter
    if min_seq_sep > 0:
        filter_value = 0 if use_blosum else -1
        for i in range(query_len):
            for j in range(query_len):
                if abs(i - j) < min_seq_sep:
                    prior[i, j] = filter_value

    # Clean diagonal
    np.fill_diagonal(prior, 0)

    # Symmetrize if requested
    if symmetrize:
        prior = np.maximum(prior, prior.T)

    return prior
