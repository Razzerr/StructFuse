import parasail
import numpy as np
import blosum as bl
from typing import Tuple

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_PARASAIL_MATRIX = parasail.matrix_create("ACDEFGHIKLMNPQRSTVWY", 1, -1)
_BLOSUM62 = bl.BLOSUM(62, default=0)

# Pre-built BLOSUM62 matrix as numpy array for vectorized scoring.
# Replaces O(n²) Python dict lookups with a single numpy indexing op.
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ORDER)}
_BLOSUM_MATRIX = np.zeros((len(_AA_ORDER), len(_AA_ORDER)), dtype=np.float32)
for _i, _aa_i in enumerate(_AA_ORDER):
    for _j, _aa_j in enumerate(_AA_ORDER):
        _BLOSUM_MATRIX[_i, _j] = (_BLOSUM62[_aa_i][_aa_j] + 4) / 15.0


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

    # Vectorized projection: find all query positions that map to a template position
    valid_mask = query_to_template >= 0  # (Lq,)
    valid_indices = np.where(valid_mask)[0]  # query indices with alignment
    template_indices = query_to_template[valid_indices]  # corresponding template indices

    if len(valid_indices) > 0:
        if use_blosum:
            # Extract the sub-matrix of template contacts for aligned positions
            sub_contact = template_contact[np.ix_(template_indices, template_indices)]
            
            # Per-position query↔template substitution scores (vectorized)
            query_aa_indices = np.array(
                [_AA_TO_IDX.get(query_seq[qi], 0) for qi in valid_indices],
                dtype=np.intp,
            )
            template_aa_indices = np.array(
                [_AA_TO_IDX.get(template_seq[ti], 0) for ti in template_indices],
                dtype=np.intp,
            )
            per_pos_score = _BLOSUM_MATRIX[query_aa_indices, template_aa_indices]  # (N,)
            
            # Pairwise confidence = product of per-position query-template scores
            blosum_scores = per_pos_score[:, None] * per_pos_score[None, :]  # (N, N)
            
            # Contact positions get positive BLOSUM, non-contact get negative
            contact_mask = sub_contact > 0
            result = np.where(contact_mask, blosum_scores, -blosum_scores)
            
            prior[np.ix_(valid_indices, valid_indices)] = result
        else:
            # Binary mode: directly copy template contacts for aligned positions
            prior[np.ix_(valid_indices, valid_indices)] = template_contact[
                np.ix_(template_indices, template_indices)
            ]

    # Apply sequence separation filter
    if min_seq_sep > 0:
        ii, jj = np.indices((query_len, query_len))
        close = np.abs(ii - jj) < min_seq_sep
        filter_value = 0 if use_blosum else -1
        prior[close] = filter_value

    # Clean diagonal
    np.fill_diagonal(prior, 0)

    # Symmetrize if requested
    if symmetrize:
        # Use averaging instead of np.maximum to preserve negative (non-contact)
        # evidence in BLOSUM mode. np.maximum would always pick the more-positive
        # value, systematically erasing confident-negative signals.
        prior = (prior + prior.T) / 2.0

    return prior
