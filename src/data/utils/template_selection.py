"""Which retrieved chains may serve as templates: the same-protein rule.

The same-cluster filter does not guarantee "not the same protein": RCSB's 30 %
clustering can place identical sequences in different clusters (`8RO2_37` and
`9FMD_41`: identical 240 residues, two singleton clusters). So a retrieved chain is
also refused when, in its alignment to the query window, identity over aligned
pairs is >= SAME_PROTEIN_IDENTITY and the aligned pairs cover >=
SAME_PROTEIN_COVERAGE of either chain. Applied in training and evaluation alike.

No parasail import here: the aligner is passed in, so the training path
(`PriorBuilder.build_one`) and `retrieval_identity_audit.py` share this one rule.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

SAME_PROTEIN_IDENTITY = 0.95
SAME_PROTEIN_COVERAGE = 0.90
# Candidates fetched beyond k, so a refused template is replaced without a new search.
CANDIDATE_MARGIN = 16
MAX_CANDIDATES = 4096


def is_same_protein(
    query_seq: str,
    template_seq: str,
    query_to_template: np.ndarray,
    identity: float = SAME_PROTEIN_IDENTITY,
    coverage: float = SAME_PROTEIN_COVERAGE,
) -> bool:
    """Rule on one alignment. `query_to_template[i]` is -1 where query residue i is unaligned."""
    aligned = np.flatnonzero(query_to_template >= 0)
    if aligned.size == 0:
        return False
    q = np.frombuffer(query_seq.encode(), dtype=np.uint8)[aligned]
    t = np.frombuffer(template_seq.encode(), dtype=np.uint8)[query_to_template[aligned]]
    if float((q == t).mean()) < identity:
        return False
    return aligned.size / len(query_seq) >= coverage or aligned.size / len(template_seq) >= coverage


def alignment_weights(
    query_seq: str,
    templates: Sequence[Tuple[str, np.ndarray]],
) -> np.ndarray:
    """Weight of each (template_seq, query_to_template), summing to 1.

    Proportional to the identical aligned residues, so identity and coverage both
    count. Retrieval scores cannot do this job: they differ in the second decimal,
    which made a softmax over them nearly uniform. Uniform when nothing matches.
    """
    q = np.frombuffer(query_seq.encode(), dtype=np.uint8)
    matches = np.zeros(len(templates), dtype=np.float32)
    for k, (template_seq, q2t) in enumerate(templates):
        aligned = np.flatnonzero(q2t >= 0)
        t = np.frombuffer(template_seq.encode(), dtype=np.uint8)
        matches[k] = float((q[aligned] == t[q2t[aligned]]).sum())
    if matches.sum() <= 0:
        return np.full(len(templates), 1.0 / max(1, len(templates)), dtype=np.float32)
    return matches / matches.sum()


def select_templates(
    fetch: Callable[[int], Sequence[Tuple[str, float]]],
    load_seq: Callable[[str], Optional[str]],
    query_seq: str,
    k: int,
    align: Callable[[str, str], np.ndarray],
    identity: Optional[float] = SAME_PROTEIN_IDENTITY,
    coverage: float = SAME_PROTEIN_COVERAGE,
) -> Tuple[List[Tuple[str, float, np.ndarray]], int]:
    """First `k` retrieved chains, in retrieval order, that pass the rule.

    `fetch(n)` returns up to n (id, score) hits; it is called again with a larger n
    only if the margin was used up. `load_seq` returns None for an unreadable template,
    which is skipped. `identity=None` turns the rule off. Returns the kept
    (id, score, query_to_template) and the number of refused chains; fewer than k are
    returned only when the candidates run out.
    """
    alignments: dict = {}
    request = k + CANDIDATE_MARGIN
    while True:
        hits = fetch(request)
        kept: List[Tuple[str, float, np.ndarray]] = []
        refused = 0
        for tpl_id, score in hits:
            tpl_seq = load_seq(tpl_id)
            if tpl_seq is None:
                continue
            if tpl_id not in alignments:
                alignments[tpl_id] = align(query_seq, tpl_seq)
            q2t = alignments[tpl_id]
            if identity is not None and is_same_protein(query_seq, tpl_seq, q2t, identity, coverage):
                refused += 1
                continue
            kept.append((tpl_id, float(score), q2t))
            if len(kept) == k:
                return kept, refused
        if len(hits) < request or request >= MAX_CANDIDATES:
            return kept, refused
        request = min(request * 4, MAX_CANDIDATES)
