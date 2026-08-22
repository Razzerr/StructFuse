"""Verify that training-time FAISS retrieval never returns val/test chains.

Approach: instantiate the datamodule, grab its PriorBuilder (which holds a
FaissIndex configured with holdout_prot_ids from holdout_id_files), query
topk_precomputed for every training PID with filter_holdout=True, and assert
no returned template's protein-ID is in the holdout set. Also spot-check with
filter_holdout=False to confirm the filter actually does something (baseline
retrievals should include holdout hits when the switch is off).

Usage:
    python scripts/verify_no_leak.py experiment=diagnostics/ceiling
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Set, Tuple

import hydra
from omegaconf import DictConfig

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.utils.faiss import _get_protein_id  # noqa: E402


def _scan(
    faiss_index,
    pids: List[str],
    holdout_prot_ids: Set[str],
    filter_holdout: bool,
    topk: int,
    tag: str,
    max_queries: Optional[int] = None,
) -> Tuple[int, int, int, List[Tuple[str, str]]]:
    # max_queries=None → scan ALL pids (pids[:None] returns the whole list).
    n_queries = 0
    n_hits_total = 0
    n_leaks = 0
    leaks: List[Tuple[str, str]] = []
    selected = pids[:max_queries]
    total = len(selected)
    # Progress every 5k queries. A full scan is ~168k FAISS queries per pass and
    # runs for hours; without a heartbeat a stalled job is indistinguishable from
    # a working one, which is how the earlier 12 h timeout ended up telling us
    # nothing. stderr keeps it out of the stdout summary.
    report_every = 5000
    t0 = time.time()
    for pid in selected:
        hits = faiss_index.topk_precomputed(
            pid, topk, filter_holdout=filter_holdout
        )
        n_queries += 1
        if n_queries % report_every == 0 or n_queries == total:
            elapsed = time.time() - t0
            rate = n_queries / elapsed if elapsed > 0 else 0.0
            eta_min = ((total - n_queries) / rate / 60) if rate > 0 else float("nan")
            print(
                f"  [{tag}] filter_holdout={filter_holdout}: "
                f"{n_queries}/{total} queries, {n_leaks} leaked so far, "
                f"{rate:.0f} q/s, ETA {eta_min:.0f} min",
                file=sys.stderr, flush=True,
            )
        for tpl_id, _sim in hits:
            n_hits_total += 1
            if _get_protein_id(tpl_id) in holdout_prot_ids:
                n_leaks += 1
                if len(leaks) < 10:
                    leaks.append((pid, tpl_id))
    print(
        f"  [{tag}] filter_holdout={filter_holdout}: "
        f"queries={n_queries} hits={n_hits_total} leaked={n_leaks}",
        flush=True,
    )
    return n_queries, n_hits_total, n_leaks, leaks


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(f"Instantiating datamodule <{cfg.data._target_}>", flush=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    pb = datamodule._prior_builder
    assert pb is not None, "PriorBuilder not created — check index_dir/topk in config"
    faiss_index = pb.faiss_index
    holdout_prot_ids = faiss_index.holdout_prot_ids
    topk = pb.topk

    print(f"Holdout protein IDs loaded: {len(holdout_prot_ids)}", flush=True)
    assert len(holdout_prot_ids) > 0, (
        "holdout_prot_ids is empty — config likely has no holdout_id_files; "
        "rerun with holdout_id_files set to val/test ID files."
    )

    # Pull training PIDs. Cluster-aware setup exposes dset_trainval; static
    # setup exposes dset_train.
    if datamodule.dset_trainval is not None:
        pids = [datamodule.dset_trainval.ids[i] for i in range(len(datamodule.dset_trainval))]
    elif datamodule.dset_train is not None:
        pids = [datamodule.dset_train.ids[i] for i in range(len(datamodule.dset_train))]
    else:
        raise RuntimeError("Neither dset_trainval nor dset_train available")

    # Query budget: default "all" (full paper-grade scan); override with
    # `+max_queries=500` for a fast pre-flight gate. "all"/None/<=0 → unlimited.
    mq_raw = cfg.get("max_queries", "all")
    if mq_raw in (None, "all", "ALL") or (isinstance(mq_raw, int) and mq_raw <= 0):
        max_queries: Optional[int] = None
    else:
        max_queries = int(mq_raw)
    n_scan = len(pids) if max_queries is None else min(max_queries, len(pids))
    print(f"Scanning {n_scan} / {len(pids)} train PIDs (max_queries={mq_raw})", flush=True)

    # 1. filter_holdout=True — MUST have zero leaks.
    _, _, n_leaks_on, _ = _scan(
        faiss_index, pids, holdout_prot_ids, filter_holdout=True, topk=topk,
        tag="train", max_queries=max_queries,
    )
    # 2. filter_holdout=False — sanity: should leak >0 to prove the filter
    #    actually changes behaviour (if baseline also has zero leaks then the
    #    index simply contains no holdout chains and the test is uninformative).
    _, _, n_leaks_off, leaks_off = _scan(
        faiss_index, pids, holdout_prot_ids, filter_holdout=False, topk=topk,
        tag="train", max_queries=max_queries,
    )

    print("", flush=True)
    print(f"RESULT: filter_holdout=True leaks = {n_leaks_on}", flush=True)
    print(f"        filter_holdout=False leaks = {n_leaks_off}", flush=True)
    if leaks_off:
        print("  Example leaks WITHOUT filter (first 10):", flush=True)
        for q, t in leaks_off[:10]:
            print(f"    query={q}  leaked_tpl={t}", flush=True)

    assert n_leaks_on == 0, f"LEAKAGE: {n_leaks_on} holdout chains retrieved with filter ON"
    if n_leaks_off == 0:
        print(
            "WARNING: no leaks even WITHOUT filter — the index likely does "
            "not contain val/test chains yet. Rebuild with --exclude_ids \"\".",
            flush=True,
        )
    else:
        print("PASS: filter blocks all holdout retrievals.", flush=True)


if __name__ == "__main__":
    main()
