#!/usr/bin/env python3
"""Computational cost of retrieval augmentation: per-stage, on a frozen chain set.

No training. No checkpoint update. It measures what one forward pass costs and
where the time goes, so the paper can state the price of retrieval instead of
asserting it is small.

WHAT IS FROZEN BEFORE ANY TIMING (and written into the manifest, so a rerun on
other hardware is comparable):

  * the chain set — an explicit, sorted, deterministic sample of evaluated test
    chains, dumped to `chains.tsv` with its lengths. Every variant times the
    SAME chains in the SAME order.
  * crop 384, batch size 1, one worker in the timed path (workers are recorded,
    not varied — a dataloader worker count changes throughput, not the per-stage
    cost this table is about).
  * precision, device, warmup and repeat counts.

ACCOUNTING IS TWO-LEVEL, which is what keeps it honest:

  Level 1, ADDITIVE and reconciled against the end-to-end figure:
      read      the query NPZ read (`dataset[i]`) — inside `total`, not before it
      data      the real eval collate at bs=1 (`_collate_eval`) — cached-ESM read,
                FAISS search, alignment, projection, padding
      to_device host-to-device transfer of the batch; a known cost, so it gets a
                stage rather than being swept into the residual
      predict   FORWARD ONLY: `_get_embedding`/cached h + relpos + `net(...)`,
                under an explicit autocast. NOT `_step`, which also computes BCE,
                Tversky, three ranking metrics and logs them — none of which an
                inference cost table should contain.
      total     all four, timed as one block
    `residual = total − (data + head)` is reported, not hidden. A large residual
    means the decomposition is wrong and the table must not be quoted.

  Level 2, EXPLANATORY probes of what sits inside `data`. These repeat work
  already counted in level 1 and must NEVER be added to it:
      esm          the cached-embedding read for the same crop
      retrieval    `FaissIndex.topk_precomputed` alone
      prior_build  `PriorBuilder.build_one` — note this performs the FAISS
                   search ITSELF, so `prior_build − retrieval` is the
                   alignment + projection + distance-histogram remainder.

CACHE STATE IS A DIMENSION, NOT AN ASSUMPTION. Precomputed ESM embeddings are
not the same thing as a warm template cache (`max_tpl_cache`), and neither is
the same as a cold start. Each is reported as its own row:

  cold           first touch of each chain, template cache empty
  warm           the same chains again, template cache populated
  prepare        the one-off cost that is NOT per-query: building the ESM cache
                 and the FAISS index (reported from artifact sizes and the
                 recorded build cost, never folded into a per-chain number)

MEMORY is reported in two places because they are loaded very differently:
peak RSS (FAISS index, embeddings, templates, NPZs — this is the large one) and
peak CUDA allocation (the model and its activations).

`cost.*` is NOT a declared config group, so the overrides must be APPENDED:
Hydra refuses a plain `cost.n_chains=3` with "Could not override".

Usage (the measured protocol; scope is K=4 only):
    python scripts/cost_benchmark.py experiment=trufor_fusion_with_dist_650M \\
        data.crop_mode=center data.num_workers=0 \\
        +cost.n_chains=200 +cost.repeats=5

CACHE-PREPARATION COST IS NOT MEASURED HERE, only sized. `esm_cache_read` times
reading a ready embedding, which is not the ESM2 forward that produced it, and
`faiss_index_bytes` is a size on disk, not a build time. Both are reported as
sizes under `cache_preparation` and must never be presented as a per-query cost
or as "the cost of preparing the cache".
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ADDITIVE decomposition. `retrieval` is NOT a sibling of `prior_build`:
# PriorBuilder.build_one performs the FAISS search itself and then aligns and
# projects, so timing both as peers would count the search twice. The additive
# stages are esm + prior_build + head; `retrieval` is timed separately and
# reported as a COMPONENT OF prior_build, with the remainder (alignment +
# projection + distance histograms) derived.
STAGES = ("read", "data", "to_device", "predict")   # additive; reconciled vs `total`
COMPONENTS = ("esm_cache_read", "retrieval", "prior_build")  # explanatory ONLY


# ── timing harness ─────────────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    """CUDA is asynchronous: without this a 'measurement' times the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(store: Dict[str, float], key: str, device: torch.device):
    _sync(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


def summarise(samples: Sequence[float]) -> Dict[str, float]:
    """Median and spread, never a single run and never a bare mean.

    One pass is dominated by whatever else the node was doing; a mean is
    dragged by the same outlier. The median with p10/p90 says what a typical
    chain costs and how variable that is.
    """
    xs = sorted(float(x) for x in samples)
    if not xs:
        return {"n": 0, "median_s": float("nan"), "p10_s": float("nan"),
                "p90_s": float("nan"), "min_s": float("nan"), "total_s": 0.0}

    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    return {"n": len(xs), "median_s": statistics.median(xs), "p10_s": q(0.10),
            "p90_s": q(0.90), "min_s": xs[0], "total_s": sum(xs)}


def reconcile(total: float, stages: Dict[str, float]) -> Dict[str, float]:
    """Stage sum vs the end-to-end figure. The residual is a result, not noise."""
    ssum = sum(stages.values())
    resid = total - ssum
    return {"stage_sum_s": ssum, "residual_s": resid,
            "residual_frac": (resid / total) if total else float("nan")}


# ── static facts that need no timing ───────────────────────────────────────

def warmup_record(rows: Sequence[Dict[str, object]], passes: int, n_chains: int,
                  repeats: int, warm_cache_occupancy: Optional[int]) -> Dict[str, object]:
    """What the manifest must say about warm-up, and the check that it was enough.

    The template cache cannot reach the forward pass, so `predict` must agree
    between the cold and warm rows. A gap there is first-encounter kernel
    selection leaking into whichever cache state ran first — exactly what an
    8-chain warm-up produced on 2026-09-14 (cold 11.8 vs warm 7.9 ms, same p10).
    Recording the coverage and the agreement makes two runs with different
    warm-up schemes distinguishable from the manifest alone; the two 2026-09-14
    manifests were not.
    """
    by_state = {str(r["cache_state"]): r for r in rows}
    cold = by_state.get("cold", {}).get("predict_median_s")
    warm = by_state.get("warm", {}).get("predict_median_s")
    agreement: Dict[str, object] = {
        "predict_median_s_cold": cold,
        "predict_median_s_warm": warm,
        "ratio_cold_over_warm": (float(cold) / float(warm)
                                 if cold and warm and float(warm) > 0 else None),
        "rule": "predict must agree between cache states; the template cache "
                "cannot reach the forward pass, so a gap is a warm-up artefact",
    }
    return {
        "gpu_warmup_passes": passes,
        "gpu_warmup_chains_per_pass": n_chains,
        "gpu_warmup_covers_every_selected_chain": True,
        "warm_template_priming": "one unmeasured build_one pass over every "
                                 "selected chain (warm row only)",
        "warm_cache_occupancy_after_priming": warm_cache_occupancy,
        "measured_repeats": repeats,
        "adequacy_check": agreement,
    }


def count_params(module: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def dir_bytes(path: Optional[Path], patterns: Sequence[str] = ("*",)) -> int:
    """Artifact size on disk. This is the cache-PREPARATION cost, not a per-query one."""
    if path is None or not Path(path).exists():
        return 0
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    total = 0
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                total += p.stat().st_size
    return total


def peak_rss_bytes() -> int:
    """Peak RSS. FAISS, the embeddings and the template NPZs live here, not on GPU."""
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS bytes.
        return maxrss * 1024 if platform.system() == "Linux" else maxrss
    except Exception:  # pragma: no cover - platform guard
        return 0


def environment(device: torch.device, cfg: DictConfig) -> Dict[str, object]:
    env: Dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "num_workers_configured": cfg.data.get("num_workers"),
        "crop_size": cfg.data.get("crop_size"),
        "crop_mode": cfg.data.get("crop_mode"),
        "eval_batch_size": 1,
        "topk": cfg.data.get("topk"),
        "max_tpl_cache": cfg.data.get("max_tpl_cache"),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["cuda"] = torch.version.cuda
    return env




def _frozen_chain_set(dataset, n: int) -> List[str]:
    """Deterministic, explicit, and stratified BY LENGTH.

    Cost here is dominated by sequence length, so the sample must span the
    length distribution. A stride over sorted ids does not guarantee that —
    ids are not ordered by length. Sort by (length, id), then take an even
    stride through that ordering; ties broken by id keep it reproducible.
    """
    lengths = dict(zip(dataset.ids, dataset.cached_lengths))
    ordered = sorted(dataset.ids, key=lambda p: (lengths.get(p, 0), p))
    if n <= 0 or n >= len(ordered):
        return ordered
    step = len(ordered) / n
    return [ordered[int(i * step)] for i in range(n)]


def _clear_template_cache(builder) -> None:
    """Empty the template cache. 'cold' means empty before EACH measured query.

    Clearing once before a loop does not give a cold measurement: the GPU warmup
    passes populate the cache with exactly the chains about to be timed, and at
    K=4 over a few hundred queries the whole working set fits. This is a cache
    of already-parsed templates only — the filesystem cache underneath stays
    warm, and the manifest says so.
    """
    for attr in ("_tpl_cache", "tpl_cache"):
        cache = getattr(builder, attr, None)
        if hasattr(cache, "clear"):
            cache.clear()


def predict_only(model, batch, device: torch.device, autocast_dtype):
    """Forward pass alone, mirroring ContactLitModule._step up to `net(...)`.

    Deliberately excludes the loss terms, the P@L metrics and the logging that
    `_step` performs: an inference-cost row must not include training-time
    bookkeeping. Precision is applied explicitly — reading `bf16-mixed` out of
    the trainer config does nothing on its own, because there is no Trainer here.
    """
    from src.models.components.pair2d_head import relpos_buckets  # local: heavy import

    pair_mask = batch["pair_mask"]
    long_mask = batch["long_mask"]
    prior, count = batch["prior"], batch["count"]
    ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
           if autocast_dtype is not None else _nullcontext())
    with ctx:
        if "h_esm" in batch and "esm_contacts" in batch:
            h, esm_contacts = batch["h_esm"], batch["esm_contacts"]
        else:
            h, esm_contacts = model._get_embedding(
                batch["pid"], batch["seq"], batch["crop_bounds"])
        rel = relpos_buckets(h.shape[1], device).unsqueeze(0) * (
            pair_mask * long_mask).unsqueeze(1)
        tpl = batch.get("tpl_dist_bins") if model.use_tpl_dist_bins else None
        return model.net(h, prior, count, rel, esm_contacts,
                         pair_mask=pair_mask.unsqueeze(1), tpl_dist_bins=tpl)


@contextmanager
def _nullcontext():
    yield


_DTYPES = {"bf16-mixed": torch.bfloat16, "bf16": torch.bfloat16,
           "16-mixed": torch.float16, "16": torch.float16,
           "32": None, "32-true": None, None: None}


def resolve_precision(cfg) -> tuple:
    """(label, autocast dtype). The label goes in the manifest verbatim."""
    label = str(cfg.trainer.get("precision", "32")) if "trainer" in cfg else "32"
    return label, _DTYPES.get(label, None)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    import numpy as np
    from src.data.components.dataset import _choose_crop

    cost = cfg.get("cost", {}) or {}
    n_chains = int(cost.get("n_chains", 200))
    repeats = int(cost.get("repeats", 3))
    warmup = int(cost.get("warmup", 2))
    topk = int(cfg.data.get("topk"))          # scope is the used protocol, K=4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prec_label, autocast_dtype = resolve_precision(cfg)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"device={device} precision={prec_label} topk={topk} n_chains={n_chains} "
          f"repeats={repeats} warmup={warmup}", flush=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    dataset = datamodule.dset_test
    builder = datamodule._prior_builder
    if builder is None:
        raise SystemExit("No PriorBuilder — this config retrieves nothing, so "
                         "there is no retrieval cost to measure.")

    chains = _frozen_chain_set(dataset, n_chains)
    lengths = dict(zip(dataset.ids, dataset.cached_lengths))
    with (out_dir / "chains.tsv").open("w") as fh:
        fh.write("sample_id\tseq_len\n")
        for pid in chains:
            fh.write(f"{pid}\t{lengths[pid]}\n")

    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    id_pos = {pid: i for i, pid in enumerate(dataset.ids)}
    emb_dir = cfg.data.get("esm_embeddings_dir")
    crop = int(cfg.data.get("crop_size") or 384)
    crop_mode = cfg.data.get("crop_mode") or "center"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: List[Dict[str, object]] = []
    warm_cache_occupancy: Optional[int] = None
    with torch.no_grad():
        for cache_state in ("cold", "warm"):
            acc: Dict[str, List[float]] = {k: [] for k in (*STAGES, *COMPONENTS, "total")}
            # Two DIFFERENT warmups, deliberately separate.
            #
            # (a) GPU warmup: kernel selection and autotuning. It must cover
            #     EVERY chain, not a handful: kernel choice is per input SHAPE,
            #     and 200 chains of distinct length are ~150 distinct shapes.
            #     Measured 2026-09-14 with an 8-chain warmup: cold `predict`
            #     median 11.9 ms vs warm 7.9 ms with IDENTICAL p10 (7.1 ms) —
            #     first-encounter shape cost leaking into whichever cache state
            #     runs first. Full-set warmup removes it from both.
            for _ in range(warmup):
                for pid in chains:
                    b = datamodule._collate_eval([dataset[id_pos[pid]]])
                    b = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in b.items()}
                    predict_only(model, b, device, autocast_dtype)
            # (b) Template-cache priming for the WARM variant: an unmeasured
            #     pass over EVERY selected chain. Priming only the first 8 (or
            #     relying on the GPU warmup) leaves the rest missing on the
            #     first measured repeat and hitting on later ones — the row
            #     would then be an average over two different cache states.
            #     `cold` deliberately skips this and clears per query instead.
            if cache_state == "warm":
                for pid in chains:
                    item = dataset[id_pos[pid]]
                    sl = _choose_crop(int(item["L"]), crop,
                                      np.random.RandomState(0), crop_mode)
                    builder.build_one(pid, item["seq"], sl.start, sl.stop,
                                      filter_holdout=False)
                cached = getattr(builder, "_tpl_cache", None)
                warm_cache_occupancy = len(cached) if cached is not None else None
                print(f"  warm priming: {len(chains)} chains, cache holds "
                      f"{warm_cache_occupancy if warm_cache_occupancy is not None else '?'} "
                      f"templates (max_tpl_cache={cfg.data.get('max_tpl_cache')})",
                      flush=True)
            for _ in range(repeats):
                for pid in chains:
                    if cache_state == "cold":
                        _clear_template_cache(builder)
                    t: Dict[str, float] = {}
                    with timed(t, "total", device):
                        with timed(t, "read", device):
                            item = dataset[id_pos[pid]]
                        with timed(t, "data", device):
                            batch = datamodule._collate_eval([item])
                        with timed(t, "to_device", device):
                            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                                     for k, v in batch.items()}
                        with timed(t, "predict", device):
                            predict_only(model, batch, device, autocast_dtype)
                    # Explanatory probes: repeat work already inside `data`, so
                    # they are reported but never added to the additive stages.
                    # Re-clear for cold so the probe sees the same state.
                    if cache_state == "cold":
                        _clear_template_cache(builder)
                    L = int(item["L"])
                    sl = _choose_crop(L, crop, np.random.RandomState(0), crop_mode)
                    with timed(t, "esm_cache_read", device):
                        if emb_dir:
                            _read_cached_esm(emb_dir, pid, sl)
                    with timed(t, "retrieval", device):
                        builder.faiss_index.topk_precomputed(pid, topk, filter_holdout=False)
                    if cache_state == "cold":
                        _clear_template_cache(builder)
                    with timed(t, "prior_build", device):
                        builder.build_one(pid, item["seq"], sl.start, sl.stop,
                                          filter_holdout=False)
                    for k, v in t.items():
                        acc[k].append(v)

            row: Dict[str, object] = {"topk": topk, "cache_state": cache_state,
                                      "precision": prec_label, "n_chains": len(chains),
                                      "repeats": repeats}
            for k in (*STAGES, *COMPONENTS, "total"):
                for stat, val in summarise(acc[k]).items():
                    row[f"{k}_{stat}"] = val
            row.update(reconcile(row["total_total_s"],
                                 {s: row[f"{s}_total_s"] for s in STAGES}))
            row["prior_build_excl_retrieval_total_s"] = (
                row["prior_build_total_s"] - row["retrieval_total_s"])
            rows.append(row)
            print(f"  cache={cache_state}: total {row['total_median_s']*1e3:.1f} ms/chain "
                  f"(read {row['read_median_s']*1e3:.1f}, data {row['data_median_s']*1e3:.1f}, "
                  f"h2d {row['to_device_median_s']*1e3:.1f}, predict "
                  f"{row['predict_median_s']*1e3:.1f}), residual "
                  f"{row['residual_frac']:.1%}", flush=True)

    manifest = {
        "environment": {**environment(device, cfg), "precision": prec_label,
                        "autocast": autocast_dtype is not None},
        "params": count_params(model),
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_cuda_bytes": (torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else 0),
        "cache_preparation": {
            "note": "SIZES ON DISK ONLY. Not a preparation TIME and not a per-query "
                    "cost. The esm_cache_read probe times reading a ready embedding, "
                    "which is not the ESM2 forward that produced it.",
            "faiss_index_bytes": dir_bytes(Path(cfg.data.get("index_dir") or ".")),
            "esm_cache_bytes": dir_bytes(Path(emb_dir) if emb_dir else None),
            "preparation_time_measured": False,
        },
        "frozen": {"chains": len(chains), "crop": crop, "crop_mode": crop_mode,
                   "batch_size": 1, "topk": topk,
                   "chain_selection": "even stride over chains sorted by (length, id)",
                   "chain_set_artifact": "chains.tsv"},
        "warmup": warmup_record(rows, warmup, len(chains), repeats,
                                warm_cache_occupancy),
        "caveats": [
            "predict = forward only (net + embedding + relpos). It excludes the BCE, "
            "Tversky, P@L metrics and logging that ContactLitModule._step performs.",
            "'cold' clears the parsed-template cache before every measured query and "
            "again before each probe. The filesystem cache underneath stays warm, so "
            "this is not a cold-storage number.",
            "retrieval and prior_build repeat work already inside `data` and are "
            "explanatory only — never add them to the additive stages. build_one "
            "performs the FAISS search itself, so prior_build - retrieval is the "
            "alignment + projection remainder.",
            "Scope is K=4, the protocol actually used. No K sweep is measured.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r})
    with (out_dir / "cost_benchmark.tsv").open("w") as fh:
        fh.write("\t".join(fields) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(f, "")) for f in fields) + "\n")
    print(f"\nWrote {out_dir}/cost_benchmark.tsv, chains.tsv, manifest.json")


def _read_cached_esm(emb_dir, pid: str, sl):
    import numpy as np
    data = np.load(Path(emb_dir) / f"{pid}.npz")
    try:
        return data["rep"][sl.start:sl.stop].astype(np.float32)
    finally:
        data.close()


if __name__ == "__main__":
    main()
