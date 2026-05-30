from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import time
import zipfile

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _load_npz_with_retry(path, retries: int = 3, **kwargs):
    """Load an NPZ file with retries to handle transient NFS errors."""
    for attempt in range(1, retries + 1):
        try:
            return np.load(path, **kwargs)
        except (EOFError, OSError, zipfile.BadZipFile) as exc:
            if attempt == retries:
                raise
            log.warning(
                f"NPZ load failed ({exc}), retry {attempt}/{retries}: {path}"
            )
            time.sleep(0.5 * attempt)


# ---------------------------------------------------------------------------
# PriorBuilder – builds template priors in DataLoader worker processes
# ---------------------------------------------------------------------------

class PriorBuilder:
    """CPU-only helper that encapsulates FAISS retrieval + parasail alignment.

    One instance is created in :pyclass:`ContactDataModule.setup()` and passed
    into :pyfunc:`collate_padded` via a closure.  Because DataLoader workers
    on Linux are **forked**, the heavy read-only objects (FAISS index,
    precomputed embeddings) live in shared copy-on-write memory.  Only the
    per-worker template-NPZ cache is private.

    Args:
        index_dir: Directory containing ``faiss.index``, ``ids.json``,
            ``embeddings.npy``.
        topk: Number of template hits to retrieve per query.
        use_blosum: Weight projected contacts by BLOSUM62 similarity.
        only_positive_transfer: Only transfer positive (contact) evidence.
        min_seq_sep: Zero out contacts closer than this in sequence.
        min_template_similarity: Discard templates below this IP similarity.
        random_retrieval: Ablation flag – use random templates.
        max_tpl_cache: Per-worker LRU cache size for template NPZ files.
    """

    def __init__(
        self,
        index_dir: str,
        topk: int = 4,
        use_blosum: bool = True,
        only_positive_transfer: bool = False,
        min_seq_sep: int = 6,
        min_template_similarity: float = 0.0,
        random_retrieval: bool = False,
        max_tpl_cache: int = 1000,
        # Stage 2 — optional per-pair features derived from template Cα coords.
        # When True, build_one returns a richer dict with the corresponding
        # arrays; otherwise the legacy (prior, count) tuple shape is preserved
        # for backward compatibility.
        compute_dist_bins: bool = False,
        # Holdout filtering. Each file is a text file of chain/PDB IDs (one per
        # line). The union is passed to FaissIndex, which normalises to
        # protein-level IDs and drops them from retrieval when build_one is
        # called with filter_holdout=True (the default). Training loaders
        # pass True; val/test loaders pass False so inference can retrieve
        # from the full index.
        holdout_id_files: Optional[List[str]] = None,
    ):
        from src.models.utils.faiss import FaissIndex

        holdout_ids: Set[str] = set()
        for path in (holdout_id_files or []):
            if not path:
                continue
            try:
                with open(path) as f:
                    for line in f:
                        tok = line.strip()
                        if tok:
                            holdout_ids.add(tok)
            except FileNotFoundError:
                # Silently skip missing files — lets ablation configs omit them.
                pass

        self.faiss_index = FaissIndex(index_dir, holdout_ids=holdout_ids)
        self.topk = int(topk)
        self.use_blosum = bool(use_blosum)
        self.only_positive_transfer = bool(only_positive_transfer)
        self.min_seq_sep = int(min_seq_sep)
        self.min_template_similarity = float(min_template_similarity)
        self.random_retrieval = bool(random_retrieval)

        self.compute_dist_bins = bool(compute_dist_bins)
        self._needs_coords = self.compute_dist_bins

        self._id_to_npz = {m["id"]: m["npz"] for m in self.faiss_index.meta}
        self._tpl_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._max_tpl_cache = int(max_tpl_cache)

    # -- template loading (with FIFO cache) --------------------------------

    def _get_tpl(self, tpl_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load (or return cached) template data.

        Returns ``None`` if the NPZ is unreadable/corrupt after retries —
        callers must skip that template. This keeps training robust to rare
        partial-write corruption in ``data/processed/*.npz`` (e.g. 8tz6_B)
        without crashing an entire epoch.
        """
        if tpl_id in self._tpl_cache:
            return self._tpl_cache[tpl_id]

        if len(self._tpl_cache) >= self._max_tpl_cache:
            oldest = next(iter(self._tpl_cache))
            del self._tpl_cache[oldest]

        npz_path = self._id_to_npz.get(tpl_id)
        try:
            td = _load_npz_with_retry(npz_path, allow_pickle=True)
            tseq_arr = td["seq"]
            tseq = (
                str(tseq_arr.item())
                if isinstance(tseq_arr, np.ndarray) and tseq_arr.shape == ()
                else str(tseq_arr)
            )
            tC = td["contact"].astype(np.uint8).copy()
            entry: Dict[str, np.ndarray] = {"seq": tseq, "contact": tC}
            if self._needs_coords:
                entry["coords"] = td["coords"].astype(np.float32).copy()
            td.close()
        except (EOFError, OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            log.warning(f"Skipping corrupt template NPZ {tpl_id} ({npz_path}): {exc}")
            return None
        self._tpl_cache[tpl_id] = entry
        return self._tpl_cache[tpl_id]

    # -- main entry point ---------------------------------------------------

    def build_one(
        self,
        pid: str,
        full_seq: str,
        crop_start: int,
        crop_end: int,
        filter_holdout: bool = True,
    ):
        """Build aggregated template prior for one sample.

        Returns:
            Legacy mode (no Stage 2 flags set): ``(prior, count)`` tuple of
            two ``(Lc, Lc)`` float32 arrays. All-zeros if no templates found.

            Rich mode (compute_dist_bins=True):
            dict with keys:
              - "prior"      : (Lc, Lc) float32
              - "count"      : (Lc, Lc) float32
              - "dist_bins"  : (9, Lc, Lc) float32 soft histogram
        """
        from src.data.utils.align import (
            project_prior,
            project_distance,
            needleman_wunsch,
            N_DIST_BINS,
        )

        Lc = crop_end - crop_start
        crop_seq = full_seq[crop_start:crop_end]

        rich = self._needs_coords

        def _empty_return():
            prior = np.zeros((Lc, Lc), np.float32)
            count = np.zeros((Lc, Lc), np.float32)
            if not rich:
                return prior, count
            out = {"prior": prior, "count": count}
            if self.compute_dist_bins:
                out["dist_bins"] = np.zeros((N_DIST_BINS, Lc, Lc), np.float32)
            # Per-chain template stats — always present in rich mode for paper-grade
            # per-protein analyses (template-quality stratification, paired tests).
            out["n_templates_retrieved"] = 0
            out["best_tpl_sim"] = 0.0
            return out

        if self.topk <= 0:
            return _empty_return()

        hits = self.faiss_index.topk_precomputed(
            pid,
            self.topk,
            min_similarity=self.min_template_similarity,
            random_retrieval=self.random_retrieval,
            filter_holdout=filter_holdout,
        )
        if not hits:
            return _empty_return()

        # Softmax weights from cosine similarities
        sims = np.array([h[1] for h in hits], dtype=np.float32)
        w = np.exp(sims - sims.max())
        w /= w.sum() + 1e-8

        prior_acc = np.zeros((Lc, Lc), dtype=np.float32)
        count_acc = np.zeros((Lc, Lc), dtype=np.float32)

        # Stage 2 accumulators
        dist_bins_acc = None
        if rich and self.compute_dist_bins:
            dist_bins_acc = np.zeros((Lc, Lc, N_DIST_BINS), dtype=np.float32)

        for (tpl_id, _sim), wk in zip(hits, w):
            tpl = self._get_tpl(tpl_id)
            if tpl is None:
                continue  # corrupt NPZ — logged in _get_tpl, skip this template
            # Compute NW once; feed into every projector to avoid recomputing
            # the O(Lq·Lt) alignment for each feature type.
            _, _, q2t, _ = needleman_wunsch(crop_seq, tpl["seq"])

            Pk = project_prior(
                crop_seq,
                tpl["seq"],
                tpl["contact"],
                min_seq_sep=0,
                symmetrize=True,
                use_blosum=self.use_blosum,
                query_to_template=q2t,
            )

            if self.only_positive_transfer:
                known = Pk == 1
                Pk_pos = known.astype(np.float32)
                cnt = known.astype(np.float32)
            else:
                known = Pk != -1
                Pk_pos = np.clip(Pk, 0, 1).astype(np.float32)
                if self.use_blosum:
                    cnt = (Pk_pos != 0).astype(np.float32)
                else:
                    cnt = known.astype(np.float32)

            if self.min_seq_sep > 0:
                ii, jj = np.indices((Lc, Lc))
                close = np.abs(ii - jj) < self.min_seq_sep
                Pk_pos[close] = 0.0
                cnt[close] = 0.0

            prior_acc += wk * Pk_pos
            count_acc += cnt

            if rich and self.compute_dist_bins:
                dbin = project_distance(
                    crop_seq,
                    tpl["seq"],
                    tpl["coords"],
                    q2t,
                    min_seq_sep=self.min_seq_sep,
                    symmetrize=True,
                )  # (Lc, Lc) int8 ∈ [0..8]
                oh = np.eye(N_DIST_BINS, dtype=np.float32)[dbin]  # (Lc, Lc, 9)
                dist_bins_acc += wk * oh

        if not rich:
            return prior_acc, count_acc

        out: Dict[str, np.ndarray] = {"prior": prior_acc, "count": count_acc}

        if self.compute_dist_bins:
            # Transpose to channel-first to match pair feature convention.
            out["dist_bins"] = dist_bins_acc.transpose(2, 0, 1).astype(np.float32)

        # Per-chain template stats (rich mode only) — used downstream for paper-grade
        # per-protein analyses (paired Wilcoxon, template-quality stratification).
        out["n_templates_retrieved"] = int(len(hits))
        out["best_tpl_sim"] = float(sims.max()) if len(hits) > 0 else 0.0

        return out


# Subset definitions for test set evaluation
SUBSET_ALL = "all"
SUBSET_GOLD = "gold"  # temporal-only, no casp16, no cluster_promoted
SUBSET_CASP16 = "casp16"
SUBSET_CLUSTER_PROMOTED = "cluster_promoted"


def load_subset_mapping(splits_json_path: Path) -> Dict[str, str]:
    """
    Load subset membership from mmcif_final_splits.json.
    
    Returns dict: pdb_id (lowercase) -> subset name
    Subset priority: casp16 > cluster_promoted > gold (temporal-only)
    """
    if not splits_json_path.exists():
        log.warning(f"Splits file not found: {splits_json_path}. Subset info unavailable.")
        return {}
    
    with open(splits_json_path) as f:
        data = json.load(f)
    
    subset_map = {}
    for pdb_id, info in data.items():
        if info.get("split") != "test":
            continue
        
        pdb_lower = pdb_id.lower()
        if info.get("casp16_test_set", False):
            subset_map[pdb_lower] = SUBSET_CASP16
        elif info.get("cluster_promoted", False):
            subset_map[pdb_lower] = SUBSET_CLUSTER_PROMOTED
        else:
            subset_map[pdb_lower] = SUBSET_GOLD
    
    log.info(f"Loaded subset mapping: {len(subset_map)} test entries")
    return subset_map


# Module-level cache so npz_lengths.json is read from disk only once per process
_NPZ_LENGTHS_CACHE: Dict[str, Dict[str, int]] = {}


def _load_npz_lengths(root: Path) -> Dict[str, int]:
    """
    Load ``npz_lengths.json`` (stem -> sequence length L) from *root*.

    The file must be pre-built with::

        python scripts/build_npz_lengths.py --processed_dir <root>

    Raises FileNotFoundError with a helpful message if missing.
    """
    key = str(root)
    if key in _NPZ_LENGTHS_CACHE:
        return _NPZ_LENGTHS_CACHE[key]

    index_path = root / "npz_lengths.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"NPZ length index not found: {index_path}\n"
            f"Build it first with:  python scripts/build_npz_lengths.py --processed_dir {root}"
        )

    log.info(f"Loading NPZ length index from {index_path}")
    with open(index_path) as f:
        lengths = json.load(f)
    log.info(f"NPZ length index: {len(lengths)} entries")

    _NPZ_LENGTHS_CACHE[key] = lengths
    return lengths


class ContactDataset(Dataset):
    """
    PyTorch dataset for protein contact prediction.

    Loads per-chain NPZ files produced by scripts/build_contacts.py.

    Args:
        id_list_file (Path): Path to file containing PDB IDs (one per line)
        root (Path): Directory containing processed NPZ files
        min_len (int): Minimum sequence length to include
        splits_json_path (Path): Path to mmcif_final_splits.json for subset info
        index_dir (Path): Path to FAISS index directory (ids.json) for cluster info

    Returns:
        Dict with keys: pid, seq, contact, mask, L, subset
    """

    def __init__(
        self,
        id_list_file: Path,
        root: Path = Path("data/processed"),
        min_len: int = 1,
        splits_json_path: Optional[Path] = None,
        exclude_subsets: Optional[List[str]] = None,
        index_dir: Optional[Path] = None,
        skip_ids_file: Optional[Path] = None,
    ):
        self.root = root
        self.ids = []
        self._chain2cluster: Dict[str, int] = {}

        # Load subset mapping if available
        if splits_json_path is not None:
            self.subset_map = load_subset_mapping(Path(splits_json_path))
        else:
            self.subset_map = {}

        # Load pre-built stem -> L index (see scripts/build_npz_lengths.py)
        index = _load_npz_lengths(root)
        self._length_index = index

        # Load PDB IDs from list file
        raw_ids = [line.strip() for line in open(id_list_file) if line.strip()]
        raw_id_set = set(raw_ids)
        _exclude = set(exclude_subsets) if exclude_subsets else set()

        # Skip-list of chain stems whose NPZ (processed or precomputed ESM) is
        # known-corrupt. Training crashes hard if the query side fails to load,
        # so such chains must be excluded from the dataset upfront.
        skip_stems: Set[str] = set()
        if skip_ids_file is not None:
            p = Path(skip_ids_file)
            if p.exists():
                with open(p) as f:
                    skip_stems = {ln.strip() for ln in f if ln.strip()}
                log.info(f"  Loaded {len(skip_stems)} skip stems from {p}")
            else:
                log.warning(f"  skip_ids_file not found: {p}")

        # Filter index entries by PDB ID and min_len (pure dict lookups, instant)
        n_excluded = 0
        n_skipped_corrupt = 0
        for stem, L in index.items():
            pdb_id = stem.split("_")[0]
            if pdb_id in raw_id_set and L >= min_len:
                if _exclude and self.subset_map.get(pdb_id.lower(), SUBSET_ALL) in _exclude:
                    n_excluded += 1
                    continue
                if stem in skip_stems:
                    n_skipped_corrupt += 1
                    continue
                self.ids.append(stem)
        if n_skipped_corrupt:
            log.info(f"  Skipped {n_skipped_corrupt} chains from skip_ids_file")

        log.info(f"Loaded {len(self.ids)} chains (min_len={min_len})")
        if n_excluded:
            log.info(f"  Excluded {n_excluded} chains from subsets: {sorted(_exclude)}")

        # Load cluster mapping from FAISS index metadata
        if index_dir is not None:
            ids_json = Path(index_dir) / "ids.json"
            if ids_json.exists():
                with open(ids_json) as f:
                    index_meta = json.load(f)
                self._chain2cluster = {
                    m["id"]: m.get("cluster_id", -1) for m in index_meta
                }
                n_matched = sum(1 for s in self.ids if s in self._chain2cluster)
                log.info(f"  Cluster info: {n_matched}/{len(self.ids)} chains matched")
            else:
                log.warning(f"  ids.json not found in {index_dir}, cluster sampling disabled")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def cached_lengths(self) -> List[int]:
        """Return sequence lengths from the in-memory index (no NPZ I/O)."""
        return [self._length_index[stem] for stem in self.ids]

    @property
    def cluster_ids(self) -> Optional[List[int]]:
        """Return cluster IDs for each sample, or None if not available."""
        if not self._chain2cluster:
            return None
        return [self._chain2cluster.get(stem, -1) for stem in self.ids]

    @property
    def has_cluster_info(self) -> bool:
        return bool(self._chain2cluster)
    
    def _get_subset(self, pid: str) -> str:
        """Get subset for a protein ID (format: XXXX_Chain)."""
        pdb_id = pid.split("_")[0].lower()
        return self.subset_map.get(pdb_id, SUBSET_ALL)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.ids[idx]
        data = _load_npz_with_retry(self.root / f"{pid}.npz", allow_pickle=True)
        try:
            # seq may be a 0-d object array
            seq_arr = data["seq"]
            seq = (
                str(seq_arr.item())
                if isinstance(seq_arr, np.ndarray) and seq_arr.shape == ()
                else str(seq_arr)
            )

            contact = data["contact"].astype(np.uint8)  # (L, L)
            mask = data["mask"].astype(np.uint8)  # (L,) - 1 if CA present
            L = int(data["L"])
            subset = self._get_subset(pid)
            # Stage 4: optionally include query Cα coords for distogram target.
            coords = None
            if "coords" in data.files:
                coords = data["coords"].astype(np.float32)
        finally:
            data.close()

        return {
            "pid": pid,
            "seq": seq,
            "contact": contact,
            "mask": mask,
            "L": L,
            "subset": subset,
            "coords": coords,
        }


def _choose_crop(
    L: int, crop_size: Optional[int], rng: np.random.RandomState, mode: str
) -> slice:
    if crop_size is None or crop_size <= 0 or crop_size >= L:
        return slice(0, L)

    if mode == "center":
        start = max(0, (L - crop_size) // 2)
    elif mode == "random":
        start = 0 if L == crop_size else rng.randint(0, L - crop_size + 1)
    else:
        # Fallback to no crop
        return slice(0, L)

    return slice(start, start + crop_size)


def collate_padded(
    batch: List[Dict],
    *,
    crop_size: Optional[int] = None,
    crop_mode: str = "random",  # "random" | "center" | "none"
    min_seq_sep: int = 0,  # build long-range mask: valid where |i-j|>=min_seq_sep
    seed: Optional[int] = None,
    include_diagonal: bool = False,  # usually set diagonal to 0 in pair masks
    prior_builder: Optional[PriorBuilder] = None,
    esm_embeddings_dir: Optional[Path] = None,
    filter_holdout: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function with cropping and padding to batch max length.

    Args:
        batch (List[Dict]): List of dataset items
        crop_size (Optional[int]): Target crop size (None = no crop)
        crop_mode (str): Crop mode - "random", "center", or "none"
        min_seq_sep (int): Minimum sequence separation for long_mask
        seed (Optional[int]): Random seed for reproducibility
        include_diagonal (bool): Include diagonal in pair masks
        prior_builder (Optional[PriorBuilder]): If provided, build template
            priors for each sample inside the DataLoader worker process.
        esm_embeddings_dir (Optional[Path]): Directory with precomputed ESM2
            embeddings ({stem}.npz with 'rep' and 'contacts' keys).
            When set, crops full-length embeddings and adds them to the batch.

    Returns:
        Dict with keys:
            - pid: list[str] - Protein IDs
            - seq: list[str] - Sequences (for ESM2 tokenization)
            - subset: list[str] - Subset membership (gold, casp16, cluster_promoted, all)
            - crop_bounds: (B, 2) int64 - Crop start/end indices
            - contact: (B, Lmax, Lmax) float32 - Binary contact maps
            - pair_mask: (B, Lmax, Lmax) float32 - Valid residue pairs
            - long_mask: (B, Lmax, Lmax) float32 - Long-range pairs (|i-j| >= min_seq_sep)
            - prior: (B, 1, Lmax, Lmax) float32 - Template priors (if prior_builder)
            - count: (B, 1, Lmax, Lmax) float32 - Template counts (if prior_builder)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    cropped = []
    for item in batch:
        L = item["L"]
        crop_slice = slice(0, L)

        if (
            crop_mode.lower() in ("random", "center")
            and crop_size
            and crop_size > 0
            and crop_size < L
        ):
            crop_slice = _choose_crop(L, crop_size, rng, crop_mode.lower())

        crop_start, crop_end = crop_slice.start, crop_slice.stop
        cropped.append(
            {
                "pid": item["pid"],
                "seq": item["seq"],
                "subset": item.get("subset", SUBSET_ALL),
                "contact": item["contact"][crop_start:crop_end, crop_start:crop_end],
                "mask": item["mask"][crop_start:crop_end],
                "L": crop_end - crop_start,
                "crop_bounds": (crop_start, crop_end),
                "coords": (
                    item["coords"][crop_start:crop_end]
                    if item.get("coords") is not None
                    else None
                ),
            }
        )

    # Pad to batch max
    Ls = [x["L"] for x in cropped]
    Lmax = max(Ls)
    B = len(cropped)

    contact = torch.zeros((B, Lmax, Lmax), dtype=torch.float32)
    pair_mask = torch.zeros((B, Lmax, Lmax), dtype=torch.float32)
    residue_mask = torch.zeros((B, Lmax), dtype=torch.float32)

    pids: List[str] = []
    seqs: List[str] = []
    subsets: List[str] = []
    crop_bounds = torch.zeros(B, 2, dtype=torch.long)

    for b, item in enumerate(cropped):
        pids.append(item["pid"])
        seqs.append(item["seq"])
        subsets.append(item["subset"])

        L = item["L"]
        crop_bounds[b, 0] = item["crop_bounds"][0]
        crop_bounds[b, 1] = item["crop_bounds"][1]

        contact[b, :L, :L] = torch.from_numpy(item["contact"].astype(np.float32))
        residue_mask[b, :L] = torch.from_numpy(item["mask"].astype(np.float32))

        # Build pair mask (1 only where both residues exist)
        res_mask_1d = residue_mask[b, :L]
        pair_mask_2d = torch.matmul(
            res_mask_1d.view(L, 1), res_mask_1d.view(1, L)
        )  # (L, L)
        if not include_diagonal:
            pair_mask_2d.fill_diagonal_(0.0)
        pair_mask[b, :L, :L] = pair_mask_2d

    # Build long-range mask (filter pairs with |i-j| < min_seq_sep)
    long_mask = pair_mask.clone()
    if min_seq_sep > 0 and Lmax > 0:
        ii, jj = torch.meshgrid(torch.arange(Lmax), torch.arange(Lmax), indexing="ij")
        sep_ok = (torch.abs(ii - jj) >= min_seq_sep).float()
        long_mask = long_mask * sep_ok

    batch_out = {
        "pid": pids,
        "seq": seqs,
        "subset": subsets,
        "crop_bounds": crop_bounds,  # (B, 2)
        "contact": contact,  # (B, Lmax, Lmax)
        "pair_mask": pair_mask,  # (B, Lmax, Lmax)
        "long_mask": long_mask,  # (B, Lmax, Lmax)
    }

    # ── Template priors (built in DataLoader worker) ──
    if prior_builder is not None:
        prior = torch.zeros((B, 1, Lmax, Lmax), dtype=torch.float32)
        count = torch.zeros((B, 1, Lmax, Lmax), dtype=torch.float32)

        # Stage 2 rich features — pre-allocate only if PriorBuilder is
        # configured to produce them; otherwise save the memory.
        dist_bins = None
        if getattr(prior_builder, "compute_dist_bins", False):
            # N_DIST_BINS = 9 channels (unknown + 8 finite bins).
            from src.data.utils.align import N_DIST_BINS
            dist_bins = torch.zeros(
                (B, N_DIST_BINS, Lmax, Lmax), dtype=torch.float32
            )

        # Per-chain template stats — populated whenever build_one returns rich dict.
        # Used by test_step for per-protein dump (paired Wilcoxon, stratification).
        n_templates_retrieved = torch.zeros(B, dtype=torch.long)
        best_tpl_sim = torch.zeros(B, dtype=torch.float32)

        for b, item in enumerate(cropped):
            Lc = item["L"]
            cb = item["crop_bounds"]
            built = prior_builder.build_one(
                item["pid"], item["seq"], cb[0], cb[1],
                filter_holdout=filter_holdout,
            )
            if isinstance(built, tuple):
                p_np, c_np = built
                extra = {}
            else:
                p_np = built["prior"]
                c_np = built["count"]
                extra = built

            L_use = min(Lc, Lmax)
            prior[b, 0, :L_use, :L_use] = torch.from_numpy(p_np[:L_use, :L_use])
            count[b, 0, :L_use, :L_use] = torch.from_numpy(c_np[:L_use, :L_use])

            if dist_bins is not None and "dist_bins" in extra:
                arr = extra["dist_bins"]  # (9, Lc, Lc)
                dist_bins[b, :, :L_use, :L_use] = torch.from_numpy(
                    arr[:, :L_use, :L_use]
                )

            n_templates_retrieved[b] = int(extra.get("n_templates_retrieved", 0))
            best_tpl_sim[b] = float(extra.get("best_tpl_sim", 0.0))

        batch_out["prior"] = prior
        batch_out["count"] = count
        batch_out["n_templates_retrieved"] = n_templates_retrieved
        batch_out["best_tpl_sim"] = best_tpl_sim
        if dist_bins is not None:
            batch_out["tpl_dist_bins"] = dist_bins

    # ── Precomputed ESM2 embeddings (loaded + cropped in DataLoader worker) ──
    if esm_embeddings_dir is not None:
        # Determine d_esm from first file
        first_pid = pids[0]
        first_emb = _load_npz_with_retry(esm_embeddings_dir / f"{first_pid}.npz")
        d_esm = first_emb["rep"].shape[1]
        first_emb.close()

        h_esm = torch.zeros((B, Lmax, d_esm), dtype=torch.float32)
        esm_contacts = torch.zeros((B, 1, Lmax, Lmax), dtype=torch.float32)

        for b, item in enumerate(cropped):
            pid = item["pid"]
            cb = item["crop_bounds"]
            Lc = item["L"]

            emb_data = _load_npz_with_retry(esm_embeddings_dir / f"{pid}.npz")
            rep_full = emb_data["rep"]        # (L_full, d_esm) float16
            cont_full = emb_data["contacts"]  # (L_full, L_full) float16
            emb_data.close()

            # Crop from full-length embeddings (embedding may be shorter than
            # crop window if the sequence was truncated during precomputation)
            rep_crop = rep_full[cb[0]:cb[1]].astype(np.float32)
            cont_crop = cont_full[cb[0]:cb[1], cb[0]:cb[1]].astype(np.float32)

            L_use = min(rep_crop.shape[0], Lmax)
            h_esm[b, :L_use, :] = torch.from_numpy(rep_crop[:L_use])
            esm_contacts[b, 0, :L_use, :L_use] = torch.from_numpy(cont_crop[:L_use, :L_use])

        batch_out["h_esm"] = h_esm
        batch_out["esm_contacts"] = esm_contacts

    return batch_out


class ClusterBucketBatchSampler(Sampler[List[int]]):
    """
    Cluster-aware length-bucketed batch sampler.

    Each epoch, samples ONE random chain per cluster, then groups
    the selected chains into length-based buckets to reduce padding waste.

    This eliminates redundancy at the 30% sequence-identity level:
    chains in the same cluster produce near-identical gradients, so
    seeing all of them every epoch wastes compute.

    Args:
        cluster_ids (List[int]): Cluster ID for each dataset sample
        lengths (List[int]): Sequence lengths for all dataset samples
        batch_size (int): Number of sequences per batch
        shuffle (bool): Shuffle within buckets
        seed (Optional[int]): Random seed (incremented each epoch)
    """

    def __init__(
        self,
        cluster_ids: List[int],
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.cluster_ids = np.asarray(cluster_ids)
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_seed = seed if seed is not None else 0
        self.epoch = 0

        # Build cluster -> list of dataset indices
        self.cluster_to_indices: Dict[int, List[int]] = {}
        for i, cid in enumerate(cluster_ids):
            self.cluster_to_indices.setdefault(cid, []).append(i)

        # Orphans (cluster_id=-1) are treated as individual "clusters"
        self._orphan_indices = self.cluster_to_indices.pop(-1, [])
        self._n_clusters = len(self.cluster_to_indices)
        self._n_orphans = len(self._orphan_indices)
        self._n_per_epoch = self._n_clusters + self._n_orphans

        log.info(
            f"ClusterBucketBatchSampler: {self._n_clusters} clusters + "
            f"{self._n_orphans} orphans = {self._n_per_epoch} samples/epoch "
            f"(from {len(cluster_ids)} total chains)"
        )

    def _sample_epoch_indices(self, rng: np.random.RandomState) -> List[int]:
        """Pick one random chain per cluster + all orphans."""
        selected = []
        for cid, indices in self.cluster_to_indices.items():
            selected.append(rng.choice(indices))
        selected.extend(self._orphan_indices)
        return selected

    def __iter__(self):
        rng = np.random.RandomState(self.base_seed + self.epoch)
        self.epoch += 1

        # 1. Sample one chain per cluster
        selected = self._sample_epoch_indices(rng)
        selected_lengths = self.lengths[selected]

        # 2. Auto-compute bins from selected lengths
        if len(selected_lengths) == 0:
            return
        percentiles = np.percentile(
            selected_lengths, np.linspace(5, 95, 19).tolist()
        ).astype(int).tolist()
        bins = sorted(set(percentiles))
        bin_ids = np.digitize(selected_lengths, bins, right=True)

        # 3. Group into buckets and build batches
        indices_per_bin: Dict[int, List[int]] = {}
        for i, bin_id in enumerate(bin_ids):
            indices_per_bin.setdefault(bin_id, []).append(selected[i])

        all_batches = []
        for bin_id in sorted(indices_per_bin.keys(), reverse=True):
            bucket = indices_per_bin[bin_id]
            if self.shuffle:
                rng.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                all_batches.append(bucket[i : i + self.batch_size])

        # Shuffle batch order so limit_train_batches samples uniformly
        # across all length ranges instead of always taking the longest
        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        # Approximate: exact count depends on bucket distribution
        return int(np.ceil(self._n_per_epoch / max(1, self.batch_size)))


class ClusterTrainValSampler:
    """
    Cluster-aware sampler that dynamically splits train/val each epoch.

    Every epoch, for each cluster with ≥2 members, randomly picks 2 different
    chains: one goes to train, the other to val.  Singleton clusters (1 member)
    and orphans (cluster_id = -1) go to train only — they have no partner for
    validation.

    Both train and val batches are length-bucketed (longest-first) to minimise
    padding waste and fail-fast on OOM.

    This guarantees:
    - Val is in-distribution with train (same cluster families)
    - No chain appears in both train and val in the same epoch
    - Different representatives are sampled every epoch (data augmentation)

    Args:
        cluster_ids: Cluster ID for each dataset sample (−1 = orphan)
        lengths: Sequence length for each dataset sample
        batch_size: Samples per batch
        seed: Base seed (incremented by epoch number)
    """

    def __init__(
        self,
        cluster_ids: List[int],
        lengths: List[int],
        batch_size: int,
        seed: int = 0,
        rotate_val: bool = True,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.base_seed = seed
        self.epoch = 0
        self.rotate_val = bool(rotate_val)

        # Build cluster → list of dataset indices
        cluster_to_indices: Dict[int, List[int]] = {}
        for i, cid in enumerate(cluster_ids):
            cluster_to_indices.setdefault(cid, []).append(i)

        # Separate orphans, singletons, multi-member clusters
        self._orphan_indices = cluster_to_indices.pop(-1, [])
        self._singleton_indices: List[int] = []
        self._multi_clusters: Dict[int, List[int]] = {}

        for cid, indices in cluster_to_indices.items():
            if len(indices) == 1:
                self._singleton_indices.append(indices[0])
            else:
                self._multi_clusters[cid] = indices

        self._n_multi = len(self._multi_clusters)
        self._n_singletons = len(self._singleton_indices)
        self._n_orphans = len(self._orphan_indices)

        # Per-epoch sizes
        self._n_train = self._n_multi + self._n_singletons + self._n_orphans
        self._n_val = self._n_multi  # one val sample per multi-member cluster

        log.info(
            f"ClusterTrainValSampler: {self._n_multi} multi-clusters "
            f"(→ train+val), {self._n_singletons} singletons + "
            f"{self._n_orphans} orphans (→ train only) | "
            f"~{self._n_train} train + ~{self._n_val} val samples/epoch "
            f"(from {len(cluster_ids)} total chains)"
        )

    def _sample_epoch(self, epoch: int):
        """Return (train_indices, val_indices) for this epoch."""
        rng = np.random.RandomState(self.base_seed + epoch)

        train_indices = []
        val_indices = []

        # Multi-member clusters: pick 2 different chains
        for cid, indices in self._multi_clusters.items():
            chosen = rng.choice(len(indices), size=2, replace=False)
            train_indices.append(indices[chosen[0]])
            val_indices.append(indices[chosen[1]])

        # Singletons → train only
        train_indices.extend(self._singleton_indices)
        # Orphans → train only
        train_indices.extend(self._orphan_indices)

        return train_indices, val_indices

    @staticmethod
    def _bucketed_batches(
        indices: List[int],
        lengths: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
        shuffle: bool = True,
    ):
        """Yield length-bucketed batches, longest first."""
        if not indices:
            return []
        sel_lengths = lengths[indices]
        percentiles = np.percentile(
            sel_lengths, np.linspace(5, 95, 19).tolist()
        ).astype(int).tolist()
        bins = sorted(set(percentiles))
        bin_ids = np.digitize(sel_lengths, bins, right=True)

        indices_per_bin: Dict[int, List[int]] = {}
        for i, bid in enumerate(bin_ids):
            indices_per_bin.setdefault(bid, []).append(indices[i])

        batches = []
        for bid in sorted(indices_per_bin.keys(), reverse=True):
            bucket = indices_per_bin[bid]
            if shuffle:
                rng.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                batches.append(bucket[i : i + batch_size])
        # Shuffle batch order so limit_train_batches samples uniformly
        # across all length ranges instead of always taking the longest.
        # Without this, ltb=0.1 silently restricted training to the longest
        # ~10% of sequences, and full-data runs ended every epoch on the
        # shortest batches, biasing the pre-validation weight state.
        if shuffle:
            rng.shuffle(batches)
        return batches

    def train_batches(self):
        """Return list of train batch index-lists for the current epoch."""
        epoch = self.epoch
        train_idx, _ = self._sample_epoch(epoch)
        rng = np.random.RandomState(self.base_seed + epoch + 100000)
        return self._bucketed_batches(
            train_idx, self.lengths, self.batch_size, rng, shuffle=True
        )

    def val_batches(self):
        """Return list of val batch index-lists for the current epoch.

        When rotate_val=False (recommended for stable trajectory measurement),
        the val composition is fixed to the epoch=0 sample regardless of how
        many epochs have elapsed.
        """
        epoch = self.epoch if self.rotate_val else 0
        _, val_idx = self._sample_epoch(epoch)
        rng = np.random.RandomState(self.base_seed + epoch + 200000)
        return self._bucketed_batches(
            val_idx, self.lengths, self.batch_size, rng, shuffle=False
        )

    def advance_epoch(self):
        """Increment the epoch counter. Call once per epoch."""
        self.epoch += 1

    @property
    def n_train_batches(self) -> int:
        return int(np.ceil(self._n_train / max(1, self.batch_size)))

    @property
    def n_val_batches(self) -> int:
        return int(np.ceil(self._n_val / max(1, self.batch_size)))


class ListBatchSampler(Sampler[List[int]]):
    """Thin wrapper that turns a pre-computed list of batches into a Sampler.

    Used to feed the output of ClusterTrainValSampler.train_batches() /
    .val_batches() into DataLoader(batch_sampler=...).
    """

    def __init__(self, batches: List[List[int]]):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class BucketBatchSampler(Sampler[List[int]]):
    """
    Length-bucketed batch sampler to reduce padding waste.

    Args:
        lengths (List[int]): Sequence lengths for all dataset items
        batch_size (int): Number of sequences per batch
        shuffle (bool): Shuffle indices within each bin
        bins (Optional[List[int]]): Custom bin boundaries (default: auto from percentiles)
        seed (seed: Optional[int]): Random seed for shuffling

    Usage:
        sampler = BucketBatchSampler(lengths, batch_size=32)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_padded)
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        bins: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )

        # Auto-compute bins from percentiles if not provided
        if bins is None:
            if len(self.lengths) == 0:
                bins = [100, 200, 300, 400]  # fallback for empty dataset
            else:
                percentiles = (
                    np.percentile(self.lengths, [20, 40, 60, 80]).astype(int).tolist()
                )
                bins = sorted(list(set(percentiles)))
        self.bins = bins
        self.bin_ids = np.digitize(self.lengths, bins, right=True)

        self.indices_per_bin = {}
        for bin_id in range(len(bins) + 1):
            self.indices_per_bin[bin_id] = np.where(self.bin_ids == bin_id)[0].tolist()

    def __iter__(self):
        # Reshuffle within each bin on every new iterator (= new epoch)
        for bin_id in self.indices_per_bin:
            indices = list(self.indices_per_bin[bin_id])  # copy to avoid mutation issues
            if self.shuffle:
                self.rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self) -> int:
        total = 0
        for bin_id in self.indices_per_bin:
            num_samples = len(self.indices_per_bin[bin_id])
            total += int(np.ceil(num_samples / max(1, self.batch_size)))
        return total
