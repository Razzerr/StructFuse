from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class ContactDataset(Dataset):
    """
    PyTorch dataset for protein contact prediction.

    Loads per-chain NPZ files produced by scripts/build_contacts.py.

    Args:
        id_list_file (Path): Path to file containing PDB IDs (one per line)
        root (Path): Directory containing processed NPZ files
        min_len (int): Minimum sequence length to include

    Returns:
        Dict with keys: pid, seq, contact, mask, L
    """

    def __init__(
        self, id_list_file: Path, root: Path = Path("data/processed"), min_len: int = 1
    ):
        self.root = root
        self.ids = []

        # Load PDB IDs from list file
        raw_ids = [line.strip() for line in open(id_list_file) if line.strip()]
        all_npz_files = list(self.root.glob("*.npz"))

        # Match NPZ files by PDB ID (format: XXXX_Chain.npz)
        matching_files = [
            npz_file
            for npz_file in all_npz_files
            if npz_file.stem.split("_")[0] in raw_ids
        ]

        log.info(f"Scanning {len(matching_files)} matching NPZ files...")
        for npz_path in tqdm(matching_files, desc="Loading dataset"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                L = int(data["L"])
                if L < min_len:
                    continue
                self.ids.append(npz_path.stem)
            except Exception as e:
                log.warning(f"Failed to load {npz_path}: {e}")
                raise e

        log.info(f"Loaded {len(self.ids)} chains (min_len={min_len})")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.ids[idx]
        data = np.load(self.root / f"{pid}.npz", allow_pickle=True)

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

        return {
            "pid": pid,
            "seq": seq,
            "contact": contact,
            "mask": mask,
            "L": L,
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

    Returns:
        Dict with keys:
            - pid: list[str] - Protein IDs
            - seq: list[str] - Sequences (for ESM2 tokenization)
            - crop_bounds: (B, 2) int64 - Crop start/end indices
            - contact: (B, Lmax, Lmax) float32 - Binary contact maps
            - pair_mask: (B, Lmax, Lmax) float32 - Valid residue pairs
            - long_mask: (B, Lmax, Lmax) float32 - Long-range pairs (|i-j| >= min_seq_sep)
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
                "contact": item["contact"][crop_start:crop_end, crop_start:crop_end],
                "mask": item["mask"][crop_start:crop_end],
                "L": crop_end - crop_start,
                "crop_bounds": (crop_start, crop_end),
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
    crop_bounds = torch.zeros(B, 2, dtype=torch.long)

    for b, item in enumerate(cropped):
        pids.append(item["pid"])
        seqs.append(item["seq"])

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
        "crop_bounds": crop_bounds,  # (B, 2)
        "contact": contact,  # (B, Lmax, Lmax)
        "pair_mask": pair_mask,  # (B, Lmax, Lmax)
        "long_mask": long_mask,  # (B, Lmax, Lmax)
    }
    return batch_out


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
        for bin_id in self.indices_per_bin:
            indices = self.indices_per_bin[bin_id]
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
