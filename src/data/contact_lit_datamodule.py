from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from src.data.components.dataset import (
    ContactDataset,
    collate_padded,
    BucketBatchSampler,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class ContactDataModule(LightningDataModule):
    """
    LightningDataModule for protein contact prediction.
    Args:
        data_root (str): Root directory with .npz files
        split_dir (str): Directory to save/load split files (default: "data/splits")
        val_ratio (float): Fraction of training data to use for validation (default: 0.1)
        split_seed (int): Random seed for train/val splitting (default: 0)
        batch_size (int): Batch size for dataloaders (default: 2)
        num_workers (int): Number of dataloader workers (default: 4)
        pin_memory (bool): Pin memory in dataloaders (default: True)
        persistent_workers (bool): Keep workers alive between epochs (default: True)
        prefetch_factor (int): Number of batches to prefetch per worker (default: 4)
        bucketed (bool): Use bucketed batch sampling (groups similar lengths) (default: True)
        crop_size (int, optional): Maximum sequence length (crops longer sequences)
        crop_mode (str): Cropping strategy - "random" or "center" (default: "random")
        min_seq_sep (int): Minimum sequence separation for contact prediction (default: 6)
        min_len (int): Minimum sequence length to include (default: 20)
        train_ids (str, optional): Path to manual train split file (overrides automatic splitting)
        val_ids (str, optional): Path to manual validation split file
        test_ids (str, optional): Path to manual test split file
    """

    def __init__(
        self,
        # roots & split settings
        data_root: str,
        split_dir: str = "data/splits",
        val_ratio: float = 0.1,
        split_seed: int = 0,
        # dataloader settings
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True, 
        prefetch_factor: int = 4,
        bucketed: bool = True,
        crop_size: Optional[int] = 512,
        crop_mode: str = "random",
        min_seq_sep: int = 6,
        min_len: int = 20,
        # optional overrides (if you want to pin splits manually)
        train_ids: Optional[str] = None,
        val_ids: Optional[str] = None,
        test_ids: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split_dir = Path(split_dir)
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.bucketed = bool(bucketed)
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.min_seq_sep = int(min_seq_sep)
        self.min_len = int(min_len)

        # Manual split files (optional)
        self.train_ids_path = train_ids
        self.val_ids_path = val_ids
        self.test_ids_path = test_ids

        # Initialize RNG for deterministic cropping
        self.crop_rng = np.random.RandomState(self.split_seed)
        
        # Will be set in setup()
        self.dset_train = None
        self.dset_val = None
        self.dset_test = None

    def _write_list(self, path: Path, ids: List[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(ids) + "\n")

    def _read_list(self, path: Path) -> List[str]:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _ensure_canonical_lists(self) -> Tuple[Path, Path]:
        """
        Ensure all_train_ids.txt and all_test_ids.txt exist in split_dir.
        """
        all_train_path = self.split_dir / "all_train_ids.txt"
        all_test_path = self.split_dir / "all_test_ids.txt"

        if not all_train_path.exists():
            train_ids = self._read_list(self.split_dir / "all_train_ids.txt")
            self._write_list(all_train_path, train_ids)
        if not all_test_path.exists():
            test_ids = self._read_list(self.split_dir / "all_test_ids.txt")
            self._write_list(all_test_path, test_ids)

        return all_train_path, all_test_path

    def _make_train_val_split(self, all_train_path: Path) -> Tuple[Path, Path]:
        all_ids = self._read_list(all_train_path)
        if len(all_ids) == 0:
            raise RuntimeError(f"No train IDs found in {all_train_path}")

        rng = np.random.RandomState(self.split_seed)
        indices = np.arange(len(all_ids))
        rng.shuffle(indices)

        num_total = len(all_ids)
        num_val = int(round(self.val_ratio * num_total))
        num_val = max(1, min(num_val, num_total - 1))  # At least 1 val, at least 1 train
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_split_ids = [all_ids[i] for i in train_indices]
        val_split_ids = [all_ids[i] for i in val_indices]

        split_tag = f"val{int(self.val_ratio*100)}_seed{self.split_seed}_min{self.min_len}"
        train_split_path = self.split_dir / f"train_split_{split_tag}.txt"
        val_split_path = self.split_dir / f"val_split_{split_tag}.txt"

        self._write_list(train_split_path, train_split_ids)
        self._write_list(val_split_path, val_split_ids)
        
        return train_split_path, val_split_path

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.

        Args:
            stage (str, optional): One of "fit", "test", or None
        """
        if stage == "fit" and self.dset_train is not None and self.dset_val is not None:
            log.info("Train/val datasets already loaded, skipping setup")
            return
        
        if stage == "test" and self.dset_test is not None:
            log.info("Test dataset already loaded, skipping setup")
            return
        
        if self.train_ids_path and self.val_ids_path and self.test_ids_path:
            train_split_path = Path(self.train_ids_path)
            val_split_path = Path(self.val_ids_path)
            test_split_path = Path(self.test_ids_path)
        else:
            # Build/ensure canonical lists
            all_train_path, all_test_path = self._ensure_canonical_lists()
            test_split_path = all_test_path
            
            split_tag = f"val{int(self.val_ratio*100)}_seed{self.split_seed}_min{self.min_len}"
            train_split_path = self.split_dir / f"train_split_{split_tag}.txt"
            val_split_path = self.split_dir / f"val_split_{split_tag}.txt"
            
            if not (train_split_path.exists() and val_split_path.exists()):
                train_split_path, val_split_path = self._make_train_val_split(all_train_path)

        assert train_split_path.exists(), f"Train split file not found: {train_split_path}"
        assert val_split_path.exists(), f"Val split file not found: {val_split_path}"
        assert test_split_path.exists(), f"Test split file not found: {test_split_path}"

        train_sample_ids = self._read_list(train_split_path)
        val_sample_ids = self._read_list(val_split_path)
        test_sample_ids = self._read_list(test_split_path)

        assert (
            set(train_sample_ids).intersection(set(val_sample_ids), set(test_sample_ids)) == set()
        ), "Train/val/test splits overlap!"
        
        if stage == "fit" or stage is None:
            log.info("Creating train/val datasets")
            self.dset_train = ContactDataset(train_split_path, root=self.data_root, min_len=self.min_len)
            log.info(f"  Train: {len(self.dset_train)} samples from {train_split_path}")
            self.dset_val = ContactDataset(val_split_path, root=self.data_root, min_len=self.min_len)
            log.info(f"  Val:   {len(self.dset_val)} samples from {val_split_path}")

        if stage == "test" or stage is None:
            log.info("Creating test dataset")
            self.dset_test = ContactDataset(test_split_path, root=self.data_root, min_len=self.min_len)
            log.info(f"  Test:  {len(self.dset_test)} samples from {test_split_path}")

    def _collate(self, batch):
        rng_seed = self.crop_rng.randint(0, 1024)
        
        return collate_padded(
            batch,
            crop_size=self.crop_size,
            crop_mode=self.crop_mode,
            min_seq_sep=self.min_seq_sep,
            include_diagonal=False,
            seed=rng_seed,
        )
        
    def _dl_kwargs(self):
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self._collate,
        )

    def train_dataloader(self):
        if self.bucketed:
            lengths = [sample["L"] for sample in self.dset_train]
            sampler = BucketBatchSampler(
                lengths, batch_size=self.batch_size, shuffle=True, seed=self.split_seed
            )
            return DataLoader(
                self.dset_train,
                batch_sampler=sampler,
                **self._dl_kwargs()
            )
        return DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            shuffle=True,
            **self._dl_kwargs()
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_val,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dl_kwargs()
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_test,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dl_kwargs()
        )
