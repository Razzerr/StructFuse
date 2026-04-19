from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from src.data.components.dataset import (
    ContactDataset,
    PriorBuilder,
    collate_padded,
    BucketBatchSampler,
    ClusterBucketBatchSampler,
    ClusterTrainValSampler,
    ListBatchSampler,
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
        pin_memory: bool = False,
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
        # subset info for per-subset evaluation
        splits_json_path: Optional[str] = None,
        # subsets to exclude from test set (e.g. ["cluster_promoted"])
        test_exclude_subsets: Optional[List[str]] = None,
        # Cluster-aware sampling: 1 chain per cluster per epoch
        cluster_sampling: bool = True,
        index_dir: Optional[str] = None,
        # Retrieval / prior-building params (used by PriorBuilder in workers)
        topk: int = 4,
        use_blosum: bool = True,
        only_positive_transfer: bool = False,
        min_template_similarity: float = 0.0,
        random_retrieval: bool = False,
        max_tpl_cache: int = 1000,
        # Stage 2 — cheap features derived from template Cα coords inside
        # PriorBuilder. Off by default so existing configs keep the legacy
        # (prior, count) output shape.
        compute_dist_bins: bool = False,
        compute_agreement: bool = False,
        compute_dist_stats: bool = False,
        # Stage 4 — ground-truth Cα-Cα distogram bins for auxiliary loss.
        compute_gt_dist_bins: bool = False,
        # Precomputed ESM2 embeddings (see scripts/precompute_esm2_embeddings.py)
        esm_embeddings_dir: Optional[str] = None,
        # Holdout protein IDs (val + test): each file is one chain/PDB ID per
        # line. PriorBuilder passes the union to FaissIndex (normalised to
        # protein-level). Training collate uses filter_holdout=True so no
        # train query ever retrieves a val/test structure; val/test collate
        # uses filter_holdout=False so inference sees the full index.
        holdout_id_files: Optional[List[str]] = None,
        # Plaintext file of chain stems whose NPZ (processed or precomputed ESM)
        # is known-corrupt. Filtered out of every dataset at setup time.
        skip_ids_file: Optional[str] = None,
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
        
        # Path to mmcif_final_splits.json for subset membership
        self.splits_json_path = Path(splits_json_path) if splits_json_path else None
        self.test_exclude_subsets = test_exclude_subsets
        self.cluster_sampling = bool(cluster_sampling)
        self.index_dir = Path(index_dir) if index_dir else None

        # Retrieval / prior-building params
        self.topk = int(topk)
        self.use_blosum = bool(use_blosum)
        self.only_positive_transfer = bool(only_positive_transfer)
        self.min_template_similarity = float(min_template_similarity)
        self.random_retrieval = bool(random_retrieval)
        self.max_tpl_cache = int(max_tpl_cache)
        self.compute_dist_bins = bool(compute_dist_bins)
        self.compute_agreement = bool(compute_agreement)
        self.compute_dist_stats = bool(compute_dist_stats)
        self.compute_gt_dist_bins = bool(compute_gt_dist_bins)
        self.esm_embeddings_dir = Path(esm_embeddings_dir) if esm_embeddings_dir else None
        self.holdout_id_files: List[str] = [str(p) for p in (holdout_id_files or [])]
        self.skip_ids_file: Optional[str] = str(skip_ids_file) if skip_ids_file else None

        # Initialize RNG for deterministic cropping
        self.crop_rng = np.random.RandomState(self.split_seed)
        
        # Will be set in setup()
        self.dset_trainval = None   # shared dataset for cluster train/val
        self.dset_train = None      # fallback: separate train dataset
        self.dset_val = None        # fallback: separate val dataset
        self.dset_test = None
        self._cluster_sampler: Optional[ClusterTrainValSampler] = None
        self._prior_builder: Optional[PriorBuilder] = None

    def _write_list(self, path: Path, ids: List[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(ids) + "\n")

    def _read_list(self, path: Path) -> List[str]:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _make_train_val_split(self, all_train_path: Path) -> Tuple[Path, Path]:
        """Fallback static split when cluster_sampling is disabled."""
        all_ids = self._read_list(all_train_path)
        if len(all_ids) == 0:
            raise RuntimeError(f"No train IDs found in {all_train_path}")

        rng = np.random.RandomState(self.split_seed)
        indices = np.arange(len(all_ids))
        rng.shuffle(indices)

        num_total = len(all_ids)
        num_val = int(round(self.val_ratio * num_total))
        num_val = max(1, min(num_val, num_total - 1))
        
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

        When cluster_sampling is enabled, a single dataset is created from
        all_train_ids.txt.  The ClusterTrainValSampler dynamically assigns
        chains to train vs val each epoch (1 per cluster to each; singletons
        and orphans go to train only).

        When cluster_sampling is disabled, falls back to static train/val
        split files.
        """
        if stage == "fit" and self.dset_trainval is not None and self.dset_val is not None:
            log.info("Train/val datasets already loaded, skipping setup")
            return
        if stage == "fit" and self.dset_train is not None and self.dset_val is not None:
            log.info("Train/val datasets already loaded, skipping setup")
            return
        if stage == "test" and self.dset_test is not None:
            log.info("Test dataset already loaded, skipping setup")
            return
        
        # Resolve test split path
        test_split_path = Path(self.test_ids_path) if self.test_ids_path else self.split_dir / "all_test_ids.txt"

        if stage == "fit" or stage is None:
            if self.cluster_sampling and self.index_dir is not None:
                # ── Cluster-aware dynamic train/val ──
                all_train_path = self.split_dir / "all_train_ids.txt"
                assert all_train_path.exists(), f"all_train_ids.txt not found: {all_train_path}"

                log.info("Creating unified train+val dataset with cluster-aware dynamic splitting")
                self.dset_trainval = ContactDataset(
                    all_train_path,
                    root=self.data_root,
                    min_len=self.min_len,
                    splits_json_path=self.splits_json_path,
                    skip_ids_file=self.skip_ids_file,
                    index_dir=self.index_dir,
                )
                log.info(f"  TrainVal: {len(self.dset_trainval)} chains from {all_train_path}")

                cluster_ids = self.dset_trainval.cluster_ids
                if cluster_ids is None:
                    raise RuntimeError(
                        "cluster_sampling=True but no cluster info loaded. "
                        "Check index_dir / ids.json."
                    )
                self._cluster_sampler = ClusterTrainValSampler(
                    cluster_ids=cluster_ids,
                    lengths=self.dset_trainval.cached_lengths,
                    batch_size=self.batch_size,
                    seed=self.split_seed,
                )

                # ── Static val from held-out test split ──
                if self.val_ids_path:
                    val_split_path = Path(self.val_ids_path)
                    assert val_split_path.exists(), f"val_ids file not found: {val_split_path}"
                    log.info(f"Creating static val dataset from {val_split_path}")
                    self.dset_val = ContactDataset(
                        val_split_path,
                        root=self.data_root,
                        min_len=self.min_len,
                        splits_json_path=self.splits_json_path,
                    skip_ids_file=self.skip_ids_file,
                        exclude_subsets=self.test_exclude_subsets,
                    )
                    log.info(f"  Val (static): {len(self.dset_val)} samples")
            else:
                # ── Fallback: static train/val split ──
                if self.train_ids_path and self.val_ids_path:
                    train_split_path = Path(self.train_ids_path)
                    val_split_path = Path(self.val_ids_path)
                else:
                    all_train_path = self.split_dir / "all_train_ids.txt"
                    assert all_train_path.exists(), f"all_train_ids.txt not found: {all_train_path}"
                    split_tag = f"val{int(self.val_ratio*100)}_seed{self.split_seed}_min{self.min_len}"
                    train_split_path = self.split_dir / f"train_split_{split_tag}.txt"
                    val_split_path = self.split_dir / f"val_split_{split_tag}.txt"
                    if not (train_split_path.exists() and val_split_path.exists()):
                        train_split_path, val_split_path = self._make_train_val_split(all_train_path)

                log.info("Creating separate train/val datasets (static split)")
                self.dset_train = ContactDataset(
                    train_split_path,
                    root=self.data_root,
                    min_len=self.min_len,
                    splits_json_path=self.splits_json_path,
                    skip_ids_file=self.skip_ids_file,
                )
                self.dset_val = ContactDataset(
                    val_split_path,
                    root=self.data_root,
                    min_len=self.min_len,
                    splits_json_path=self.splits_json_path,
                    skip_ids_file=self.skip_ids_file,
                )
                log.info(f"  Train: {len(self.dset_train)} | Val: {len(self.dset_val)}")

            # ── PriorBuilder (shared across train / val / test DataLoaders) ──
            if self._prior_builder is None and self.index_dir is not None and self.topk > 0:
                log.info(
                    f"Creating PriorBuilder (topk={self.topk}, blosum={self.use_blosum}, "
                    f"index_dir={self.index_dir})"
                )
                self._prior_builder = PriorBuilder(
                    index_dir=str(self.index_dir),
                    topk=self.topk,
                    use_blosum=self.use_blosum,
                    only_positive_transfer=self.only_positive_transfer,
                    min_seq_sep=self.min_seq_sep,
                    min_template_similarity=self.min_template_similarity,
                    random_retrieval=self.random_retrieval,
                    max_tpl_cache=self.max_tpl_cache,
                    compute_dist_bins=self.compute_dist_bins,
                    compute_agreement=self.compute_agreement,
                    compute_dist_stats=self.compute_dist_stats,
                    holdout_id_files=self.holdout_id_files,
                )

        if stage == "test" or stage is None:
            assert test_split_path.exists(), f"Test split file not found: {test_split_path}"
            log.info("Creating test dataset with subset info")
            self.dset_test = ContactDataset(
                test_split_path,
                root=self.data_root,
                min_len=self.min_len,
                splits_json_path=self.splits_json_path,
                skip_ids_file=self.skip_ids_file,
                exclude_subsets=self.test_exclude_subsets,
            )
            log.info(f"  Test:  {len(self.dset_test)} samples from {test_split_path}")

            # Ensure PriorBuilder exists for test as well
            if self._prior_builder is None and self.index_dir is not None and self.topk > 0:
                log.info(
                    f"Creating PriorBuilder for test (topk={self.topk}, "
                    f"index_dir={self.index_dir})"
                )
                self._prior_builder = PriorBuilder(
                    index_dir=str(self.index_dir),
                    topk=self.topk,
                    use_blosum=self.use_blosum,
                    only_positive_transfer=self.only_positive_transfer,
                    min_seq_sep=self.min_seq_sep,
                    min_template_similarity=self.min_template_similarity,
                    random_retrieval=self.random_retrieval,
                    max_tpl_cache=self.max_tpl_cache,
                    compute_dist_bins=self.compute_dist_bins,
                    compute_agreement=self.compute_agreement,
                    compute_dist_stats=self.compute_dist_stats,
                    holdout_id_files=self.holdout_id_files,
                )

    def _collate_train(self, batch):
        """Collate with random cropping for training.

        ``filter_holdout=True`` ensures no train query retrieves a val/test
        structure — leakage prevention is enforced at query-time rather than
        by excluding val/test from the index.
        """
        rng_seed = self.crop_rng.randint(0, 2**31)
        return collate_padded(
            batch,
            crop_size=self.crop_size,
            crop_mode="random",
            min_seq_sep=self.min_seq_sep,
            include_diagonal=False,
            seed=rng_seed,
            prior_builder=self._prior_builder,
            esm_embeddings_dir=self.esm_embeddings_dir,
            compute_gt_dist_bins=self.compute_gt_dist_bins,
            filter_holdout=True,
        )

    def _collate_eval(self, batch):
        """Collate with center cropping for deterministic val/test.

        ``filter_holdout=False``: val/test queries may retrieve any chain in
        the full index (including other val/test chains). This is safe —
        we're not updating gradients from val/test retrieval — and gives the
        model real template context at eval time.
        """
        return collate_padded(
            batch,
            crop_size=self.crop_size,
            crop_mode="center",
            min_seq_sep=self.min_seq_sep,
            include_diagonal=False,
            seed=42,  # fixed seed for reproducibility
            prior_builder=self._prior_builder,
            esm_embeddings_dir=self.esm_embeddings_dir,
            compute_gt_dist_bins=self.compute_gt_dist_bins,
            filter_holdout=False,
        )
        
    def _dl_kwargs(self, collate_fn=None):
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn or self._collate_train,
        )

    def train_dataloader(self):
        kwargs = self._dl_kwargs(collate_fn=self._collate_train)
        # ── Cluster-aware dynamic split ──
        if self._cluster_sampler is not None:
            batches = self._cluster_sampler.train_batches()
            return DataLoader(
                self.dset_trainval,
                batch_sampler=ListBatchSampler(batches),
                **kwargs,
            )

        # ── Fallback: static split ──
        lengths = self.dset_train.cached_lengths
        if self.bucketed:
            sampler = BucketBatchSampler(
                lengths, batch_size=self.batch_size, shuffle=True, seed=self.split_seed
            )
            return DataLoader(
                self.dset_train,
                batch_sampler=sampler,
                **kwargs,
            )
        return DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs,
        )

    def val_dataloader(self):
        kwargs = self._dl_kwargs(collate_fn=self._collate_eval)

        # ── Static val (from held-out test split) takes priority ──
        if self.dset_val is not None:
            if self._cluster_sampler is not None:
                self._cluster_sampler.advance_epoch()
            if self.bucketed:
                lengths = self.dset_val.cached_lengths
                sampler = BucketBatchSampler(
                    lengths, batch_size=self.batch_size, shuffle=False, seed=self.split_seed
                )
                return DataLoader(
                    self.dset_val,
                    batch_sampler=sampler,
                    **kwargs,
                )
            return DataLoader(
                self.dset_val,
                batch_size=self.batch_size,
                shuffle=False,
                **kwargs,
            )

        # ── Cluster-aware dynamic val (legacy, no static val_ids) ──
        if self._cluster_sampler is not None:
            batches = self._cluster_sampler.val_batches()
            self._cluster_sampler.advance_epoch()
            return DataLoader(
                self.dset_trainval,
                batch_sampler=ListBatchSampler(batches),
                **kwargs,
            )

        raise RuntimeError("No val dataset available. Set val_ids or enable cluster_sampling.")

    def test_dataloader(self):
        return DataLoader(
            self.dset_test,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dl_kwargs(collate_fn=self._collate_eval),
        )
