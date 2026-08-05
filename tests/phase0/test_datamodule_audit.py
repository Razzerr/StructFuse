"""Regression tests for validate-only setup and paper-run audit identities."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.data.contact_lit_datamodule as datamodule_module  # noqa: E402
from src.data.contact_lit_datamodule import ContactDataModule  # noqa: E402
from src.utils.audit import dump_audit_manifest  # noqa: E402


def test_eval_only_experiments_disable_optimized_metric():
    config_root = Path(__file__).resolve().parents[2] / "configs"
    train_cfg = OmegaConf.load(config_root / "train.yaml")
    assert train_cfg.optimized_metric == "val/loss"
    for name in ("esm2_only.yaml", "esm2_650m_only.yaml"):
        experiment = OmegaConf.load(config_root / "experiment" / "baseline" / name)
        assert experiment.train is False
        assert experiment.test is True
        assert experiment.optimized_metric is None


class _FakeDataset:
    created_paths = []

    def __init__(self, split_path, **_kwargs):
        self.split_path = Path(split_path)
        self.cached_lengths = [32]
        self.cluster_ids = None
        self.created_paths.append(self.split_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError(f"Dataset iteration not expected in this unit test: {index}")


def test_validate_only_setup_loads_static_val_without_train():
    original = datamodule_module.ContactDataset
    _FakeDataset.created_paths = []
    datamodule_module.ContactDataset = _FakeDataset
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_ids = root / "val_ids.txt"
            val_ids.write_text("1abc_A\n")
            dm = ContactDataModule(
                data_root=str(root),
                val_ids=str(val_ids),
                index_dir=None,
                topk=0,
                cluster_sampling=True,
                num_workers=0,
            )
            # Lightning passes TrainerFn.VALIDATING, not necessarily a raw str.
            dm.setup(stage=TrainerFn.VALIDATING)
            assert dm.dset_val is not None
            assert dm.dset_train is None
            assert dm.dset_trainval is None
            assert _FakeDataset.created_paths == [val_ids]

            first = dm.dset_val
            dm.setup(stage=TrainerFn.VALIDATING)
            assert dm.dset_val is first
            assert _FakeDataset.created_paths == [val_ids]
    finally:
        datamodule_module.ContactDataset = original


def test_audit_manifest_fingerprints_index_splits_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index_dir = root / "index"
        index_dir.mkdir()
        (index_dir / "ids.json").write_text(json.dumps([
            {"id": "1abc_A", "npz": "1abc_A.npz"},
            {"id": "2def_B", "npz": "2def_B.npz"},
        ], indent=2))
        (index_dir / "faiss.index").write_bytes(b"faiss-artifact")
        (index_dir / "embeddings.npy").write_bytes(b"embedding-artifact")

        split_dir = root / "splits"
        split_dir.mkdir()
        train = split_dir / "all_train_ids.txt"
        val = split_dir / "val.txt"
        test = split_dir / "test.txt"
        holdout = split_dir / "holdout.txt"
        skip = root / "skip.txt"
        for path, text in (
            (train, "1abc_A\n"),
            (val, "2def_B\n"),
            (test, "3ghi_C\n"),
            (holdout, "2def_B\n3ghi_C\n"),
            (skip, "bad_A\n"),
        ):
            path.write_text(text)
        checkpoint = root / "model.ckpt"
        checkpoint.write_bytes(b"checkpoint")

        cfg = OmegaConf.create({
            "task_name": "audit_test",
            "seed": 42,
            "tags": [],
            "data": {
                "data_root": str(root),
                "split_dir": str(split_dir),
                "train_ids": None,
                "val_ids": str(val),
                "test_ids": str(test),
                "splits_json_path": None,
                "holdout_id_files": [str(holdout)],
                "skip_ids_files": [str(skip)],
                "index_dir": str(index_dir),
                "topk": 4,
                "random_retrieval": False,
                "min_template_similarity": 0.0,
                "esm_embeddings_dir": None,
            },
            "model": {},
            "trainer": {},
        })
        trainer = SimpleNamespace(
            loggers=[SimpleNamespace(experiment=SimpleNamespace(id="audit-run"))],
            logger=None,
        )
        path = dump_audit_manifest(
            cfg,
            trainer,
            ckpt_path=str(checkpoint),
            output_root=str(root / "audit"),
        )
        manifest = json.loads(path.read_text())
        index = manifest["retrieval"]["index_identity"]
        assert index["ids_json"]["entry_count"] == 2
        assert index["ids_json"]["fingerprint_type"] == "sha256"
        assert index["faiss_index"]["fingerprint"]
        assert index["embeddings"]["fingerprint"]
        assert manifest["checkpoint_identity"]["fingerprint"]
        splits = manifest["data"]["splits"]
        assert splits["train"]["path"] == str(train.resolve())
        assert splits["validation"]["fingerprint_type"] == "sha256"
        assert splits["test"]["fingerprint_type"] == "sha256"
        assert splits["holdout_filters"][0]["fingerprint"]
        assert splits["skip_ids"][0]["fingerprint"]


def _run_all():
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
