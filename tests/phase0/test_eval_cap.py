"""Evaluation cap: at most N chains per sequence cluster, val/test only.

Chains inside one 30%-identity cluster are near-duplicates, so an uncapped mean
over chains is family-weighted. These tests pin the properties the protocol
depends on: the retained subset is deterministic and model-independent, CASP16
is never truncated, training is never capped, and an unclustered chain is a hard
error rather than a silent pass.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.components.dataset import (  # noqa: E402
    ContactDataset,
    load_chain_cluster_tsv,
)

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append((name, exc))
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")


class Fixture:
    """A synthetic split: 3 clusters of 10 chains + one CASP16 entry."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.root = tmp / "processed"
        self.root.mkdir()
        self.entries = []
        rows = ["id\tcluster_id\tsource\tentity_id"]
        lengths = {}

        for cluster in range(3):
            entry = f"{cluster + 1}abc"
            self.entries.append(entry)
            for chain in range(10):
                stem = f"{entry}_{chr(ord('A') + chain)}"
                self._write_npz(stem)
                lengths[stem] = 40
                rows.append(f"{stem}\t{100 + cluster}\tentity\t1")

        # CASP16 entry, its own cluster, more chains than the cap
        self.casp_entry = "9zzz"
        self.entries.append(self.casp_entry)
        for chain in range(6):
            stem = f"{self.casp_entry}_{chr(ord('A') + chain)}"
            self._write_npz(stem)
            lengths[stem] = 40
            rows.append(f"{stem}\t200\tentity\t1")

        (tmp / "npz_lengths.json").write_text(json.dumps(lengths))
        (self.root / "npz_lengths.json").write_text(json.dumps(lengths))

        self.clusters_tsv = tmp / "chain_clusters.tsv"
        self.clusters_tsv.write_text("\n".join(rows) + "\n")

        self.ids_file = tmp / "ids.txt"
        self.ids_file.write_text("\n".join(self.entries) + "\n")

        # load_subset_mapping only labels entries whose split is "test";
        # val_holdout entries carry the same label because split_test_val.py
        # carves validation out of the evaluated test pool.
        splits = {
            e: {"split": "test", "casp16_test_set": e == self.casp_entry}
            for e in self.entries
        }
        self.splits_json = tmp / "splits.json"
        self.splits_json.write_text(json.dumps(splits))

    def _write_npz(self, stem):
        L = 40
        np.savez(
            self.root / f"{stem}.npz",
            seq=np.array("A" * L, dtype=object),
            contact=np.zeros((L, L), dtype=np.uint8),
            mask=np.ones(L, dtype=np.uint8),
            L=L,
        )

    def dataset(self, **kw):
        return ContactDataset(
            self.ids_file,
            root=self.root,
            splits_json_path=self.splits_json,
            **kw,
        )


def _with_fixture(fn):
    def run():
        with tempfile.TemporaryDirectory() as td:
            fn(Fixture(Path(td)))
    return run


def test_uncapped_keeps_every_chain(fx):
    assert len(fx.dataset().ids) == 36, len(fx.dataset().ids)


def test_cap_limits_each_cluster_but_exempts_casp16(fx):
    ds = fx.dataset(max_chains_per_cluster=4, chain_clusters_file=fx.clusters_tsv)
    stems = ds.ids
    # 3 ordinary clusters capped at 4, CASP16's 6 chains kept in full
    assert len(stems) == 3 * 4 + 6, len(stems)
    casp = [s for s in stems if s.startswith("9zzz")]
    assert len(casp) == 6, casp
    for cluster in range(3):
        got = [s for s in stems if s.startswith(f"{cluster + 1}abc")]
        assert len(got) == 4, (cluster, got)


def test_retained_subset_is_deterministic_and_sorted(fx):
    a = fx.dataset(max_chains_per_cluster=3, chain_clusters_file=fx.clusters_tsv).ids
    b = fx.dataset(max_chains_per_cluster=3, chain_clusters_file=fx.clusters_tsv).ids
    assert a == b, "cap is not reproducible across constructions"
    # first-N-by-sorted-stem, so a model without retrieval sees the same chains
    assert [s for s in a if s.startswith("1abc")] == ["1abc_A", "1abc_B", "1abc_C"]


def test_cap_is_monotone_in_c(fx):
    sizes = [
        len(fx.dataset(max_chains_per_cluster=c,
                       chain_clusters_file=fx.clusters_tsv).ids)
        for c in (1, 2, 4, 8, 16)
    ]
    assert sizes == sorted(sizes), sizes
    assert sizes[-1] == 36, sizes  # cap above the largest cluster is a no-op


def test_exempt_subsets_can_be_overridden(fx):
    ds = fx.dataset(max_chains_per_cluster=4,
                    chain_clusters_file=fx.clusters_tsv,
                    cap_exempt_subsets=[])
    assert len(ds.ids) == 4 * 4, len(ds.ids)  # CASP16 now capped too


def test_unclustered_chain_is_an_error_not_a_silent_pass(fx):
    rows = fx.clusters_tsv.read_text().splitlines()
    rows = [r.replace("\t100\t", "\t-1\t") if r.startswith("1abc_A\t") else r
            for r in rows]
    fx.clusters_tsv.write_text("\n".join(rows) + "\n")
    try:
        fx.dataset(max_chains_per_cluster=4, chain_clusters_file=fx.clusters_tsv)
    except ValueError as exc:
        assert "no cluster" in str(exc)
        return
    raise AssertionError("expected ValueError for an unclustered chain")


def test_cap_requires_the_cluster_file(fx):
    try:
        fx.dataset(max_chains_per_cluster=4)
    except ValueError as exc:
        assert "chain_clusters_file" in str(exc)
        return
    raise AssertionError("expected ValueError when the cluster file is missing")


def test_tsv_reader_roundtrips(fx):
    m = load_chain_cluster_tsv(fx.clusters_tsv)
    assert m["1abc_A"] == 100 and m["9zzz_A"] == 200
    assert len(m) == 36


def test_datamodule_never_caps_training():
    """The cap kwargs must reach val/test construction sites only."""
    src = (ROOT / "src/data/contact_lit_datamodule.py").read_text()
    for marker in ("self.dset_train = ContactDataset(",
                   "self.dset_trainval = ContactDataset("):
        start = src.index(marker)
        block = src[start:src.index(")", src.index("\n", start))]
        assert "_eval_cap_kwargs" not in block, f"training dataset is capped: {marker}"
    assert src.count("**self._eval_cap_kwargs(),") == 4, "expected 4 eval sites"


for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
        check(name, _with_fixture(fn) if fn.__code__.co_argcount else fn)

print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
sys.exit(1 if FAILED else 0)
