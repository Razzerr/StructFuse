"""Does the cached ESM feature at crop row k belong to full-chain residue
crop_start + k?

A consistent feature/label offset is invisible in evaluation — the model learns
the offset and its predictions match the shifted labels — but it degrades
learning, because residue i's features are paired with residue i+1's contacts.
It therefore has to be tested on the indexing itself.

A shift scan over real ESM contacts cannot do this. Those are imperfect
predictors of the truth, so the peak can move for reasons that have nothing to
do with indexing, and a flat feature makes every offset tie. These tests instead
put a KNOWN marker at every position — rep[i] encodes i — and assert the exact
rows and pairs that come out of the real `collate_padded`. Deterministic, no
model, no biology.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.components.dataset import collate_padded  # noqa: E402

D_ESM = 4


def _chain_code(pid: str) -> int:
    """Small stable integer per chain, exact in float16."""
    return 100 + int(pid[1:]) if pid[1:].isdigit() else 100 + (hash(pid) % 900)


def _item(pid: str, length: int, missing: tuple[int, ...] = ()) -> dict:
    """One dataset item whose contact map encodes the pair (i, j) uniquely."""
    idx = np.arange(length)
    contact = (idx[:, None] * 1000 + idx[None, :]).astype(np.float32)
    mask = np.ones(length, dtype=np.float32)
    for i in missing:
        mask[i] = 0.0
    return {
        "pid": pid,
        "seq": "A" * length,
        "contact": contact,
        "mask": mask,
        "coords": np.zeros((length, 3), dtype=np.float32),
        "L": length,
        "subset": "gold",
        "cluster_id": 1,
    }


def _write_cache(dirpath: Path, pid: str, length: int, axis: str = "row") -> None:
    """Cached ESM arrays carrying their own index: rep[i, 0] = i.

    float16 represents integers exactly only up to 2048, so a pair marker like
    i*1000+j silently becomes inf. Mark ONE axis per file instead — `row` writes
    contacts[i, j] = i, `col` writes j — and test the axes separately.
    """
    rep = np.zeros((length, D_ESM), dtype=np.float16)
    rep[:, 0] = np.arange(length)
    # Channel 1 identifies the CHAIN. Without it every cache file carries the
    # same rep[i,0]=i marker, so loading the wrong file for a row is invisible.
    rep[:, 1] = _chain_code(pid)
    idx = np.arange(length)
    marker = idx[:, None] if axis == "row" else idx[None, :]
    contacts = np.broadcast_to(marker, (length, length)).astype(np.float16)
    assert np.isfinite(contacts).all() and length <= 2048
    np.savez(dirpath / f"{pid}.npz", rep=rep, contacts=np.ascontiguousarray(contacts))


def _collate(items, cache_dir, crop_size, crop_mode="center", seed=0,
             esm_max_len=1022):
    return collate_padded(
        items, crop_size=crop_size, crop_mode=crop_mode,
        min_seq_sep=6, seed=seed, esm_embeddings_dir=cache_dir,
        esm_max_len=esm_max_len,
    )


def test_cached_rep_row_matches_full_chain_index():
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", 300)
        b = _collate([_item("p1", 300)], cache, crop_size=128)
        start, end = b["crop_bounds"][0].tolist()
        assert end - start == 128, (start, end)
        got = b["h_esm"][0, : end - start, 0].numpy()
        want = np.arange(start, end, dtype=np.float32)
        assert np.array_equal(got, want), (got[:5], want[:5])


def test_cached_contacts_row_and_column_are_both_aligned():
    for axis, want_fn in (("row", lambda s, k, l: s + k), ("col", lambda s, k, l: s + l)):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            _write_cache(cache, "p1", 300, axis=axis)
            b = _collate([_item("p1", 300)], cache, crop_size=128)
            start, _ = b["crop_bounds"][0].tolist()
            for k, l in [(0, 0), (0, 5), (7, 60), (127, 127)]:
                got = float(b["esm_contacts"][0, 0, k, l])
                assert got == float(want_fn(start, k, l)), (axis, k, l, got)


def test_labels_use_the_same_crop_origin_as_the_features():
    """The load-bearing one: features and labels must share an origin."""
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", 300)
        b = _collate([_item("p1", 300)], cache, crop_size=128)
        start, _ = b["crop_bounds"][0].tolist()
        for k, l in [(3, 40), (10, 100), (60, 61)]:
            label = float(b["contact"][0, k, l])
            feat_row = float(b["esm_contacts"][0, 0, k, l])
            assert label == float((start + k) * 1000 + (start + l)), (k, l, label)
            assert feat_row == float(start + k), (k, l, feat_row)


def test_random_crop_keeps_the_correspondence():
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", 500)
        for seed in range(6):
            b = _collate([_item("p1", 500)], cache, crop_size=96,
                         crop_mode="random", seed=seed)
            start, end = b["crop_bounds"][0].tolist()
            got = b["h_esm"][0, : end - start, 0].numpy()
            assert np.array_equal(got, np.arange(start, end, dtype=np.float32)), seed


def test_multiple_chains_in_one_batch_are_not_swapped():
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        for pid, n in (("p1", 200), ("p2", 260), ("p3", 300)):
            _write_cache(cache, pid, n)
        items = [_item("p1", 200), _item("p2", 260), _item("p3", 300)]
        b = _collate(items, cache, crop_size=64)
        for row, pid in enumerate(b["pid"]):
            start, end = b["crop_bounds"][row].tolist()
            got = b["h_esm"][row, : end - start, 0].numpy()
            assert np.array_equal(got, np.arange(start, end, dtype=np.float32)), pid
            # and the row must carry THIS chain's cache, not another chain's
            code = b["h_esm"][row, : end - start, 1].numpy()
            assert np.all(code == _chain_code(pid)), (
                pid, float(code[0]), _chain_code(pid))


def test_random_training_crop_is_clamped_to_the_cached_prefix():
    """Long chain, short cache: the training draw must stay inside coverage."""
    cov = 150
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", cov)
        for seed in range(8):
            b = _collate([_item("p1", 400)], cache, crop_size=100,
                         crop_mode="random", seed=seed, esm_max_len=cov)
            start, end = b["crop_bounds"][0].tolist()
            assert end <= cov, (seed, start, end)
            got = b["h_esm"][0, : end - start, 0].numpy()
            assert np.array_equal(got, np.arange(start, end, dtype=np.float32)), seed
            assert np.all(b["h_esm"][0, : end - start, 0].numpy() != 0) or start == 0


def test_centre_crop_is_NOT_clamped():
    """Evaluation windows are fully covered already; moving them would be a
    regression, not a fix. At L=1200 the centre crop must stay at [408:792]."""
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", 1022)
        b = _collate([_item("p1", 1200)], cache, crop_size=384, crop_mode="center")
        start, end = b["crop_bounds"][0].tolist()
        assert (start, end) == (408, 792), (start, end)


def test_cache_shortfall_raises_instead_of_zero_filling():
    """A counter would not stop training on wrong data, so this must raise."""
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "p1", 150)
        try:
            # centre crop is deliberately not clamped, so this overruns
            _collate([_item("p1", 400)], cache, crop_size=300, crop_mode="center")
        except ValueError as exc:
            msg = str(exc)
            assert "p1" in msg and "crop=" in msg and "rep_full=" in msg, msg
        else:
            raise AssertionError("silent zero-fill: no exception raised")


def test_pair_mask_equals_outer_of_the_residue_mask_with_padding():
    """Absolute, not relational: two all-zero masks would satisfy lm == pm*sep."""
    missing = (3, 4, 17, 99)
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        _write_cache(cache, "short", 80)
        _write_cache(cache, "p1", 120)
        # a shorter chain in the batch forces padding on the longer one
        b = _collate([_item("p1", 120, missing=missing), _item("short", 80)],
                     cache, crop_size=None)
        row = list(b["pid"]).index("p1")
        pm = b["pair_mask"][row].numpy()
        lm = b["long_mask"][row].numpy()
        n = pm.shape[0]
        assert n >= 120, "fixture must pad to the longest chain"

        expect_mask = np.ones(120, dtype=np.float32)
        for i in missing:
            expect_mask[i] = 0.0
        expect = np.zeros((n, n), dtype=np.float32)
        expect[:120, :120] = np.outer(expect_mask, expect_mask)
        np.fill_diagonal(expect, 0.0)

        assert np.array_equal(pm, expect), "pair_mask != outer(mask, mask)"
        assert pm.sum() > 0, "fixture is degenerate"
        for i in missing:
            assert pm[i].sum() == 0 and pm[:, i].sum() == 0, i

        sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        assert np.array_equal(lm, pm * (sep >= 6))

        # p1 IS the longest chain, so it has no padding and pm[120:] is an empty
        # slice — asserting on it proves nothing. The padded rows live on the
        # SHORT chain, at 80..n.
        srow = list(b["pid"]).index("short")
        spm, slm = b["pair_mask"][srow].numpy(), b["long_mask"][srow].numpy()
        s_expect = np.zeros((n, n), dtype=np.float32)
        s_expect[:80, :80] = 1.0
        np.fill_diagonal(s_expect, 0.0)
        assert np.array_equal(spm, s_expect), "short chain: pair_mask wrong outside 80x80"
        assert spm[80:, :].sum() == 0 and spm[:, 80:].sum() == 0, "padding must be masked"
        assert np.array_equal(slm, spm * (sep >= 6))
        assert n > 80, "fixture must actually pad the short chain"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
