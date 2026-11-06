"""scripts/precompute_esm2_embeddings.py: which chains get a cache.

Plain-python: `python tests/phase0/test_precompute_selection.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.precompute_esm2_embeddings import chains_to_embed  # noqa: E402

FILES = [Path(f"{s}.npz") for s in ("1abc_A", "1abc_a", "2xyz_B", "3mem_A")]


def test_skip_lists_are_chain_level_and_keep_case_twins_apart():
    got = chains_to_embed(FILES, {"1abc_a"})
    assert [p.stem for p in got] == ["1abc_A", "2xyz_B", "3mem_A"], got


def test_entry_list_keeps_only_query_chains():
    got = chains_to_embed(FILES, {"1abc_a"}, {"1abc", "2xyz"})
    assert [p.stem for p in got] == ["1abc_A", "2xyz_B"], got


def test_no_entry_list_means_no_restriction():
    assert len(chains_to_embed(FILES, set(), None)) == len(FILES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
