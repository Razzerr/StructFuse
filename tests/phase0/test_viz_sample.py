"""The validation figure shows a chain with contacts, not whichever sorts first.

Plain-python: `python tests/phase0/test_viz_sample.py`.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.utils.visualize import index_with_most_contacts  # noqa: E402


def test_picks_the_chain_with_most_scored_contacts():
    contact, mask = torch.zeros(3, 5, 5), torch.ones(3, 5, 5)
    contact[1, 0, 4] = 1
    contact[2, 0, 4] = contact[2, 1, 4] = 1
    assert index_with_most_contacts(contact, mask) == 2


def test_contacts_outside_the_mask_do_not_count():
    contact, mask = torch.zeros(2, 5, 5), torch.ones(2, 5, 5)
    contact[0, 0, 1] = contact[0, 1, 2] = 1
    mask[0, 0, 1] = mask[0, 1, 2] = 0
    contact[1, 0, 4] = 1
    assert index_with_most_contacts(contact, mask) == 1


def test_ties_and_empty_batches_are_deterministic():
    assert index_with_most_contacts(torch.zeros(3, 4, 4), torch.ones(3, 4, 4)) == 0


def test_the_figure_uses_it():
    src = (ROOT / "src/models/contact_lit_module.py").read_text()
    body = src[src.index("    def _log_visualization("):]
    assert 'idx = index_with_most_contacts(viz_cache["contact"], viz_cache["valid_mask"])' in body[:400]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
