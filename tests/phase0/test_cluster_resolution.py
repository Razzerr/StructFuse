"""Chain → sequence-cluster resolution: the precondition the whole pipeline rests on.

Two failure modes have actually bitten us and both are covered here:

  * `clusters_30*.txt` is keyed by polymer ENTITY, so resolving a chain needs the
    mmCIF `_entity_poly.pdbx_strand_id` bridge. Collapsing entity tokens to PDB
    entries mis-assigned 20.95 % of entries and let 100 %-identity templates past
    the same-cluster filter.
  * The CIF reader is plain Python (no `gemmi` in the experiment env). A
    `;`-delimited multi-line sequence must be ONE token or every later column in
    the loop shifts, which silently mis-maps chains to the wrong entity.

Plain-python: `python tests/phase0/test_cluster_resolution.py`.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.resolve_chain_clusters import (  # noqa: E402
    allows_exact_cluster_lookup,
    build_chain_to_entity_map,
    entity_chain_map_from_file,
    load_cluster_map,
    resolve_cluster,
)

# Entity 1 carries a `;`-delimited sequence and a multi-chain strand list; entity 2
# is the `6xmx_H` case from the retrieval audit; entity 3 is RNA, which RCSB does
# not cluster (it clusters protein sequences only) and which therefore models the
# real "entity present in the file, absent from the cluster map" gap.
CIF = """data_6XMX
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_strand_id
1 'polypeptide(L)'
;MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
APILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ
;
A,B,C
2 'polypeptide(L)' MRTEYCGQLRLSHVGQ H,I
3 polyribonucleotide ACGUACGU R
#
loop_
_atom_site.group_PDB
_atom_site.id
ATOM 1
ATOM 2
"""


def _fixture(tmp: Path) -> Path:
    mmcif = tmp / "mmCIF"
    mmcif.mkdir()
    (mmcif / "6xmx.cif").write_text(CIF)
    return mmcif


def test_entity_map_survives_multiline_field_and_chain_lists():
    with tempfile.TemporaryDirectory() as tmp:
        mapping = entity_chain_map_from_file(
            "6xmx", _fixture(Path(tmp)) / "6xmx.cif"
        )
        # Every strand of entity 1, not just the first.
        for chain in ("a", "b", "c"):
            assert mapping[f"6xmx_{chain}"] == "6xmx_1", mapping
        # Column alignment held past the `;`-delimited sequence block.
        assert mapping["6xmx_h"] == "6xmx_2", mapping
        assert mapping["6xmx_i"] == "6xmx_2", mapping
        assert mapping["6xmx_r"] == "6xmx_3", mapping


def test_empty_entity_map_raises_instead_of_reporting_everything_unmapped():
    """A broken parser must not look like a PDB with no entities."""
    with tempfile.TemporaryDirectory() as tmp:
        mmcif = Path(tmp) / "mmCIF"
        mmcif.mkdir()
        (mmcif / "1abc.cif").write_text("data_1ABC\n#\n")  # no _entity_poly at all
        try:
            build_chain_to_entity_map(mmcif, ["1abc_A"], workers=1)
        except RuntimeError as exc:
            assert "EMPTY chain" in str(exc)
        else:
            raise AssertionError("empty chain→entity map was accepted")


def test_resolve_cluster_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mmcif = _fixture(root)
        clusters = root / "clusters.txt"
        clusters.write_text("6XMX_1 2LCE_1\n6XMX_2 7GUS_1\n")
        member2cluster = load_cluster_map(clusters)
        chain2entity = build_chain_to_entity_map(mmcif, ["6xmx_A", "6xmx_H", "6xmx_R"], workers=1)

        # Resolved through the entity bridge — different entities, different clusters.
        assert resolve_cluster("6xmx_A", member2cluster, chain2entity) == (0, "entity")
        assert resolve_cluster("6xmx_H", member2cluster, chain2entity) == (1, "entity")
        # Entity known but absent from the cluster file (RNA): unusable, not "cluster 0".
        assert resolve_cluster("6xmx_R", member2cluster, chain2entity) == (
            -1,
            "entity_unclustered",
        )
        # No mmCIF for this entry at all.
        assert resolve_cluster("9zzz_A", member2cluster, chain2entity) == (-1, "unmapped")


def test_numeric_chain_suffix_is_not_read_as_an_entity_id():
    """`6xmx_1` as a chain stem means AUTH chain "1", not entity 1."""
    member2cluster = {"6xmx_1": 16149, "af_test_1": 9}
    assert resolve_cluster("6xmx_1", member2cluster, {}) == (-1, "numeric_exact_suppressed")
    # AlphaFold IDs are not PDB chain IDs, so exact lookup stays available.
    assert resolve_cluster("AF_TEST_1", member2cluster, {}) == (9, "exact")
    assert allows_exact_cluster_lookup("af_test_1")
    assert not allows_exact_cluster_lookup("6xmx_1")
    assert allows_exact_cluster_lookup("6xmx_h")


def test_duplicate_cluster_members_keep_first_assignment():
    with tempfile.TemporaryDirectory() as tmp:
        clusters = Path(tmp) / "clusters.txt"
        clusters.write_text("1ABC_1 2DEF_1\n1ABC_1 3GHI_1\n")
        member2cluster = load_cluster_map(clusters)
        assert member2cluster["1abc_1"] == 0
        assert member2cluster["3ghi_1"] == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
