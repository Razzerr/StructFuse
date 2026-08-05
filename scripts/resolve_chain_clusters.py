#!/usr/bin/env python3
"""Resolve the authoritative chain -> sequence-cluster map for the whole dataset.

This is a data-curation *precondition*: it runs before the FAISS index exists and
before the splits are drawn, and everything downstream consumes its output.

RCSB's `clusters_30.txt` (DIAMOND @ 30 % identity, regenerated weekly) is keyed by
polymer **entity** — `12E8_2` means entry 12E8, entity 2 — while retrieval works on
**chains**. The bridge is the mmCIF `_entity_poly.pdbx_strand_id` record, which lists
AUTH chain IDs and therefore matches the chain IDs used everywhere else.

Chains whose entity carries no cluster assignment are reported as `-1` and written to
a skip list. They are then excluded from the splits, the index and training — an
inclusion criterion, not a runtime special case. Treating "unknown" as admissible was
the defect that let identical-sequence templates through the same-cluster filter.

Outputs (see --out-*):
  chain_clusters.tsv       stem, cluster_id, source, entity_id   (every chain)
  no_cluster_ids.txt       stems that resolved to -1
  no_cluster_entries.txt   PDB entries where EVERY chain resolved to -1

Deliberately stdlib + tqdm only, so this step never depends on the experiment
environment staying in a particular state — the mmCIF reader below is plain Python.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def norm_id(value: str) -> str:
    return str(value).strip().lower()


def split_strand_ids(value: str) -> List[str]:
    if value is None:
        return []
    raw = str(value).strip().strip("'\"")
    if raw in ("", ".", "?"):
        return []
    return [tok.strip().strip("'\"") for tok in raw.replace(";", ",").split(",") if tok.strip()]


def load_cluster_map(cluster_file: Path) -> Dict[str, int]:
    member2cluster: Dict[str, int] = {}
    n_clusters = 0
    n_duplicates = 0
    with cluster_file.open() as handle:
        for line in handle:
            toks = line.strip().split()
            if not toks:
                continue
            cid = n_clusters
            for tok in toks:
                member = norm_id(tok)
                if member in member2cluster and member2cluster[member] != cid:
                    n_duplicates += 1
                    continue
                member2cluster[member] = cid
            n_clusters += 1
    logger.info("Loaded %d clusters covering %d members", n_clusters, len(member2cluster))
    if n_duplicates:
        logger.warning("Ignored %d duplicate cluster-member assignments", n_duplicates)
    return member2cluster


def index_mmcif_paths(mmcif_dir: Path, wanted_pdbs: Set[str]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for pattern in ("**/*.cif.gz", "**/*.cif"):
        for path in mmcif_dir.glob(pattern):
            name = path.name
            if name.endswith(".cif.gz"):
                pdb = name[:-7].lower()
            elif name.endswith(".cif"):
                pdb = name[:-4].lower()
            else:
                continue
            if pdb in wanted_pdbs and pdb not in paths:
                paths[pdb] = path
    missing = wanted_pdbs.difference(paths)
    if missing:
        logger.warning("Missing mmCIF paths for %d PDB IDs", len(missing))
    return paths


# Everything we need (`_entity_poly`) appears well before the coordinate table,
# so tokenising stops there instead of walking millions of `_atom_site` rows.
_CIF_STOP_TAG = "_atom_site."
_CIF_KEYWORDS = ("data_", "save_", "stop_", "global_")


def _split_cif_line(line: str) -> List[str]:
    """Split one CIF line into tokens, honouring quotes and trailing comments.

    A quote only closes when followed by whitespace or end-of-line, which is the
    CIF rule that lets values such as ``5'-end`` stay a single token.
    """
    out: List[str] = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            quote = ch
            i += 1
            start = i
            while i < n and not (line[i] == quote and (i + 1 >= n or line[i + 1] in " \t")):
                i += 1
            out.append(line[start:i])
            i += 1
            continue
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        out.append(line[start:i])
    return out


def _cif_tokens(path: Path) -> Iterator[str]:
    """Yield CIF tokens: tags, values, and the ``loop_`` keyword.

    Multi-line ``;``-delimited text fields (e.g. the one-letter sequence inside
    the `_entity_poly` loop) are emitted as a SINGLE token, which is what keeps
    loop columns aligned. Written in plain Python on purpose: this script must
    run without depending on the experiment environment.
    """
    open_func = gzip.open if str(path).endswith(".gz") else open
    with open_func(path, "rt", encoding="latin-1", errors="replace") as handle:
        in_text = False
        buf: List[str] = []
        for line in handle:
            if in_text:
                if line.startswith(";"):
                    in_text = False
                    yield "\n".join(buf)
                    buf = []
                    rest = line[1:].strip()
                    if rest:
                        yield from _split_cif_line(rest)
                else:
                    buf.append(line.rstrip("\n"))
                continue
            if line.startswith(";"):
                in_text = True
                buf = [line[1:].rstrip("\n")]
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith(_CIF_STOP_TAG):
                return
            yield from _split_cif_line(stripped)


def entity_chain_map_from_file(pdb_id: str, path: Path) -> Dict[str, str]:
    """Map ``<pdb>_<auth_chain>`` → ``<pdb>_<entity>`` for one mmCIF file.

    `_entity_poly.pdbx_strand_id` lists AUTH chain IDs, which is the namespace
    `ids.json` uses (gemmi's ``make_structure_from_block`` names chains by
    ``auth_asym_id``). `_pdbx_poly_seq_scheme` is a fallback for the rare files
    without an `_entity_poly` category; `_struct_asym` is deliberately NOT used
    because it carries label_asym_id, a different namespace.
    """
    pdb = norm_id(pdb_id)
    poly: Dict[str, str] = {}       # entity_id -> raw strand-id value
    scheme: Dict[str, str] = {}     # auth chain -> entity_id (fallback)
    single: Dict[str, str] = {}     # non-loop `_entity_poly.*` tag -> value

    it = _cif_tokens(path)
    token = next(it, None)
    while token is not None:
        if token == "loop_":
            tags: List[str] = []
            token = next(it, None)
            while token is not None and token.startswith("_"):
                tags.append(token.lower())
                token = next(it, None)
            if not tags:
                continue
            cols = {tag: i for i, tag in enumerate(tags)}
            e_i = cols.get("_entity_poly.entity_id")
            s_i = cols.get("_entity_poly.pdbx_strand_id")
            p_e = cols.get("_pdbx_poly_seq_scheme.entity_id")
            p_c = cols.get("_pdbx_poly_seq_scheme.pdb_strand_id")
            row: List[str] = []
            while (
                token is not None
                and not token.startswith("_")
                and token != "loop_"
                and not token.lower().startswith(_CIF_KEYWORDS)
            ):
                row.append(token)
                if len(row) == len(tags):
                    if e_i is not None and s_i is not None:
                        poly[row[e_i]] = row[s_i]
                    elif p_e is not None and p_c is not None:
                        scheme.setdefault(row[p_c], row[p_e])
                    row = []
                token = next(it, None)
            continue
        if token.startswith("_"):
            tag = token.lower()
            value = next(it, None)
            if tag.startswith("_entity_poly.") and value is not None:
                single[tag] = value
            token = next(it, None)
            continue
        token = next(it, None)

    entity = single.get("_entity_poly.entity_id")
    strands = single.get("_entity_poly.pdbx_strand_id")
    if entity is not None and strands is not None:
        poly.setdefault(entity, strands)

    out: Dict[str, str] = {}
    for entity_id, strand_value in poly.items():
        entity_id = entity_id.strip().strip("'\"")
        if entity_id in ("", ".", "?"):
            continue
        entity_stem = norm_id(f"{pdb}_{entity_id}")
        for chain in split_strand_ids(strand_value):
            out[norm_id(f"{pdb}_{chain}")] = entity_stem
    if not out:
        for chain, entity_id in scheme.items():
            chain = chain.strip().strip("'\"")
            entity_id = entity_id.strip().strip("'\"")
            if chain in ("", ".", "?") or entity_id in ("", ".", "?"):
                continue
            out[norm_id(f"{pdb}_{chain}")] = norm_id(f"{pdb}_{entity_id}")
    return out


def _entity_map_worker(item: tuple) -> tuple:
    """Pool worker: parse one mmCIF, never raise (errors travel as a string)."""
    pdb, path = item
    try:
        return entity_chain_map_from_file(pdb, Path(path)), None
    except Exception as exc:  # noqa: BLE001
        return {}, f"{path}: {exc}"


def build_chain_to_entity_map(
    mmcif_dir: Path, stems: Iterable[str], workers: int = 1
) -> Dict[str, str]:
    wanted_pdbs = {norm_id(stem).split("_")[0] for stem in stems}
    paths = index_mmcif_paths(mmcif_dir, wanted_pdbs)
    chain2entity: Dict[str, str] = {}
    n_failed = 0
    items = [(pdb, str(path)) for pdb, path in paths.items()]
    desc = f"Parsing mmCIF entity-chain maps (workers={workers})"

    def _consume(results):
        nonlocal n_failed
        for mapping, err in tqdm(results, total=len(items), desc=desc):
            if err is not None:
                n_failed += 1
                if n_failed <= 10:
                    logger.warning("Could not parse %s", err)
                continue
            chain2entity.update(mapping)

    if workers > 1 and len(items) > 1:
        # Each file is independent and the merge is a plain dict update, so this
        # is embarrassingly parallel. Processes (not threads) because the CIF
        # tokeniser is pure Python and GIL-bound.
        chunksize = max(1, min(256, len(items) // (workers * 8) or 1))
        with mp.get_context("fork").Pool(workers) as pool:
            _consume(pool.imap_unordered(_entity_map_worker, items, chunksize=chunksize))
    else:
        _consume(_entity_map_worker(item) for item in items)

    if n_failed:
        logger.warning("Failed to parse %d mmCIF files", n_failed)
    logger.info("Built chain→entity map for %d chain IDs", len(chain2entity))
    # Fail loudly instead of silently reporting every chain as `unmapped`: an
    # empty map means the parser broke, not that the PDB has no entities.
    if paths and not chain2entity:
        raise RuntimeError(
            f"Parsed {len(paths)} mmCIF files but produced an EMPTY chain→entity map. "
            "Refusing to continue — every chain would be reported as 'unmapped'."
        )
    return chain2entity


def allows_exact_cluster_lookup(norm_stem: str) -> bool:
    """Avoid treating numeric PDB chain IDs as polymer entity IDs."""
    if norm_stem.startswith("af_"):
        return True
    suffix = norm_stem.rsplit("_", 1)[-1] if "_" in norm_stem else ""
    return not suffix.isdigit()


def resolve_cluster(
    stem: str, member2cluster: Dict[str, int], chain2entity: Dict[str, str]
) -> tuple[int, str]:
    key = norm_id(stem)
    entity = chain2entity.get(key)
    if entity is not None and entity in member2cluster:
        return member2cluster[entity], "entity"
    if entity is not None:
        return -1, "entity_unclustered"
    if not allows_exact_cluster_lookup(key):
        return -1, "numeric_exact_suppressed"
    if key in member2cluster:
        return member2cluster[key], "exact"
    return -1, "unmapped"


def load_chain_stems(processed_dir: Path, chain_list: Path | None) -> List[str]:
    """Every chain the pipeline knows about, in a stable (sorted) order.

    Default source is ``<processed_dir>/npz_lengths.json`` (built by
    ``scripts/build_npz_lengths.py``), which is the same index ``ContactDataset``
    uses — so the cluster map covers exactly the chains that can ever be loaded.
    """
    if chain_list is not None:
        stems = [ln.strip() for ln in chain_list.read_text().splitlines() if ln.strip()]
        logger.info("Loaded %d chain stems from %s", len(stems), chain_list)
        return sorted(set(stems))

    lengths_path = processed_dir / "npz_lengths.json"
    if not lengths_path.exists():
        raise FileNotFoundError(
            f"{lengths_path} not found. Build it first with:\n"
            f"  python scripts/build_npz_lengths.py --processed_dir {processed_dir}"
        )
    with lengths_path.open() as handle:
        stems = sorted(json.load(handle).keys())
    logger.info("Loaded %d chain stems from %s", len(stems), lengths_path)
    return stems


def write_lines(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(lines)
    with path.open("w") as handle:
        for item in items:
            handle.write(f"{item}\n")
    return len(items)


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed_2026")
    parser.add_argument(
        "--chain-list",
        default="",
        help="Optional explicit file of chain stems (one per line); "
        "overrides <processed-dir>/npz_lengths.json.",
    )
    parser.add_argument("--cluster-file", default="data/clusters_30_2026.txt")
    parser.add_argument(
        "--mmcif-dir",
        default="/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2026/mmCIF",
    )
    parser.add_argument("--out-tsv", default="data/output_splits_2026/chain_clusters.tsv")
    parser.add_argument("--out-no-cluster-ids", default="data/output_splits_2026/no_cluster_ids.txt")
    parser.add_argument(
        "--out-no-cluster-entries", default="data/output_splits_2026/no_cluster_entries.txt"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Parallel mmCIF parser processes. Default is deliberately modest so "
            "this stays polite on a login node; pass a higher value inside an "
            "srun/sbatch allocation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resolution statistics without writing any output file.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    cluster_file = Path(args.cluster_file)
    mmcif_dir = Path(args.mmcif_dir)
    if not cluster_file.exists():
        raise FileNotFoundError(cluster_file)
    if not mmcif_dir.exists():
        raise FileNotFoundError(mmcif_dir)

    stems = load_chain_stems(processed_dir, Path(args.chain_list) if args.chain_list else None)
    workers = max(1, int(args.workers))
    member2cluster = load_cluster_map(cluster_file)
    chain2entity = build_chain_to_entity_map(mmcif_dir, stems, workers=workers)

    rows: list[dict[str, object]] = []
    stats: Dict[str, int] = {}
    no_cluster: List[str] = []
    per_entry_total: Dict[str, int] = defaultdict(int)
    per_entry_unclustered: Dict[str, int] = defaultdict(int)

    for stem in stems:
        cluster_id, source = resolve_cluster(stem, member2cluster, chain2entity)
        stats[source] = stats.get(source, 0) + 1
        entry = norm_id(stem).split("_")[0]
        per_entry_total[entry] += 1
        if cluster_id == -1:
            no_cluster.append(stem)
            per_entry_unclustered[entry] += 1
        rows.append(
            {
                "id": stem,
                "cluster_id": cluster_id,
                "source": source,
                "entity_id": chain2entity.get(norm_id(stem), ""),
            }
        )

    # An entry is dropped from the splits only when NONE of its chains is
    # clustered; a partially-clustered entry keeps its usable chains.
    dead_entries = sorted(
        entry for entry, n in per_entry_unclustered.items() if n == per_entry_total[entry]
    )

    logger.info("Chains: %d", len(stems))
    logger.info("Assignment sources: %s", stats)
    logger.info(
        "Unclustered chains: %d (%.3f%%) across %d entries; %d entries fully unclustered",
        len(no_cluster),
        100.0 * len(no_cluster) / max(len(stems), 1),
        len({norm_id(s).split("_")[0] for s in no_cluster}),
        len(dead_entries),
    )

    if args.dry_run:
        logger.info("Dry run: no files written")
        return

    write_audit(Path(args.out_tsv), rows)
    logger.info("Wrote %s (%d rows)", args.out_tsv, len(rows))
    n = write_lines(Path(args.out_no_cluster_ids), no_cluster)
    logger.info("Wrote %s (%d chains)", args.out_no_cluster_ids, n)
    n = write_lines(Path(args.out_no_cluster_entries), dead_entries)
    logger.info("Wrote %s (%d entries)", args.out_no_cluster_entries, n)


if __name__ == "__main__":
    main()
