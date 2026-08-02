#!/usr/bin/env python3
"""Rebuild only the cluster_id fields in an existing FAISS ids.json.

This fixes the multi-entity PDB bug where cluster IDs were assigned after
collapsing chain/entity IDs to a PDB-level protein ID. The script does not read
or modify `faiss.index` or `embeddings.npy`; it only rewrites metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import shutil
import tempfile
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
    run without touching the experiment environment.
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


def build_chain_to_entity_map(mmcif_dir: Path, stems: Iterable[str]) -> Dict[str, str]:
    wanted_pdbs = {norm_id(stem).split("_")[0] for stem in stems}
    paths = index_mmcif_paths(mmcif_dir, wanted_pdbs)
    chain2entity: Dict[str, str] = {}
    n_failed = 0
    for pdb, path in tqdm(paths.items(), desc="Parsing mmCIF entity-chain maps"):
        try:
            chain2entity.update(entity_chain_map_from_file(pdb, path))
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            if n_failed <= 10:
                logger.warning("Could not parse %s: %s", path, exc)
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


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


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
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--cluster-file", default="data/clusters_30.txt")
    parser.add_argument(
        "--mmcif-dir",
        default="/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2025/mmCIF",
    )
    parser.add_argument("--out-ids-json", default="")
    parser.add_argument("--audit-tsv", default="")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    index_dir = Path(args.index_dir)
    ids_path = index_dir / "ids.json"
    out_path = Path(args.out_ids_json) if args.out_ids_json else ids_path
    audit_path = (
        Path(args.audit_tsv)
        if args.audit_tsv
        else index_dir / "ids_cluster_metadata_rebuild_audit.tsv"
    )

    if not ids_path.exists():
        raise FileNotFoundError(ids_path)
    cluster_file = Path(args.cluster_file)
    mmcif_dir = Path(args.mmcif_dir)
    if not cluster_file.exists():
        raise FileNotFoundError(cluster_file)
    if not mmcif_dir.exists():
        raise FileNotFoundError(mmcif_dir)

    logger.info("Loading %s", ids_path)
    with ids_path.open() as handle:
        meta = json.load(handle)
    stems = [m["id"] for m in meta]

    member2cluster = load_cluster_map(cluster_file)
    chain2entity = build_chain_to_entity_map(mmcif_dir, stems)

    rows: list[dict[str, object]] = []
    stats: Dict[str, int] = {}
    changed = 0
    for m in meta:
        old = int(m.get("cluster_id", -1))
        new, source = resolve_cluster(m["id"], member2cluster, chain2entity)
        if old != new:
            changed += 1
        stats[source] = stats.get(source, 0) + 1
        rows.append(
            {
                "id": m["id"],
                "old_cluster_id": old,
                "new_cluster_id": new,
                "source": source,
                "entity_id": chain2entity.get(norm_id(m["id"]), ""),
                "changed": int(old != new),
            }
        )
        m["cluster_id"] = int(new)

    logger.info("Entries: %d", len(meta))
    logger.info("Changed cluster_id: %d", changed)
    logger.info("Assignment sources: %s", stats)

    write_audit(audit_path, rows)
    logger.info("Wrote audit TSV: %s", audit_path)

    if args.dry_run:
        logger.info("Dry run: not writing ids.json")
        return

    if args.backup and out_path == ids_path:
        backup_path = ids_path.with_suffix(".json.pre_chain_entity_fix.bak")
        if not backup_path.exists():
            shutil.copy2(ids_path, backup_path)
            logger.info("Wrote backup: %s", backup_path)
        else:
            logger.info("Backup already exists: %s", backup_path)

    write_json_atomic(out_path, meta)
    logger.info("Wrote rebuilt ids.json: %s", out_path)


if __name__ == "__main__":
    main()
