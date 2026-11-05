#!/usr/bin/env python3
"""Copy an index's metadata with cluster ids taken from a new chain_clusters.tsv.

Same rows, same vectors: `faiss.index` and `embeddings.npy` are symlinked, only
`ids.json` is rewritten. Lets a retrieval audit run on corrected cluster labels
without rebuilding the index (e.g. after the chain-ID case fix of 2026-11-05).

Rows whose new label is -1 stay in the copy with cluster_id -1. The runtime
`FaissIndex` refuses such an index by design (unclustered chains must never be
indexed); `retrieval_identity_audit.LightweightFaissIndex` blocks them, which is
what excluding them at build time would do. So the copy is for audits only; a
training/evaluation index must come from `build_index.py`.

Usage:
    python scripts/relabel_index_clusters.py --index-dir data/index_t33_2026 \
        --chain-clusters <new chain_clusters.tsv> --out-dir <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_chain_clusters(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_id, i_cl = header.index("id"), header.index("cluster_id")
        for line in fh:
            if line.strip():
                cols = line.rstrip("\n").split("\t")
                out[cols[i_id]] = int(cols[i_cl])
    return out


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relabel(meta: list, new: dict) -> tuple[list, dict]:
    """Exact (case-sensitive) id lookup; a row missing from `new` is an error."""
    missing = [m["id"] for m in meta if m["id"] not in new]
    if missing:
        raise SystemExit(f"{len(missing)} index rows absent from the new chain_clusters.tsv, e.g. {missing[:5]}")
    out, changed, to_unclustered = [], 0, []
    for m in meta:
        cid = new[m["id"]]
        if cid != int(m.get("cluster_id", -1)):
            changed += 1
        if cid == -1:
            to_unclustered.append(m["id"])
        out.append({**m, "cluster_id": cid})
    return out, {"rows": len(meta), "changed": changed, "now_unclustered": len(to_unclustered),
                 "now_unclustered_ids": to_unclustered}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--chain-clusters", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    src = args.index_dir.resolve()
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src / "ids.json").read_text())
    new_meta, stats = relabel(meta, read_chain_clusters(args.chain_clusters))
    (out / "ids.json").write_text(json.dumps(new_meta))
    for name in ("faiss.index", "embeddings.npy"):
        (out / name).symlink_to(src / name)
    manifest = {
        "source_index": str(src), "chain_clusters": str(args.chain_clusters.resolve()),
        "chain_clusters_md5": md5(args.chain_clusters), "source_ids_json_md5": md5(src / "ids.json"),
        **stats, "purpose": "audit only: runtime FaissIndex refuses rows with cluster_id -1",
    }
    (out / "relabel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rows {stats['rows']} | cluster label changed {stats['changed']} | "
          f"now unclustered {stats['now_unclustered']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
