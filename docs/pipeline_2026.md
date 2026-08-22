# Data pipeline — generation `2026`

Everything derived is tagged by `paths.data_version` (default `"2026"`).
Setting `paths.data_version=2025` reproduces the previous generation from one
Hydra override, and the audit manifest records which build produced each run.

Cluster assignment is a **data-curation precondition**: a chain with no RCSB
cluster cannot be checked by the same-cluster retrieval filter, so it is
excluded from the pipeline entirely rather than special-cased at query time.

---

## 1. Flow

```
 ┌─ SOURCES ────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │  pdb_snapshot_2026/mmCIF          data/clusters_30_2026.txt              │
 │  257,629 entries, 90.3 GB         RCSB DIAMOND @30% id, entity tokens    │
 │  (rsync PDBj, 2026-08)            md5 2dd005893e0c1580dd05c5280097a7fa   │
 │                                   (regenerated weekly → archived here)   │
 └────────────┬──────────────────────────────────┬──────────────────────────┘
              │                                  │
              ▼                                  │
 ┌─ CONTACTS ─────────────────────┐              │
 │ scripts/build_contacts.py      │              │
 │   → data/processed_2026/       │              │
 │     1,037,344 chain NPZs       │              │
 │     {seq, contacts, coords,    │              │
 │      mask, source_path}        │              │
 │                                │              │
 │ scripts/build_npz_lengths.py   │              │
 │   → npz_lengths.json           │              │
 └────────────┬───────────────────┘              │
              │                                  │
              │  chain list                      │  entity → cluster
              ▼                                  ▼
 ┌─ CLUSTER RESOLUTION ─────────────────────────────────────────────────────┐
 │ scripts/resolve_chain_clusters.py    ← the ONLY place chain→cluster lives │
 │   mmCIF _entity_poly.pdbx_strand_id  (AUTH chain ids; stdlib CIF parser)  │
 │   chain → entity → cluster                                               │
 │                                                                          │
 │   → output_splits_2026/chain_clusters.tsv    id, cluster_id, source,     │
 │                                              entity_id                   │
 │   → output_splits_2026/no_cluster_ids.txt     13,513 chains  (1.303%)    │
 │   → output_splits_2026/no_cluster_entries.txt  1,960 entries (zero       │
 │                                                clustered chains)         │
 └────────────┬─────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────────┬──────────────────────┬────────────┐
              ▼                          ▼                      ▼            ▼
 ┌─ SPLITS ──────────────┐  ┌─ EMBEDDINGS ─────┐  ┌─ INDEX ───────┐  ┌ TRAIN ┐
 │ prepare_data_splits   │  │ precompute_esm2  │  │ build_index   │  │ data. │
 │   drop_unclustered_   │  │   _embeddings    │  │   skips       │  │ skip_ │
 │   entries() FIRST     │  │   --skip_ids_    │  │   cluster_id  │  │ ids_  │
 │   cutoff 2024-04-30   │  │     files        │  │   == -1       │  │ files │
 │   + cluster promotion │  │                  │  │               │  └───────┘
 │   → mmcif_final_      │  │ precomputed/     │  │ index_t6_2026 │
 │     splits.json       │  │  esm_t6_8M_2026  │  │ index_t33_2026│
 │                       │  │  esm_t33_650M_   │  │  ├ faiss.index│
 │ split_test_val.py     │  │   2026           │  │  ├ ids.json   │
 │   → val_holdout_ids   │  │                  │  │  └ embeddings │
 │   → test_ids          │  │ {rep, contacts}  │  │     .npy      │
 └───────────────────────┘  └──────────────────┘  └───────────────┘
```

Split sizes (cutoff **2024-04-30** — release date of the first CASP16 target):

| split | entries | note |
|---|---:|---|
| train | 70,304 | 167,756 chains |
| val | 15,442 | from the evaluated post-cutoff pool |
| test | 23,164 | incl. 146,386 cluster-promoted entries |
| CASP16 | 42 | `9b0l` / `9sfa` absent from the snapshot |

The train set is smaller than the 2025 generation (96,393) because the cutoff is
frozen while 19 months of new depositions all land post-cutoff and drag their
clusters across via cluster promotion. Deliberate: the date is principled, the
set size is not.

### Leakage guards (three, independent)

1. **Split** — cluster + temporal, so no cluster spans train and val/test.
2. **Train-time holdout filter** — retrieval drops any candidate whose protein id
   is in `val_holdout_ids ∪ test_ids` (`filter_holdout=True` for train collate
   only). Verified by `verify_no_leak.py`.
3. **Same-cluster filter** — every retrieval path drops candidates sharing the
   query's cluster. Unknown cluster **blocks**; `FaissIndex.__init__` now raises
   if `ids.json` contains any `cluster_id == -1`, so it cannot degrade silently.

---

## 2. State

Done:

- [x] snapshot rsynced; parser equivalence verified (705/705 overlapping chains
      in subdir `i5` differ only in `source_path`)
- [x] `data/processed_2026` — 1,037,344 NPZs; `npz_lengths.json`
- [x] `chain_clusters.tsv`, `no_cluster_ids.txt`, `no_cluster_entries.txt`
- [x] splits regenerated; `verify_data_integrity.py` passes every split and
      cluster invariant, failing only on the two absent index dirs

Blocked / remaining: everything from step 3 down.

---

## 3. Server runbook

Wrapper — note the cache exports. Without `TORCH_HOME` the ESM checkpoint
download goes to `$HOME` and dies on the disk quota.

```bash
cd /mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse
MAMBA_BIN=/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/micromamba/micromamba
MAMBA_ENV=/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/conda/envs/structfuse

R() {  # R <cpus> <mem> <time> "<command>"
  srun --partition=proxima --gpus=1 --cpus-per-task="$1" --mem="$2" --time="$3" \
    bash -c "
      eval \"\$(${MAMBA_BIN} shell hook --shell bash)\"
      micromamba activate ${MAMBA_ENV}
      export HF_HOME=\$PWD/models TORCH_HOME=\$PWD/models PROJECT_ROOT=\$PWD
      export TRITON_CACHE_DIR=\$PWD/.triton_\${SLURM_JOB_ID}
      $4"
}
```

`--gpus=1` is required even for CPU-only jobs on `proxima`.

### Step 1 — embeddings (long pole)

Excludes 13,514 chains — 13,513 unclustered plus `8tz6_B` from
`corrupt_ids.txt` — so 1,023,830 of the 1,037,344 processed chains remain.
Resumable: it skips outputs that already exist, so a walltime kill is safe to
re-issue.

Batches are bounded by `batch * L^2` (`--max_pair_elems`, default 4e6), not by
sequence count: ESM2 materialises one attention map per layer-head over the full
L x L grid, so a fixed `--batch_size` that is trivial at L=100 is 16 GB at
L=1022. Both models completed under this bound (8M in 42 min for the tail after
an earlier OOM, 650M in 13 h 56 m for the full set).

```bash
R 8 64G 24:00:00 "python scripts/precompute_esm2_embeddings.py \
  --data_root data/processed_2026 \
  --output_dir data/precomputed/esm_t6_8M_2026 \
  --model_name esm2_t6_8M_UR50D --batch_size 32"

R 8 96G 48:00:00 "python scripts/precompute_esm2_embeddings.py \
  --data_root data/processed_2026 \
  --output_dir data/precomputed/esm_t33_650M_2026 \
  --model_name esm2_t33_650M_UR50D --batch_size 4"
```

**Check:** the log must print `Excluded 13514 chains via skip lists` before it
loads the model. If it prints `Excluded 0`, the resolver output is missing and
the run must be killed — everything downstream inherits the contamination.

Then confirm the file count:

```bash
ls data/precomputed/esm_t33_650M_2026 | wc -l   # expect 1,023,830
```

### Step 2 — indexes

Fast path: no ESM forward, just mean-pool the cached per-residue reps
(historically ~13 min for 864k chains).

```bash
R 16 128G 04:00:00 "python scripts/build_index.py \
  --processed_dir data/processed_2026 \
  --out_dir data/index_t6_2026 --esm_model esm2_t6_8M_UR50D \
  --precomputed_embeddings_dir data/precomputed/esm_t6_8M_2026 \
  --exclude_ids ''"

R 16 128G 04:00:00 "python scripts/build_index.py \
  --processed_dir data/processed_2026 \
  --out_dir data/index_t33_2026 --esm_model esm2_t33_650M_UR50D \
  --precomputed_embeddings_dir data/precomputed/esm_t33_650M_2026 \
  --exclude_ids ''"
```

`--exclude_ids ''` is deliberate — the index is **full** (val/test included) and
leakage is prevented at query time by guard 2. This is the fix from 2026-04-18;
do not re-introduce build-time exclusion.

**Check:** the log reports how many chains were skipped for `cluster_id == -1`.
Expect that number to be 0, because the unclustered chains have no embeddings to
begin with — they were already dropped in step 1.

### Step 3 — integrity gate

```bash
R 8 64G 02:00:00 "python scripts/verify_data_integrity.py --check-embeddings"
```

Nine assertions, all must pass:

1. `no_cluster_ids.txt` agrees with the `-1` rows in `chain_clusters.tsv`
2. no unclustered chain appears in train / val / test
3. train ∩ val, train ∩ test, val ∩ test are empty **at chain level**
4. …and **at cluster level** — the check the split never had, and the one that
   turns "no structural leak between the sets" into a verified statement
5. every `ids.json` row has `cluster_id >= 0`
6. no unclustered chain is indexed
7. `ids.json` cluster ids agree with `chain_clusters.tsv`
8. `ids.json` has no duplicate ids
9. `len(ids.json) == embeddings.npy.shape[0] == faiss.index.ntotal`

### Step 4 — leakage scan + test suite

```bash
R 16 64G 24:00:00 "python scripts/verify_no_leak.py"   # full scan, no max_queries
R 4  16G 00:30:00 "python -m pytest tests/phase0 -q"
```

`pytest` on the server covers `test_fusion_distance_bins.py` and
`test_no_template_fusion.py`, which cannot run in the local `AI` env.

### Step 4b — evaluation protocol sanity

The first run on the rebuilt data must show, in the test logs and in W&B:

- `Eval cap 8/cluster: kept N of M chains over K clusters` at dataset setup,
  for **val and test only** — if it appears for train, the cap leaked into the
  training set;
- `test/cluster_balanced = 1` and a plausible `test/n_clusters` (~6,958 for the
  full test set, fewer after the cap trims singleton-poor clusters);
- `test/P@L_long` differing from `test/P@L_long_chainmacro` — identical values
  would mean every cluster has one chain, i.e. the cluster map did not load.

`test/cluster_balanced = 0` means `data.chain_clusters_file` never reached the
eval dataset. The run is not lost — `per_sample_metrics.tsv` still carries every
per-chain value — but the logged `test/*` are family-weighted and must be
re-aggregated offline before use.

### Step 5 — 8M gate

Paired, so both the absolute shift and the recomputed retrieval delta are
readable:

- `frontier` k=4 on 8M — compare against `8uhg6bgv` 0.5221 (TruFor) /
  `iej9a561` 0.5187 (grouped)
- `no_templates` on 8M — compare the delta against the current +29.08 pp

This pairing also fills a real gap: **no TruFor no-templates control exists** on
any generation.

Only after the gate reads sane: the full matrix (8M panel; 650M × 3 seeds +
B1/B2/B4).

---

## 4. Open decisions

- **Headline fusion stack.** TruFor+dist won 3/3 650M seeds (+0.55 pp,
  non-overlapping ranges) and 8/9 matched 8M cells. Recommendation: promote it
  and run 3 seeds of that alone rather than 6 covering both. Confirm before the
  650M launches, not before the gate.
- **CASP17 — checked 2026-08-21, answer is NO.** CASP17 is mid-season: targets
  opened 2026-04-27, the last one was entered 2026-07-31, human expirations run
  to 2026-09-11, and experimental coordinates are due 2026-09-01. Of the 239
  listed targets (148 protein) only **12 have a released PDB accession** so far
  (2026-05-27 … 2026-08-12), of which 4 are designed RNA — leaving ~8 protein
  entries, and the two newest (`12er`/`12es`, released 2026-08-12) may postdate
  our snapshot. A subset of n≈6-8 is thinner than CASP16's already-thin n=42 and
  cannot carry a claim.

  Two consequences worth stating rather than assuming. First, every CASP17
  target postdates the 2024-04-30 cutoff, so those entries are **already on the
  test side** — labelling them would add a name, not data, and skipping them
  costs nothing. Second, no CASP17 target can have leaked into training, which
  makes the frozen cutoff more defensible, not less.

  Revisit only if the paper is still in review after the CASP17 results meeting
  (Dec 2026), when the structures and the official assessment are public. Wire-in
  would then be a target list analogous to `data/targetlist.csv`.

## 5. Out of scope

The test set's own redundancy is untouched by any of this: 41,448 chains from
1,283 clusters in the 2025 generation, and 44.9% still retrieved a ≥90%-identity
template after strict filtering, because the near-duplicate pool is deep. §2.3
still needs the identity-stratified table and the cluster-balanced macro. This
work fixes protocol correctness, not benchmark composition.
