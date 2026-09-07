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
| test | 23,164 | cluster-promoted count below is UNVERIFIED |
| CASP16 | 42 | `9b0l` / `9sfa` absent from the snapshot |

The train set is smaller than the 2025 generation (96,393) because the cutoff is
frozen while 19 months of new depositions all land post-cutoff and drag their
clusters across via cluster promotion. Deliberate: the date is principled, the
set size is not.

**Open, must be resolved before Methods.** An earlier draft of this table said
test contained "146,386 cluster-promoted entries", which is impossible — the
number exceeds test's own 23,164 entries. It is probably a chain count, or a
count over a different pool, but neither reading has been checked. Resolve it
against the artifact rather than by inference:

```bash
python - <<'EOF'
import json, collections
d = json.load(open("data/output_splits_2026/mmcif_final_splits.json"))
print(type(d), list(d)[:6] if isinstance(d, dict) else len(d))
EOF
```

Then report entries vs chains separately for every split and rewrite the table
with the unit named in the column header.

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

| # | step | state |
|---|---|---|
| 0 | snapshot + `processed_2026` (1,037,344 NPZs) + `npz_lengths.json` | **done** — parser equivalence verified, 705/705 overlapping chains in subdir `i5` differ only in `source_path` |
| 0 | `chain_clusters.tsv`, `no_cluster_ids.txt`, `no_cluster_entries.txt` | **done** |
| 0 | splits regenerated | **done** — every split and cluster invariant passes |
| 1 | embeddings `esm_t6_8M_2026` | **done** — 1,023,830 chains |
| 1 | embeddings `esm_t33_650M_2026` | **done** — 1,023,830 chains, 13 h 56 m |
| 2 | `index_t6_2026`, `index_t33_2026` | **done** — 1,023,830 entries each, `excluded=0`, 34,328 clusters |
| 3 | `verify_data_integrity.py --check-embeddings` | **done** — 22/22, incl. cluster-level disjointness and ids/emb/faiss row parity |
| 4 | `verify_no_leak.py` full scan | **done** — 0 leaks with the filter, 97,586 without (see below) |
| 4 | `pytest tests/phase0` on the server | **done** — 98 passed |
| 4b | evaluation-protocol sanity | pending — read off the first run's logs, not a separate job |
| 5 | 8M gate: `frontier` k=4 + `no_templates`, paired | **done** — `6ppbumcl` / `wnu2h8io`, retrieval Δ = +16.93pp cluster-balanced, contract verified live |
| 6 | full matrix (8M panel; 650M x 3 seeds + B1/B2/B4) | **next** — gate passed |

Every gate is green and the 8M gate has passed; the full matrix is next.

**Leak-scan result, 2026-08-22** (`index_t6_2026`, full scan, no `max_queries`):
167,756 train queries retrieving 671,024 templates. With `filter_holdout=True`,
**0** retrieved templates belong to a val/test protein. With the filter off,
**97,586** do — 14.54 % of all hits. The second number is what makes the first
meaningful: it proves the filter is load-bearing rather than facing an index that
happens to contain no holdout chains. Quote both in Methods, never the zero
alone.

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

### Resolving run artifacts

Never paste checkpoint or TSV paths by hand — resolve them from the task name.
Both helpers take the `task_name` the run was launched with and pick the newest
matching artifact.

```bash
CKPT() {  # CKPT <task_name> -> newest real checkpoint (skips last.ckpt)
  find logs/"$1" -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}
TSV() {   # TSV <task_name> -> newest per_sample_metrics.tsv
  find logs/"$1" -name per_sample_metrics.tsv -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}
```

Check what they resolve to before using them — an empty value means the task
name is wrong or the run never reached the test phase:

```bash
for t in paper_8m_gate_frontier_k4 paper_8m_gate_no_templates; do
  printf '%-32s ckpt=%s\n%-32s tsv =%s\n' "$t" "$(CKPT $t)" "" "$(TSV $t)"
done
```

`R` runs **interactively** and dies with the terminal — a dropped SSH connection
kills the job. Use it only for short gates where you want the answer immediately.
For anything longer than ~15 minutes use the detached form:

```bash
B() {  # B <name> <cpus> <mem> <time> "<command>"
  sbatch --partition=proxima --gpus=1 --job-name="$1" \
         --cpus-per-task="$2" --mem="$3" --time="$4" \
         --output="logs/slurm-$1-%j.out" --wrap "
    eval \"\$(${MAMBA_BIN} shell hook --shell bash)\"
    micromamba activate ${MAMBA_ENV}
    export HF_HOME=\$PWD/models TORCH_HOME=\$PWD/models PROJECT_ROOT=\$PWD
    export TRITON_CACHE_DIR=\$PWD/.triton_\${SLURM_JOB_ID}
    $5"
}
```

Which to use: `B` for embeddings, index builds, `verify_no_leak.py` and every
training run; `R` for `verify_data_integrity.py`, `pytest`, and ad-hoc checks.

### Step 0 — data generation (already done; recorded for reproducibility)

These produced the artifacts marked done in the state table. They are listed
because a reader reproducing the work needs them, and because one of them has a
footgun.

```bash
# a) mmCIF snapshot -> per-chain NPZ, and the length cache
python scripts/build_contacts.py      # see --help; writes data/processed_2026
python scripts/build_npz_lengths.py   # writes npz_lengths.json

# b) the ONLY place chain -> cluster is derived
python scripts/resolve_chain_clusters.py \
  --processed-dir data/processed_2026 \
  --cluster-file  data/clusters_30_2026.txt \
  --out-tsv       data/output_splits_2026/chain_clusters.tsv \
  --out-no-cluster-ids data/output_splits_2026/no_cluster_ids.txt

# c) splits.  --limit_files 0 IS MANDATORY - see the warning below
python scripts/prepare_data_splits.py \
  --input_dir  <mmcif snapshot dir> \
  --output_dir data/output_splits_2026 \
  --clusters_file data/clusters_30_2026.txt \
  --exclude_entries_file data/output_splits_2026/no_cluster_entries.txt \
  --cutoff_date 2024-04-30 \
  --limit_files 0

# d) val / test partition of the evaluated pool.  No CLI - edit the constants at
#    the top of the file (SEED=42, VAL_FRACTION=0.4, CLUSTERS_FILE, OUT_DIR).
python scripts/split_test_val.py
```

> **`prepare_data_splits.py --limit_files` defaults to 100, not to all.** Run
> without the explicit `--limit_files 0` it reads one hundred mmCIF files and
> writes a split that is structurally valid, passes the integrity gate, and is
> completely wrong. There is no error and no warning. Check the entry counts in
> the state table against the produced `mmcif_final_splits.json` before trusting
> any split.

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

The first run on the rebuilt data must also show the validation contract is
live (changed 2026-08-22 — canonical val metrics are cluster-balanced, matching
the headline estimand):

- `Validation contract: cluster_balanced=1 n_clusters=4460` in the run log after
  the first validation epoch. The `val/cluster_balanced` and `val/n_clusters`
  metrics themselves go to W&B only (`prog_bar=False`), so grepping the text log
  for those key names finds nothing — grep for `Validation contract:` instead.
- `val/cluster_balanced = 1` in W&B. A `0` means no usable `cluster_id` reached
  validation: the module logged an error and fell back to the pooled statistic,
  and that run's threshold and checkpoint are NOT comparable to the others.
- `val/n_clusters` close to **4,460**. The cap bounds chains *per* cluster, not
  the number of clusters, so it should not visibly drop — only clusters whose
  chains all lack valid long-range pairs fall out. A large drop is a defect
  signal, not the cap working.
- **Both** `val/f1_long` and `val/f1_long_micro` present, and differing. The
  first is cluster-balanced and is what `ModelCheckpoint` / `EarlyStopping`
  monitor; the second is the old pooled statistic and drives nothing. Identical
  values would mean the cluster map never loaded.

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

## 3b. Experiment plan — stages

Data generation `2026`, cluster-balanced headline, C=8 eval cap, `min_delta=0.001`.
Every stage gates the next. Run counts are cells, not jobs-with-retries.

| # | stage | runs | GPU (measured 8M: ~15 min/epoch, ~27 epochs) | status |
|---|---|---:|---|---|
| **A** | Gate: frontier k=4 + no_templates, paired | 2 | done (10.7 h + 5.7 h) | **done** — Δ +16.93pp |
| **B** | Gate statistics: `paired_significance.py --bootstrap cluster` | 0 | CPU minutes | **next** |
| **C** | Cap sensitivity | 0 train, 3 eval | done | **PASSED** — C=8 vs full: 0.00194 metric / 0.00193 delta, both inside 0.002 (~4% margin) |
| **D** | 8M core panel + TruFor reference cell | 9 | ~60 h train + ~5 h bs=1 eval | **UNBLOCKED** |
| **E** | **DECISION: headline fusion stack** (grouped vs TruFor+dist) | 0 | — | blocked on D |
| **F** | 8M supplementary panel, chosen stack only | 6 | ~40 h | blocked on E |
| **G** | 650M: frontier x3 seeds + B1/B2/B4 | 6 | not measured on this generation | blocked on E |
| **H** | Analyses: paired significance, identity audit, stratification | 0 | CPU | blocked on F/G |
| **I** | Paper: re-cut every number, Tables 1-3, figures | 0 | — | blocked on H |

### Stage D — cells (`launch_paper_8M.sh --core`, plus one TruFor cell)

`frontier_8M` (re-run as `paper_8m_frontier_k4`; the gate run used `min_delta=0` and
ran 44 epochs, so it must NOT be reused as the panel reference — every ablation
would face a longer-trained reference), `tpl_contact_only`, `no_triangle`,
`no_dist`, `no_templates`, `random_retrieval`, `bce_only`, `baseline/esm2_only`,
plus `ablation/trufor_fusion_with_dist` from `launch_trufor_ablation_8M.sh`.

The TruFor cell rides along with the core panel so stage E can be decided as soon
as D lands, instead of paying for a second full panel to answer it.

### Stages B/C — keep the isolation

B: bootstrap over CLUSTERS. C: the SAME selected checkpoints and thresholds, no
re-validation — the only thing that varies is the eval cap. Reconstructing C=8
from the uncapped TSVs must reproduce the ordinary capped result to numerical
tolerance; that equality is the sanity gate, and if it fails the rest of C means
nothing. Acceptance criterion stays as pre-registered: C=8 stands if P@L_long and
the paired retrieval gain each differ from full by <= 0.002 with no qualitative
change.

### Evaluation policy (settled 2026-09-07)

The model is padding-dependent: `PairFeatures` normalises with `InstanceNorm2d`
over the whole L x L map (padding included, in `eval()` too) and axial attention
calls SDPA without `attn_mask`. A chain's prediction therefore depends on which
chains share its batch, which makes runs over different dataset compositions
incomparable. Measured, not assumed: at `eval_batch_size=1` two passes over
different compositions are **bit-identical** (0 of 34,959 chains differ).

**Policy: report from a dedicated eval-only pass at `data.eval_batch_size=1`.**

- **Training is unchanged.** `data.batch_size` still drives the train loader and
  the effective batch after accumulation.
- **In-training validation stays at the training batch size.** Its composition is
  identical in every run (`val_ids` fixed, `rotate_val=false`, cap fixed), so
  checkpoint selection is already comparable run to run, and `bs=1` every epoch
  would be far more expensive than it is worth.
- **Final numbers come from one eval-only pass per cell** with
  `validate_before_test=true data.eval_batch_size=1`, so the F1 threshold is
  calibrated under the same policy it is applied in. `P@L` and `AUC-PR` are
  threshold-free and unaffected by that detail.
- The architectural sensitivity itself remains, and gets one Methods sentence:
  batching changes predictions, so evaluation is performed one chain at a time.

Fixing the model instead (masked normalisation + `attn_mask`) is the more correct
change but alters the architecture, forces a full retrain and loses the
FlashAttention-2 path. It is not justified by the measured effect: removing
padding *raises* both models (+1.02pp frontier, +1.44pp no-templates) and moves
the retrieval delta by −0.42pp, with every conclusion intact.

### Stage E — the decision, and why it is not already made

On the 2025 generation TruFor+dist beat grouped in 8/9 matched 8M cells and 3/3
650M seeds (+0.55pp). But **that margin is below the scale of the aggregation
correction** — the 2026-08-21 decision recorded it as provisional precisely
because it was measured with chain-level bootstrap. It must be re-established on
the rebuilt data with cluster resampling before it can move the headline.

Deciding first halves stages F and G: one stack, not two.

**Selection rule, fixed before the results are seen:** choose on
**cluster-balanced validation** (`val/f1_long`, the monitored key), and use test
only for the final report. TruFor is not assumed to win, and the project does not
depend on it winning — grouped remaining the headline is an equally acceptable
outcome, and is the cheaper one, since the 650M grouped configs already exist.

### Stage F — the ablation panel must match the chosen model

This is the part the stage table above states too loosely. Stage D is mostly
GROUPED ablations plus one full TruFor cell. If E selects TruFor, a
"grouped minus triangle" number does NOT measure triangle's contribution inside
TruFor — different fusion, different gradient path.

`launch_paper_8M.sh --supplementary` hardcodes six grouped experiments and does
not switch with the decision. Two things follow:

1. Stage F's six cells have to be re-pointed at the chosen stack, and matched to
   the claims actually going into the manuscript — not run wholesale in both
   variants.
2. **Two TruFor cells do not exist yet** and are needed if TruFor wins:
   - `ablation/trufor_no_templates` — without it the retrieval kill-switch
     (the headline +16.93pp) would be measured on the grouped stack while the
     headline model is TruFor. This is the single most important control in
     Section 2.3.
   - `ablation/trufor_no_dist` — the `(-dist, +triangle)` cell, absent since the
     2026-07-22 panel, which is why the distance x triangle interaction term has
     never been computable on the TruFor stack.

   Do not create them speculatively; create them if and when E selects TruFor.

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
