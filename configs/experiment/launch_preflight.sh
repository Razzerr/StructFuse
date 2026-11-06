#!/bin/bash
# Paper preflight for the 650M/index_t33 path.
#
# Usage:
#   ./configs/experiment/launch_preflight.sh --all
#   ./configs/experiment/launch_preflight.sh --only verify
#   ./configs/experiment/launch_preflight.sh --all --dry-run
#   ./configs/experiment/launch_preflight.sh --only cost-smoke      # then --only cost
#   ./configs/experiment/launch_preflight.sh --only export-maps     # Figure 4, headline
#   ./configs/experiment/launch_preflight.sh --only export-maps-notpl
#   ./configs/experiment/launch_preflight.sh --only esmfold-casp16-smoke   # then --only esmfold-casp16
#   ./configs/experiment/launch_preflight.sh --only casefix-audit   # chain-ID case fix: labels + identity audit
#   ./configs/experiment/launch_preflight.sh --only cb-rebuild      # Cβ contact labels -> data/processed_2026cb + gate
#   ./configs/experiment/launch_preflight.sh --only cb-low-list     # skip list of Cα-only chains (>50 % Cα fallback)
#   ./configs/experiment/launch_preflight.sh --only seq-overlap     # val/test chains that are the same protein as a train chain
#   ./configs/experiment/launch_preflight.sh --only cb-data         # 2026cb: missing embeddings -> indexes -> integrity gate
#   ./configs/experiment/launch_preflight.sh --only verify-clusters # chain cluster labels re-derived independently of the resolver
#   ./configs/experiment/launch_preflight.sh --only pytest          # whole tests/phase0 with the real parasail, gemmi and torch
# cost*, export-maps*, esmfold*, casefix-audit, cb-rebuild, cb-low-list, seq-overlap, cb-data, verify-clusters and pytest are opt-in and never part of --all.

set -euo pipefail

DRY_RUN=false
MODE=""
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            MODE="all"
            shift
            ;;
        --only)
            MODE="only"
            ONLY="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '2,19p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --all or --only {verify|coverage|coverage8m|ceiling|correlation|cost-smoke|cost|export-maps|export-maps-notpl|esmfold-casp16-smoke|esmfold-casp16|casefix-audit|cb-rebuild|cb-low-list|seq-overlap|cb-data|verify-clusters|pytest}." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

# Hydra overrides per job. They are NOT all the same: `diagnostics/ceiling` is
# the 2025 diagnostic config and composes the default 8M backbone, so running
# the coverage scan under it would report coverage for the wrong stack. Coverage
# is pinned to the 650M headline experiment with the evaluation crop policy.
submit_preflight() {
    local key="$1"
    local script="$2"
    local job_name="$3"
    local time_limit="$4"
    local overrides="${5:-experiment=diagnostics/ceiling}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ${script} ${overrides} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 16 "256G" >/dev/null
}

run_selected() {
    local key="$1"
    local script="$2"
    local job_name="$3"
    local time_limit="$4"
    local overrides="${5:-}"
    # `cost*` and `export-maps*` are opt-in: --all is the data preflight, not a
    # benchmark run and not the figure export.
    if [[ ( "${key}" == cost* || "${key}" == export-maps* ) && "${MODE}" == "all" ]]; then
        return
    fi
    if [[ "${MODE}" == "all" || "${ONLY}" == "${key}" ]]; then
        submit_preflight "${key}" "${script}" "${job_name}" "${time_limit}" "${overrides}"
    fi
}

if [[ "${MODE}" == "only" ]]; then
    case "${ONLY}" in
        verify|coverage|coverage8m|ceiling|correlation|cost|cost-smoke|export-maps|export-maps-notpl|esmfold-casp16-smoke|esmfold-casp16|casefix-audit|cb-rebuild|cb-low-list|seq-overlap|cb-data|verify-clusters|pytest) ;;
        *)
            echo "Invalid --only value: ${ONLY}" >&2
            exit 2
            ;;
    esac
fi

echo "650M preflight, commit=$(git_commit), dry_run=${DRY_RUN}"
run_selected "verify" "scripts/verify_no_leak.py" "paper_verify_no_leak_t33" "24:00:00"
run_selected "coverage" "scripts/template_coverage.py" "paper_template_coverage_650m_2026" "08:00:00" \
    "experiment=trufor_fusion_with_dist_650M data.crop_mode=center data.num_workers=8"
run_selected "coverage8m" "scripts/template_coverage.py" "paper_template_coverage_8m_2026" "08:00:00" \
    "experiment=ablation/trufor_fusion_with_dist data.crop_mode=center data.num_workers=8"
# cost-smoke runs 3 chains: the first real exercise of the timing path, so a
# mistake costs seconds. Only run `cost` after it comes back clean.
run_selected "cost-smoke" "scripts/cost_benchmark.py" "paper_cost_smoke_650m" "00:30:00" \
    "experiment=trufor_fusion_with_dist_650M data.crop_mode=center data.num_workers=0 \
     +cost.n_chains=3 +cost.repeats=1 +cost.warmup=1"
run_selected "cost" "scripts/cost_benchmark.py" "paper_cost_650m" "04:00:00" \
    "experiment=trufor_fusion_with_dist_650M data.crop_mode=center data.num_workers=0 \
     +cost.n_chains=200 +cost.repeats=5 +cost.warmup=2"
# ── Figure 4 export: the ten pre-registered qualitative cases, one job per
# trained model. Provenance is PINNED to the two bs=1 evaluations the paper
# reports (final_runs.md), not discovered as "the newest file": the exporter
# takes the checkpoint path from the evaluation's audit manifest and refuses
# unless the file it loads reproduces that manifest's fingerprint, reads the
# threshold from the evaluation's eval_metrics.json at full precision, and
# refuses unless every per-sample metric of that evaluation is reproduced.
# `paper/` is untracked: copy export_requests.tsv to the server first.
EXPORT_REQUESTS="${EXPORT_REQUESTS:-paper/qualitative_cases/artifacts_650m_s42_2026/export_requests.tsv}"

run_export() {
    local key="$1" experiment="$2" eval_run_id="$3" eval_run_dir="$4" tag="$5"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local audit="${TEMP_DIR}/audit/${eval_run_id}/manifest.json"
    local run_dir="${LOGS_DIR}/${eval_run_dir}"
    if [[ "${DRY_RUN}" != "true" ]]; then
        local missing=""
        [[ -f "${SOURCE_ROOT}/${EXPORT_REQUESTS}" ]] || missing="${missing} ${EXPORT_REQUESTS}"
        [[ -f "${audit}" ]] || missing="${missing} ${audit}"
        [[ -f "${run_dir}/per_sample_metrics.tsv" ]] || missing="${missing} ${run_dir}/per_sample_metrics.tsv"
        [[ -f "${run_dir}/eval_metrics.json" ]] || missing="${missing} ${run_dir}/eval_metrics.json"
        if [[ -n "${missing}" ]]; then
            echo "SKIP ${key}: pinned inputs missing:${missing}" >&2
            return
        fi
    fi
    submit_preflight "${key}" "scripts/export_contact_maps.py" "paper_export_maps_${tag}" "01:00:00" \
        "experiment=${experiment} data.eval_batch_size=1 data.num_workers=0 \
         +export.requests=${EXPORT_REQUESTS} +export.tag=${tag} \
         +export.eval_run_dir=${run_dir} +export.eval_audit_manifest=${audit}"
}
# The two bs=1 evaluations of final_runs.md: W&B run id and hydra run dir.
run_export "export-maps"       "trufor_fusion_with_dist_650M" "n87663ed" "paper_650m_trufor_s42_bs1/runs/2026-09-11_08-45-58"              "structfuse"
run_export "export-maps-notpl" "trufor_no_templates_650M"     "qux75yw4" "paper_650m_trufor_no_templates_s42_bs1/runs/2026-09-11_08-47-11" "no_templates"

# ── External reference: ESMFold on the evaluated CASP16 chains (opt-in). ──
# One-off first, on the login node: bash scripts/setup_esmfold_env.sh
# Labels come from the evaluation collate of a no-retrieval experiment; the gate
# checks them against the pinned bs=1 table of the headline run (n87663ed).
ESMFOLD_VENV="${ESMFOLD_VENV:-/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/venvs/esmfold}"
run_esmfold() {
    local key="$1" max_chains="$2" time_limit="$3"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local ref="${LOGS_DIR}/paper_650m_trufor_s42_bs1/runs/2026-09-11_08-45-58/per_sample_metrics.tsv"
    if [[ "${DRY_RUN}" != "true" ]]; then
        local missing=""
        [[ -x "${ESMFOLD_VENV}/bin/python" ]] || missing="${missing} ${ESMFOLD_VENV} (run scripts/setup_esmfold_env.sh)"
        [[ -f "${ref}" ]] || missing="${missing} ${ref}"
        if [[ -n "${missing}" ]]; then
            echo "SKIP ${key}: inputs missing:${missing}" >&2
            return
        fi
    fi
    local extra=""
    [[ -n "${max_chains}" ]] && extra="+ext.max_chains=${max_chains}"
    local job_name="paper_${key//-/_}"
    local command="HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ESMFOLD_VENV}/bin/python scripts/esmfold_predict.py \
experiment=trufor_no_templates_650M data.esm_embeddings_dir=null data.eval_batch_size=1 data.num_workers=0 \
+ext.subset=casp16 +ext.reference_tsv=${ref} ${extra} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 8 "128G" >/dev/null
}
run_esmfold "esmfold-casp16-smoke" 3  "00:30:00"
run_esmfold "esmfold-casp16"       "" "03:00:00"

# Chain-ID case fix (2026-11-05). One job, nothing of the 2026 generation is
# overwritten: (1) re-resolve chain -> cluster with the fixed resolver into
# data/casefix_2026/, (2) copy index_t33_2026 metadata with those labels (same
# rows, symlinked vectors; audit-only), (3) replay retrieval for the evaluated
# test chains on the corrected labels and audit top-1 identity. No control TSV:
# the old model's metrics do not belong to the new templates.
CASEFIX_DIR="data/casefix_2026"
run_casefix() {
    local key="casefix-audit"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    if [[ "${DRY_RUN}" != "true" && -e "${PROJECT_DIR}/${CASEFIX_DIR}" ]]; then
        echo "SKIP ${key}: ${PROJECT_DIR}/${CASEFIX_DIR} already exists; move it away to rerun" >&2
        return
    fi
    local job_name="paper_casefix_audit"
    local command="{ mkdir -p ${CASEFIX_DIR} && \
python scripts/resolve_chain_clusters.py --processed-dir data/processed_2026 --cluster-file data/clusters_30_2026.txt \
--out-tsv ${CASEFIX_DIR}/chain_clusters.tsv --out-no-cluster-ids ${CASEFIX_DIR}/no_cluster_ids.txt \
--out-no-cluster-entries ${CASEFIX_DIR}/no_cluster_entries.txt --workers 16 && \
python scripts/relabel_index_clusters.py --index-dir data/index_t33_2026 \
--chain-clusters ${CASEFIX_DIR}/chain_clusters.tsv --out-dir ${CASEFIX_DIR}/index_t33_relabel && \
python scripts/retrieval_identity_audit.py --label 650m_trufor_s42_2026_casefix \
--index-dir ${CASEFIX_DIR}/index_t33_relabel --identity-scope both --control-per-sample-tsv '' \
--allow-score-mismatch --out-dir ${TEMP_DIR}/casefix_2026/identity_audit; } 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "08:00:00" 16 "128G" >/dev/null
}
run_casefix

# Cβ contact labels (CASP definition; decided 2026-11-05). Same mmCIF snapshot and
# chain filters as the 2026 build (min_len 20, max_len 1280, cutoff 8 Å, min_sep 6),
# only --contact_atom differs. The gate requires identical stems, seq, L, mask and
# Cα coords versus data/processed_2026, so the ESM caches and FAISS vectors stay valid.
CB_DIR="data/processed_2026cb"
run_cb_rebuild() {
    local key="cb-rebuild"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    if [[ "${DRY_RUN}" != "true" && -e "${PROJECT_DIR}/${CB_DIR}" ]]; then
        echo "SKIP ${key}: ${PROJECT_DIR}/${CB_DIR} already exists; move it away to rerun" >&2
        return
    fi
    local job_name="paper_cb_rebuild"
    local command="{ python scripts/build_contacts.py --contact_atom CB \
--mmcif_dir /mnt/storage_6/project_data/pl0735-01/old_pl0468-02/pdb_snapshot_2026/mmCIF \
--out_dir ${CB_DIR} --cutoff 8.0 --min_sep 6 --min_len 20 --max_len 1280 --num_workers 32 && \
python scripts/build_npz_lengths.py --processed_dir ${CB_DIR} && \
python scripts/compare_contact_generations.py --old-dir data/processed_2026 --new-dir ${CB_DIR} \
--out-dir ${TEMP_DIR}/cb_rebuild_gate --workers 32; } 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "08:00:00" 32 "128G" >/dev/null
}
run_cb_rebuild

# Cα-only chains after the Cβ switch (decided 2026-11-05: exclude like unclustered
# chains). Writes the skip list next to the corrected splits.
run_cb_low_list() {
    local key="cb-low-list"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local job_name="paper_cb_low_list"
    local command="python scripts/list_low_cb_chains.py --processed-dir ${CB_DIR} \
--out data/output_splits_2026cb/low_cb_ids.txt --threshold 0.5 --workers 16 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "02:00:00" 16 "64G" >/dev/null
}
run_cb_low_list

# RCSB can place identical sequences in different clusters, so a cluster-disjoint
# split need not be sequence-disjoint. Writes the evaluation skip list of chains that
# are the same protein as a training chain (decided 2026-11-05).
run_seq_overlap() {
    local key="seq-overlap"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local job_name="paper_seq_overlap"
    local command="{ python scripts/sequence_overlap_audit.py --processed-dir ${CB_DIR} \
--split-dir data/output_splits_2026cb --out-dir ${TEMP_DIR}/seq_overlap --workers 16 \
--out-skip-list data/output_splits_2026cb/same_protein_as_train_ids.txt \
--skip-ids-files data/output_splits_2026cb/no_cluster_ids.txt \
data/output_splits_2026cb/low_cb_ids.txt && \
python scripts/verify_data_integrity.py --check-embeddings; } 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "04:00:00" 16 "128G" >/dev/null
}
run_seq_overlap

# Finish the 2026cb generation. The ESM caches are keyed by chain and the sequences
# are unchanged, so 2026cb shares them through symlinks and only the chains the case
# fix brought back get embedded (the script skips existing outputs). The indexes are
# rebuilt so that labels, NPZ paths and rows all belong to 2026cb. The same-protein
# list is deliberately NOT passed to the index: those chains stay template memory.
run_cb_data() {
    local key="cb-data"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local split="data/output_splits_2026cb"
    local skip="${split}/no_cluster_ids.txt ${split}/low_cb_ids.txt"
    local pre="data/precomputed"
    local job_name="paper_cb_data"
    local command="{ { [ -e ${pre}/esm_t6_8M_2026cb ] || ln -s esm_t6_8M_2026 ${pre}/esm_t6_8M_2026cb; } && \
{ [ -e ${pre}/esm_t33_650M_2026cb ] || ln -s esm_t33_650M_2026 ${pre}/esm_t33_650M_2026cb; } && \
python scripts/precompute_esm2_embeddings.py --data_root ${CB_DIR} --output_dir ${pre}/esm_t6_8M_2026cb \
--model_name esm2_t6_8M_UR50D --batch_size 32 --skip_ids_files ${skip} && \
python scripts/precompute_esm2_embeddings.py --data_root ${CB_DIR} --output_dir ${pre}/esm_t33_650M_2026cb \
--model_name esm2_t33_650M_UR50D --batch_size 4 --skip_ids_files ${skip} && \
python scripts/build_index.py --processed_dir ${CB_DIR} --out_dir data/index_t6_2026cb --exclude_ids '' \
--chain_clusters_tsv ${split}/chain_clusters.tsv --skip_ids_files ${skip} \
--precomputed_embeddings_dir ${pre}/esm_t6_8M_2026cb && \
python scripts/build_index.py --processed_dir ${CB_DIR} --out_dir data/index_t33_2026cb --exclude_ids '' \
--chain_clusters_tsv ${split}/chain_clusters.tsv --skip_ids_files ${skip} \
--precomputed_embeddings_dir ${pre}/esm_t33_650M_2026cb && \
python scripts/verify_data_integrity.py --check-embeddings; } 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "12:00:00" 16 "128G" >/dev/null
}
run_cb_data

run_verify_clusters() {
    local key="verify-clusters"
    if [[ "${MODE}" != "only" || "${ONLY}" != "${key}" ]]; then
        return
    fi
    local job_name="paper_verify_clusters"
    local command="python scripts/verify_chain_clusters.py \
--chain-clusters data/output_splits_2026cb/chain_clusters.tsv --cluster-file data/clusters_30_2026.txt \
--out-dir ${TEMP_DIR}/verify_chain_clusters --workers 32 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "03:00:00" 32 "128G" >/dev/null
}
run_verify_clusters

run_pytest() {
    if [[ "${MODE}" != "only" || "${ONLY}" != "pytest" ]]; then
        return
    fi
    local job_name="paper_pytest_phase0"
    # paper/ is gitignored and not on the server, so its two generator tests run locally.
    local command="python -m pytest tests/phase0 -q --color=no \
--ignore=tests/phase0/test_paper_figure_generators.py \
--ignore=tests/phase0/test_qualitative_figure.py 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "01:00:00" 8 "64G" >/dev/null
}
run_pytest

run_selected "ceiling" "scripts/ceiling.py" "paper_ceiling_t33" "02:00:00"
run_selected "correlation" "scripts/feature_correlation.py" "paper_feature_corr_t33" "02:00:00"
