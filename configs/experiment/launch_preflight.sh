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
# cost* and export-maps* are opt-in and never part of --all.

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
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --all or --only {verify|coverage|coverage8m|ceiling|correlation|cost-smoke|cost|export-maps|export-maps-notpl}." >&2
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
        verify|coverage|coverage8m|ceiling|correlation|cost|cost-smoke|export-maps|export-maps-notpl) ;;
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

run_selected "ceiling" "scripts/ceiling.py" "paper_ceiling_t33" "02:00:00"
run_selected "correlation" "scripts/feature_correlation.py" "paper_feature_corr_t33" "02:00:00"
