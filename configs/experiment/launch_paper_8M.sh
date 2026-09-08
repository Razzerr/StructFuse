#!/bin/bash
# Final 8M mechanism/ablation runs.
#
# Usage:
#   ./configs/experiment/launch_paper_8M.sh --gate          # 2 jobs, run first
#   ./configs/experiment/launch_paper_8M.sh --final-eval    # bs=1 reported numbers, after --core
#   ./configs/experiment/launch_paper_8M.sh --trufor-controls  # 2 jobs, headline controls
#   ./configs/experiment/launch_paper_8M.sh --smoke
#   ./configs/experiment/launch_paper_8M.sh --core
#   ./configs/experiment/launch_paper_8M.sh --supplementary
#   ./configs/experiment/launch_paper_8M.sh --all
# Add --skip-frontier when the canonical frontier_8M run already exists.
# Add --dry-run to print the launch matrix without submitting jobs.

set -euo pipefail

DRY_RUN=false
# Anything before this belongs to a previous data generation. Override with
# --min-ckpt-date if the panel is ever re-run on another rebuild.
MIN_CKPT_DATE="${MIN_CKPT_DATE:-2026-08-22}"
SKIP_FRONTIER=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gate|--smoke|--core|--trufor-controls|--supplementary|--final-eval|--all)
            if [[ -n "${MODE}" ]]; then
                echo "Choose exactly one launch mode." >&2
                exit 2
            fi
            MODE="${1#--}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --min-ckpt-date)
            MIN_CKPT_DATE="$2"
            shift 2
            ;;
        --skip-frontier)
            SKIP_FRONTIER=true
            shift
            ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --smoke, --core, --supplementary, or --all." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_training() {
    local experiment="$1"
    local job_name="$2"
    local time_limit="${3:-24:00:00}"
    local extra_args="${4:-}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=${experiment} test=true task_name=${job_name} ${extra_args} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 16 "256G" >/dev/null
}

# Final reported numbers come from an eval-only pass at eval_batch_size=1: the
# model is padding-dependent, so batching makes runs with different dataset
# composition incomparable (measured: bs=1 is bit-identical across compositions,
# bs=12 differs on 18.8% of chains). validate_before_test=true so the F1
# threshold is calibrated under the policy it is applied in. Measured cost:
# ~22 min test + ~13 min validate per cell.
submit_final_eval() {
    local experiment="$1"
    local train_task="$2"
    # logs/ still holds runs from the 2025 data generation, and "newest .ckpt"
    # would silently pick one of those for any cell not yet re-run. Refuse
    # anything older than the rebuild.
    local ckpt
    ckpt="$(find "${REPO_ROOT:-.}/logs/${train_task}" -name '*.ckpt' ! -name 'last.ckpt' \
            -newermt "${MIN_CKPT_DATE}" -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2-)"
    if [[ -z "${ckpt}" ]]; then
        local stale
        stale="$(find "${REPO_ROOT:-.}/logs/${train_task}" -name '*.ckpt' ! -name 'last.ckpt' \
                 2>/dev/null | head -1)"
        if [[ -n "${stale}" ]]; then
            echo "SKIP ${train_task}: only checkpoints older than ${MIN_CKPT_DATE} exist" \
                 "(e.g. ${stale}) — that is a previous data generation, not this panel." >&2
        else
            echo "SKIP ${train_task}: no checkpoint found (run not finished?)" >&2
        fi
        return
    fi
    local job_name="${train_task}_bs1"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/eval.py \
experiment=${experiment} task_name=${job_name} ckpt_path=${ckpt} \
validate_before_test=true data.eval_batch_size=1 logger=wandb 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "04:00:00" 16 "256G" >/dev/null
}

# Mirrors submit_core cell for cell; run after --core has finished.
submit_final_evals() {
    submit_final_eval "frontier_8M"                  "paper_8m_frontier_k4"
    submit_final_eval "ablation/tpl_contact_only"    "paper_8m_stage1_tpl_contact"
    submit_final_eval "ablation/no_triangle"         "paper_8m_stage2_no_triangle"
    submit_final_eval "ablation/no_dist"             "paper_8m_no_dist"
    submit_final_eval "ablation/no_templates"        "paper_8m_no_templates"
    submit_final_eval "ablation/random_retrieval"    "paper_8m_random_retrieval"
    submit_final_eval "ablation/bce_only"            "paper_8m_bce_only"
    # Submitted by launch_trufor_ablation_8M.sh --fusion-decision, but its reported
    # numbers must come from the SAME bs=1 pass as everything it is compared
    # against. Running it separately by hand is how cap_full_no_templates ended up
    # mixing protocols; the stage-E comparison would inherit that.
    submit_final_eval "ablation/trufor_fusion_with_dist" "paper_8m_trufor_full_k4"
    submit_final_eval "ablation/trufor_no_templates"  "paper_8m_trufor_no_templates"
    submit_final_eval "ablation/trufor_no_dist"       "paper_8m_trufor_no_dist"
    # B3 is attention-only: no trained weights, so no ckpt_path. It runs through
    # train.py in the core panel and does the same here, just at bs=1.
    local b3="paper_8m_esm2_raw_bs1"
    submit_job "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py \
experiment=baseline/esm2_only test=true task_name=${b3} data.eval_batch_size=1 \
2>&1 | tee ${TEMP_DIR}/${b3}.log" "${b3}" "08:00:00" 16 "256G" >/dev/null
}

submit_smoke() {
    local repro_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/smoke_repro_test.py 2>&1 | tee ${TEMP_DIR}/paper_smoke_repro.log"
    local repro_id
    repro_id="$(submit_job "${repro_cmd}" "paper_smoke_repro" "02:00:00" 16 "256G")"

    local eval_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=frontier_8M task_name=paper_smoke_eval test=true ++trainer.max_epochs=1 ++trainer.limit_train_batches=0.01 ++trainer.limit_val_batches=10 ++trainer.limit_test_batches=10 data.num_workers=0 2>&1 | tee ${TEMP_DIR}/paper_smoke_eval.log"
    local eval_id
    eval_id="$(submit_job "${eval_cmd}" "paper_smoke_eval" "03:00:00" 16 "256G" "afterok:${repro_id}")"

    # Separate eval-only canary: no preceding fit(), so this exercises
    # DataModule.setup("validate") and val-threshold calibration for B1/B3.
    local eval_only_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=baseline/esm2_only task_name=paper_smoke_eval_only test=true ++trainer.limit_val_batches=2 ++trainer.limit_test_batches=2 data.num_workers=0 2>&1 | tee ${TEMP_DIR}/paper_smoke_eval_only.log"
    submit_job "${eval_only_cmd}" "paper_smoke_eval_only" "02:00:00" 16 "256G" "afterok:${eval_id}" >/dev/null
}

submit_core() {
    if [[ "${SKIP_FRONTIER}" == "true" ]]; then
        echo "Skipping paper_8m_frontier_k4 (--skip-frontier); using completed run iej9a561." >&2
    else
        submit_training "frontier_8M" "paper_8m_frontier_k4"
    fi
    submit_training "ablation/tpl_contact_only" "paper_8m_stage1_tpl_contact"
    submit_training "ablation/no_triangle" "paper_8m_stage2_no_triangle"
    submit_training "ablation/no_dist" "paper_8m_no_dist"
    submit_training "ablation/no_templates" "paper_8m_no_templates"
    submit_training "ablation/random_retrieval" "paper_8m_random_retrieval"
    submit_training "ablation/bce_only" "paper_8m_bce_only"
    submit_training "baseline/esm2_only" "paper_8m_esm2_raw" "08:00:00"
}

# Paired validation gate for a rebuilt data generation: real retrieval vs no
# templates, nothing else. Run this BEFORE --core so a broken rebuild costs two
# jobs instead of eight. Reports the absolute shift against the previous
# generation and the recomputed retrieval delta.
submit_gate() {
    submit_training "frontier_8M" "paper_8m_gate_frontier_k4"
    submit_training "ablation/no_templates" "paper_8m_gate_no_templates"
}

# The two controls the headline stack needs after Stage E selected TruFor+dist:
# a retrieval kill-switch and the (-dist,+triangle) cell, both on TruFor. The
# grouped panel stays as its own mechanism analysis and is NOT an ablation of
# TruFor; nothing here re-runs it.
submit_trufor_controls() {
    submit_training "ablation/trufor_no_templates" "paper_8m_trufor_no_templates"
    submit_training "ablation/trufor_no_dist"      "paper_8m_trufor_no_dist"
}

# NOTE: --supplementary is a MIXED bag, not a grouped panel and not a TruFor one:
# standard-concat fusion, the historical dist-less TruFor (A2, confounded), a
# dilated head and the k-sweep. After Stage E it is not a ready panel for the
# selected stack. Running it would only support claims — K-robustness, head
# comparison — that the manuscript has to actually make, on TruFor.
submit_supplementary() {
    submit_training "ablation/standard_fusion" "paper_8m_standard_fusion"
    submit_training "ablation/trufor_fusion" "paper_8m_trufor_fusion"
    submit_training "ablation/dilated_head" "paper_8m_dilated_head"
    submit_training "ablation/k1_templates" "paper_8m_k1"
    submit_training "ablation/k8_templates" "paper_8m_k8"
    submit_training "ablation/k16_templates" "paper_8m_k16"
}

echo "8M paper launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}, skip_frontier=${SKIP_FRONTIER}"
case "${MODE}" in
    gate)
        submit_gate
        ;;
    trufor-controls)
        submit_trufor_controls
        ;;
    final-eval)
        submit_final_evals
        ;;
    smoke)
        submit_smoke
        ;;
    core)
        submit_core
        ;;
    supplementary)
        submit_supplementary
        ;;
    all)
        submit_core
        submit_supplementary
        ;;
esac
