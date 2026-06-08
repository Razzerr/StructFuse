#!/bin/bash
# LEGACY: retained for historical k=4/k=8 reproduction only.
# Final paper runs use:
#   ./configs/experiment/launch_paper_8M.sh --supplementary
# ============================================================================
# Faza 0.6 — 8M k-sweep early: k=4 (frontier_8M) vs k=8.
# Two comparative training runs on the 8M backbone — feeds the decision on
# whether to bump frontier.yaml data.topk above 4 before launching the
# 3-seed 650M headline (H1).
#
# Decision rule (plan Faza 0.6): if test/P@L_long(k=8) - test/P@L_long(k=4)
# > 0.5pp on 8M → update frontier.yaml data.topk=8 before H1.
#
# Usage:
#   ./configs/experiment/launch_k_sweep_8M.sh              # submit both
#   ./configs/experiment/launch_k_sweep_8M.sh --dry-run    # print only
# ============================================================================

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PROJECT_DIR="/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse"
LOGS_DIR="${PROJECT_DIR}/logs"
TEMP_DIR="${PROJECT_DIR}/.temp"
$DRY_RUN || mkdir -p "${LOGS_DIR}" "${TEMP_DIR}"

# --------------------------------------------------------------------------
# submit <experiment_args> <job_name> [time_limit]
#   experiment_args : everything passed to src/train.py (composed Hydra overrides)
#   job_name        : SLURM job name + W&B task_name + tee log filename
#   time_limit      : optional, default 24:00:00
# --------------------------------------------------------------------------
submit() {
    local experiment_args="$1"
    local job_name="$2"
    local time_limit="${3:-24:00:00}"

    local temp_log="${TEMP_DIR}/${job_name}.log"
    local cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py ${experiment_args} test=true task_name=${job_name} 2>&1 | tee ${temp_log}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch  job=${job_name}  time=${time_limit}  args=${experiment_args}"
        return
    fi

    sbatch \
        --job-name="${job_name}" \
        --output="${LOGS_DIR}/slurm_${job_name}_%j.out" \
        --error="${LOGS_DIR}/slurm_${job_name}_%j.err" \
        --partition=proxima \
        --time="${time_limit}" \
        --ntasks=1 \
        --gpus-per-node=1 \
        --cpus-per-task=16 \
        --mem=256G \
        --nodes=1 \
        --wrap="$(cat <<EOF
#!/bin/bash
set -eo pipefail
cd "${PROJECT_DIR}"

# ── Environment setup ──
eval "\$(/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/micromamba/micromamba shell hook --shell bash)"
micromamba activate /mnt/storage_6/project_data/pl0735-01/old_pl0468-02/conda/envs/structfuse
export HF_HOME="${PROJECT_DIR}/models"
export TORCH_HOME="${PROJECT_DIR}/models"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_\${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.inductor_cache_\${SLURM_JOB_ID}"
export WANDB_DIR="${PROJECT_DIR}/logs"
export WANDB_CONFIG_DIR="${PROJECT_DIR}/.wandb"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.wandb/cache"
export NETRC="${PROJECT_DIR}/.wandb/.netrc"
export PROJECT_ROOT="${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

# ── Run ──
${cmd}
EOF
)"

    echo "Submitted: ${job_name}  (args=${experiment_args}, time=${time_limit})"
}

echo "================================================"
echo " Faza 0.6 — 8M k-sweep early: k=4 vs k=8"
echo " dry-run: ${DRY_RUN}"
echo "================================================"

# k=4: frontier_8M (data.topk=4 inherited from data/contact.yaml)
submit "experiment=frontier_8M"                          "k_sweep_8M_k4"  "24:00:00"

# k=8: same frontier_8M, override both data.topk (the real switch) and
# model.topk (audit-log consistency for fallback _prior_builder). experiment=
# is a single-item Hydra group, can't stack via +experiment=.
submit "experiment=frontier_8M data.topk=8 model.topk=8" "k_sweep_8M_k8"  "24:00:00"

echo "================================================"
echo " Total: 2 jobs"
echo " Results: W&B + .temp/audit/<run_id>/manifest.json + .temp/<job_name>.log"
echo "================================================"
