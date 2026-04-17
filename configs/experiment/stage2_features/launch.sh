#!/bin/bash
# ============================================================================
# Stage 2 — cheap features z template Cα coords. Ablation matrix + frontier.
# Green light (all.yaml): P@L_long ep10 ≥ 0.582 (+1pp vs baseline).
#
# Usage:
#   ./configs/experiment/stage2_features/launch.sh              # submit all jobs
#   ./configs/experiment/stage2_features/launch.sh --dry-run    # print sbatch commands
# ============================================================================

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PROJECT_DIR="/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse"
LOGS_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Seeds for multi-seed experiments
SEEDS=(42 0 1337)

# --------------------------------------------------------------------------
# submit <experiment_config> <job_name> <seed> [time_limit]
#   experiment_config : Hydra experiment path (e.g. main/structfuse)
#   job_name          : SLURM job name (also used in log filenames)
#   seed              : random seed (passed as Hydra override)
#   time_limit        : optional, default 12:00:00
# --------------------------------------------------------------------------
submit() {
    local experiment="$1"
    local job_name="$2"
    local seed="$3"
    local time_limit="${4:-12:00:00}"

    local cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=${experiment} seed=${seed}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch  job=${job_name}  time=${time_limit}  experiment=${experiment}  seed=${seed}"
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
        --cpus-per-task=32 \
        --mem=512G \
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

    echo "Submitted: ${job_name}  (experiment=${experiment}, time=${time_limit})"
}

echo "================================================"
echo " Stage 2 — Template-coord features (ablations + frontier)"
echo " dry-run: ${DRY_RUN}"
echo "================================================"

submit "stage2_features/dist_only"  "s2_dist_s42"  42 "12:00:00"
submit "stage2_features/agree_only" "s2_agree_s42" 42 "12:00:00"
submit "stage2_features/stats_only" "s2_stats_s42" 42 "12:00:00"
submit "stage2_features/all"        "s2_all_s42"   42 "12:00:00"

echo "================================================"
echo " Total: 4 jobs"
echo "================================================"
