#!/bin/bash
# ============================================================================
# Stage 2 diagnosis — non-training ceiling + feature correlation analysis.
#   ceiling            : scripts/ceiling.py  → .temp/ceiling_results.tsv
#   feature_correlation: scripts/feature_correlation.py  → .temp/feature_correlation.tsv
#
# Both scripts are inference-only (no backbone forward, no training). They
# iterate the val loader, flatten pair-level features, fit linear models /
# compute MI. Short wall time; no GPU compute needed, but we keep 1 GPU for
# consistency with the training environment (CUDA tensors in data pipeline).
#
# Usage:
#   ./configs/experiment/diagnostics/launch_ceiling.sh              # submit both
#   ./configs/experiment/diagnostics/launch_ceiling.sh --dry-run    # print only
# ============================================================================

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PROJECT_DIR="/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse"
LOGS_DIR="${PROJECT_DIR}/logs"
TEMP_DIR="${PROJECT_DIR}/.temp"
mkdir -p "${LOGS_DIR}" "${TEMP_DIR}"

# --------------------------------------------------------------------------
# submit <script_path> <job_name> [time_limit]
#   script_path : path to the standalone python script (relative to repo root)
#   job_name    : SLURM job name (also used in log filenames)
#   time_limit  : optional, default 01:00:00
# --------------------------------------------------------------------------
submit() {
    local script="$1"
    local job_name="$2"
    local time_limit="${3:-01:00:00}"

    # Predictable tee path in .temp so we don't have to hunt slurm logs.
    local temp_log="${TEMP_DIR}/${job_name}.log"
    local cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ${script} experiment=diagnostics/ceiling 2>&1 | tee ${temp_log}"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch  job=${job_name}  time=${time_limit}  script=${script}"
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

    echo "Submitted: ${job_name}  (script=${script}, time=${time_limit})"
}

echo "================================================"
echo " Stage 2 diagnosis — ceiling + correlation"
echo " dry-run: ${DRY_RUN}"
echo "================================================"

submit "scripts/ceiling.py"              "ceiling_stage2"         "01:00:00"
submit "scripts/feature_correlation.py"  "feat_corr_stage2"       "00:30:00"
submit "scripts/template_coverage.py"    "tpl_coverage"           "00:20:00"

echo "================================================"
echo " Total: 2 jobs"
echo " Results: .temp/ceiling_results.tsv, .temp/feature_correlation.tsv"
echo " Full stdout+stderr: .temp/ceiling_stage2.log, .temp/feat_corr_stage2.log"
echo "================================================"
