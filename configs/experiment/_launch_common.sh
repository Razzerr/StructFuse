#!/bin/bash

# Shared SLURM plumbing for the paper launchers. Source this file; do not run it.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse}"
MAMBA_BIN="${MAMBA_BIN:-/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/micromamba/micromamba}"
MAMBA_ENV="${MAMBA_ENV:-/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/conda/envs/structfuse}"
SLURM_PARTITION="${SLURM_PARTITION:-proxima}"

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${LAUNCHER_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_DIR}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    SOURCE_ROOT="${LOCAL_REPO_ROOT}"
fi

LOGS_DIR="${PROJECT_DIR}/logs"
TEMP_DIR="${PROJECT_DIR}/.temp"
LAUNCH_MANIFEST_DIR="${TEMP_DIR}/launch_manifests"

git_commit() {
    git -C "${SOURCE_ROOT}" rev-parse HEAD
}

worktree_is_dirty() {
    [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=normal)" ]]
}

require_clean_worktree() {
    if worktree_is_dirty; then
        if [[ "${DRY_RUN:-false}" == "true" ]]; then
            echo "WARNING: worktree is dirty; real paper runs would be refused." >&2
            return
        fi
        echo "ERROR: refusing to launch paper runs from a dirty worktree." >&2
        echo "Commit or stash all tracked and untracked changes first." >&2
        exit 2
    fi
}

prepare_output_dirs() {
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        mkdir -p "${LOGS_DIR}" "${TEMP_DIR}" "${LAUNCH_MANIFEST_DIR}"
    fi
}

record_submission() {
    local job_id="$1"
    local job_name="$2"
    local command="$3"
    local commit="$4"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        return
    fi

    local manifest="${LAUNCH_MANIFEST_DIR}/$(date -u +%Y%m%dT%H%M%SZ)_${job_name}.tsv"
    {
        printf "job_id\tjob_name\tcommit\tcommand\n"
        printf "%s\t%s\t%s\t%s\n" "${job_id}" "${job_name}" "${commit}" "${command}"
    } > "${manifest}"
}

# submit_job <command> <job_name> <time> <cpus> <memory> [dependency]
#
# Prints only the SLURM job ID to stdout. Human-readable status goes to stderr,
# allowing callers to capture the ID and build afterok dependencies.
submit_job() {
    local command="$1"
    local job_name="$2"
    local time_limit="$3"
    local cpus="$4"
    local memory="$5"
    local dependency="${6:-}"
    local commit
    commit="$(git_commit)"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY-RUN] job=${job_name} time=${time_limit} cpus=${cpus} mem=${memory} dependency=${dependency:-none}" >&2
        echo "[DRY-RUN] commit=${commit}" >&2
        echo "[DRY-RUN] command=${command}" >&2
        printf "dryrun-%s\n" "${job_name}"
        return
    fi

    local dependency_args=()
    if [[ -n "${dependency}" ]]; then
        dependency_args=(--dependency="${dependency}")
    fi

    local job_id
    job_id="$(
        sbatch --parsable \
            --job-name="${job_name}" \
            --output="${LOGS_DIR}/slurm_${job_name}_%j.out" \
            --error="${LOGS_DIR}/slurm_${job_name}_%j.err" \
            --partition="${SLURM_PARTITION}" \
            --time="${time_limit}" \
            --ntasks=1 \
            --gpus-per-node=1 \
            --cpus-per-task="${cpus}" \
            --mem="${memory}" \
            --nodes=1 \
            "${dependency_args[@]}" \
            --wrap="$(cat <<EOF
#!/bin/bash
set -eo pipefail
cd "${PROJECT_DIR}"

eval "\$(${MAMBA_BIN} shell hook --shell bash)"
micromamba activate "${MAMBA_ENV}"

export HF_HOME="${PROJECT_DIR}/models"
export TORCH_HOME="${PROJECT_DIR}/models"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_\${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.inductor_cache_\${SLURM_JOB_ID}"
export WANDB_DIR="${PROJECT_DIR}/logs"
export WANDB_CONFIG_DIR="${PROJECT_DIR}/.wandb"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.wandb/cache"
export NETRC="${PROJECT_DIR}/.wandb/.netrc"
export PROJECT_ROOT="${PROJECT_DIR}"
export STRUCTFUSE_GIT_COMMIT="${commit}"
export PYTHONUNBUFFERED=1

echo "StructFuse commit: ${commit}"
echo "Command: ${command}"
${command}
EOF
)"
    )"
    job_id="${job_id%%;*}"

    record_submission "${job_id}" "${job_name}" "${command}" "${commit}"
    echo "Submitted ${job_name}: job_id=${job_id}, commit=${commit}" >&2
    printf "%s\n" "${job_id}"
}
