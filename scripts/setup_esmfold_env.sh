#!/bin/bash
# One-off setup for the ESMFold external reference. Run on the LOGIN node (it
# needs internet; the GPU job then runs offline):
#   bash scripts/setup_esmfold_env.sh
#
# - A venv on top of the structfuse interpreter (--system-site-packages), so torch
#   and the project's packages are shared and `conda/envs/structfuse` is NOT modified.
# - transformers installed into that venv only.
# - facebook/esmfold_v1 downloaded into the HF_HOME the SLURM jobs use
#   (${PROJECT_DIR}/models, see _launch_common.sh).
# The venv lives outside the repository so the launcher's clean-worktree check
# is unaffected.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/storage_3/home/nszostak/pl0735-01/project_data/old_pl0468-02/StructFuse}"
MAMBA_ENV="${MAMBA_ENV:-/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/conda/envs/structfuse}"
ESMFOLD_VENV="${ESMFOLD_VENV:-/mnt/storage_6/project_data/pl0735-01/old_pl0468-02/venvs/esmfold}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.1}"
MODEL_ID="${MODEL_ID:-facebook/esmfold_v1}"

if [[ ! -x "${ESMFOLD_VENV}/bin/python" ]]; then
    "${MAMBA_ENV}/bin/python" -m venv --system-site-packages "${ESMFOLD_VENV}"
fi
"${ESMFOLD_VENV}/bin/python" -m pip install "${TRANSFORMERS_SPEC}"

HF_HOME="${PROJECT_DIR}/models" "${ESMFOLD_VENV}/bin/python" - "${MODEL_ID}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print("weights:", snapshot_download(sys.argv[1]))
EOF

"${ESMFOLD_VENV}/bin/python" - <<'EOF'
import torch, transformers
from transformers import EsmForProteinFolding  # noqa: F401  (import check only)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| ESMFold class OK")
EOF
echo "ESMFold venv ready: ${ESMFOLD_VENV}"
