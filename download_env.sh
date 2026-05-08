#!/usr/bin/env bash
# Usage on a fresh worker machine:
#   Default — git clone + pip install flash-attn from PyPI:
#     curl -fsSL https://raw.githubusercontent.com/NNChawla/CHOROS_AUTO/main/download_env.sh | bash
#   With a pre-built wheel instead of PyPI:
#     FLASH_ATTN_WHEEL=/path/to/flash_attn-*.whl bash download_env.sh
#   Legacy scp mode (requires secrets.sh or SSH_PASS env var):
#     scp nayan@app.arcadea.us:/srv/CHOROS_AUTO/download_env.sh .
#     scp nayan@app.arcadea.us:/srv/CHOROS_AUTO/secrets.sh .
#     USE_SCP=1 bash download_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/secrets.sh" ]]; then
    source "${SCRIPT_DIR}/secrets.sh"
fi

GITHUB_REPO="${GITHUB_REPO:-https://github.com/NNChawla/CHOROS_AUTO.git}"
REPO_DIR="${REPO_DIR:-/workspace/CHOROS_AUTO}"
USE_SCP="${USE_SCP:-0}"
SSH_HOST="nayan@app.arcadea.us"
SSH_SRC="/srv/CHOROS_AUTO"

mkdir -p /workspace/outputs

if [[ "${USE_SCP}" == "1" ]]; then
    # ── SCP fallback (original behaviour) ──────────────────────────────────
    : "${SSH_PASS:?'SSH_PASS not set. Create secrets.sh or export SSH_PASS=... before running.'}"
    export SSHPASS="${SSH_PASS}"
    sudo apt-get install -y sshpass

    sshpass -v -e scp -r "${SSH_HOST}:${SSH_SRC}/outputs/hparam_search" outputs/hparam_search
    for dir in pipeline training src setup; do
        sshpass -v -e scp -r "${SSH_HOST}:${SSH_SRC}/${dir}" "${dir}"
    done

    # Grab the pre-built wheel from the server unless one was already provided
    if [[ -z "${FLASH_ATTN_WHEEL:-}" ]]; then
        sshpass -v -e scp -r "${SSH_HOST}:${SSH_SRC}/utils" utils
        FLASH_ATTN_WHEEL=$(ls utils/flash_attn-*.whl 2>/dev/null | head -1 || true)
    fi
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}" bash setup/setup_env.sh
else
    # ── Git-based (default) ─────────────────────────────────────────────────
    if [[ -d "${REPO_DIR}/.git" ]]; then
        echo "[INFO] Repo already exists at ${REPO_DIR} — pulling latest"
        git -C "${REPO_DIR}" pull --ff-only
    else
        git clone "${GITHUB_REPO}" "${REPO_DIR}"
    fi
    # FLASH_ATTN_WHEEL can still be forwarded to use a local wheel instead of PyPI
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}" bash "${REPO_DIR}/setup/setup_env.sh"
fi

rm -r srv/CHOROS_AUTO

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
export CHOROS_DATA_ROOT=./srv/CHOROS/data

python setup/pack_dataset.py ./srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir ./srv/CHOROS/data/kinematics

# 
# # Derive a stable, unique seed offset from this machine's hostname so no two
# # workers explore the same TPE startup region.
# WORKER_OFFSET=$(python3 -c "
# import hashlib, socket
# h = socket.gethostname().encode()
# print(int(hashlib.sha256(h).hexdigest()[:4], 16) % 9000 + 1)
# ")
# echo "Worker offset: ${WORKER_OFFSET}  (derived from hostname: $(hostname))"
# 
# python training/hparam_search.py --gpu 0 --n_trials 2000 --study_name baseline_v3 \
#     --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
#     --worker_offset "${WORKER_OFFSET}" \
#     --skip_final