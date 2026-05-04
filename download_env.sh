#!/usr/bin/env bash
# Usage on a fresh worker machine:
#   1. scp nayan@app.arcadea.us:/srv/CHOROS_AUTO/download_env.sh .
#   2. scp nayan@app.arcadea.us:/srv/CHOROS_AUTO/secrets.sh .
#   3. bash download_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load SSH password from secrets.sh (gitignored) or environment
if [[ -f "${SCRIPT_DIR}/secrets.sh" ]]; then
    source "${SCRIPT_DIR}/secrets.sh"
fi
: "${SSH_PASS:?'SSH_PASS not set. Create secrets.sh or export SSH_PASS=... before running.'}"
export SSHPASS="${SSH_PASS}"

sudo apt-get install -y sshpass

mkdir -p /workspace/outputs

sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/outputs/hparam_search outputs/hparam_search
sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/pipeline pipeline
sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/training training
sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/src src
sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/utils utils
sshpass -v -e scp -r nayan@app.arcadea.us:/srv/CHOROS_AUTO/setup setup

FLASH_ATTN_WHEEL=/workspace/utils/flash_attn-2.8.4-cp311-cp311-linux_x86_64.whl bash setup/setup_env.sh
rm -r srv/CHOROS_AUTO

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
export CHOROS_DATA_ROOT=./srv/CHOROS/data

# Derive a stable, unique seed offset from this machine's hostname so no two
# workers explore the same TPE startup region.
WORKER_OFFSET=$(python3 -c "
import hashlib, socket
h = socket.gethostname().encode()
print(int(hashlib.sha256(h).hexdigest()[:4], 16) % 9000 + 1)
")
echo "Worker offset: ${WORKER_OFFSET}  (derived from hostname: $(hostname))"

python training/hparam_search.py --gpu 0 --n_trials 2000 --study_name baseline_v3 \
    --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
    --worker_offset "${WORKER_OFFSET}" \
    --skip_final