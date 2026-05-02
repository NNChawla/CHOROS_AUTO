#!/usr/bin/env bash
# setup_env.sh — create and populate the CHOROS conda environment,
# then build + install flash-attention from source (FA3-capable build).
#
# Usage:
#   bash setup_env.sh                        # create env, build flash-attn from source
#   bash setup_env.sh --check                # run environment checks only, no changes
#   bash setup_env.sh --coordinator          # also set up PostgreSQL Optuna DB server
#   bash setup_env.sh --export-wheel DIR     # build flash-attn, save .whl to DIR, then install
#   ENV_NAME=MY_ENV bash setup_env.sh
#
# To skip the source build and install a pre-built wheel:
#   FLASH_ATTN_WHEEL=/path/to/flash_attn-*.whl bash setup_env.sh
#   FLASH_ATTN_WHEEL=https://host/flash_attn-*.whl bash setup_env.sh
#
# Wheel compatibility: a wheel built here is reusable on any server that has
# the same Python version (3.11), CUDA version (12.8), and GPU arch (sm120
# for RTX 5090).  Different GPU arch → rebuild needed.
#
# Distributed hyperparameter search:
#   Run --coordinator once on the host that will serve the Optuna PostgreSQL DB
#   (app.arcadea.us).  All other machines only need psycopg2-binary (installed
#   automatically by install_packages) to connect as workers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-CHOROS}"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.7.0"
CUDA_LABEL="cu128"          # matches CUDA 12.8
FLASH_ATTN_REPO="https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTN_DIR="${TMPDIR:-/tmp}/flash-attention-build"
# flash-attn CUDA kernel compilation uses ~25 GB RAM per parallel job.
# Cap MAX_JOBS to whichever is smaller: half the cores, or RAM/25.
_HALF_CORES=$(( $(nproc) / 2 ))
_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
_RAM_JOBS=$(( _RAM_GB / 25 ))
MAX_JOBS=$(( _HALF_CORES < _RAM_JOBS ? _HALF_CORES : _RAM_JOBS ))
MAX_JOBS=$(( MAX_JOBS < 1 ? 1 : MAX_JOBS ))

# Pre-built wheel: set to a local path or URL to skip the source build.
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
# Directory to save the built wheel into (--export-wheel DIR).
FLASH_ATTN_WHEEL_OUT=""

CHECK_ONLY=0
COORDINATOR=0
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --check)        CHECK_ONLY=1 ;;
        --coordinator)  COORDINATOR=1 ;;
        --export-wheel) shift; FLASH_ATTN_WHEEL_OUT="${1:?'--export-wheel requires a directory'}" ;;
        *)              die "Unknown argument: ${1}" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

check_pass() { echo -e "  ${GREEN}✓${NC}  $*"; }
check_fail() { echo -e "  ${RED}✗${NC}  $*"; }
check_warn() { echo -e "  ${YELLOW}!${NC}  $*"; }

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
run_checks() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  CHOROS Environment Checks"
    echo "════════════════════════════════════════════════════════"
    CHECKS_PASSED=1

    # OS
    if [[ -f /etc/os-release ]]; then
        OS_NAME=$(. /etc/os-release && echo "$PRETTY_NAME")
        check_pass "OS: $OS_NAME"
    else
        check_warn "Could not identify OS"
    fi

    # RAM
    TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    if (( TOTAL_RAM_GB >= 16 )); then
        check_pass "RAM: ${TOTAL_RAM_GB} GB"
    else
        check_warn "RAM: ${TOTAL_RAM_GB} GB (16 GB+ recommended for flash-attn build)"
    fi

    # Disk space on /tmp (flash-attn build can use ~10 GB)
    TMP_FREE_GB=$(df -BG "${TMPDIR:-/tmp}" | awk 'NR==2 {gsub("G",""); print $4}')
    if (( TMP_FREE_GB >= 15 )); then
        check_pass "Disk (${TMPDIR:-/tmp}): ${TMP_FREE_GB} GB free"
    else
        check_warn "Disk (${TMPDIR:-/tmp}): ${TMP_FREE_GB} GB free (15 GB+ recommended)"
    fi

    # CPU cores
    CORES=$(nproc)
    check_pass "CPU: ${CORES} cores  (will use MAX_JOBS=${MAX_JOBS} for build)"

    # git
    if command -v git &>/dev/null; then
        GIT_VER=$(git --version)
        check_pass "git: $GIT_VER"
    else
        check_fail "git not found — required to clone flash-attention"
        CHECKS_PASSED=0
    fi

    # conda / mamba
    if command -v conda &>/dev/null; then
        CONDA_VER=$(conda --version)
        check_pass "conda: $CONDA_VER"
    else
        check_fail "conda not found — install Miniconda first"
        CHECKS_PASSED=0
    fi

    # CUDA toolkit (nvcc)
    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',' || true)
        CUDA_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
        check_pass "nvcc (CUDA toolkit): $NVCC_VER"
        if (( CUDA_MAJOR < 12 )); then
            check_warn "CUDA < 12 detected — flash-attn 3 requires CUDA ≥ 12"
        fi
        if [[ "$NVCC_VER" != "$PYTORCH_VERSION"* ]]; then
            check_warn "nvcc ${NVCC_VER} may differ from PyTorch CUDA — the build step will install a matching toolkit into the conda env automatically"
        fi
    else
        check_warn "nvcc not found — the build step will install the CUDA toolkit into the conda env"
    fi

    # NVIDIA driver + GPUs
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
        if [[ -n "$GPU_INFO" ]]; then
            FA3_AVAILABLE=0
            while IFS=',' read -r gpu_name cc; do
                gpu_name=$(echo "$gpu_name" | xargs)
                cc=$(echo "$cc" | xargs)
                cc_major=$(echo "$cc" | cut -d. -f1)
                cc_minor=$(echo "$cc" | cut -d. -f2)
                sm="sm${cc//./}"
                if (( cc_major >= 10 )); then
                    check_pass "GPU: ${gpu_name}  (${sm} — Blackwell, FA3 fully supported)"
                    FA3_AVAILABLE=1
                elif (( cc_major == 9 )); then
                    check_pass "GPU: ${gpu_name}  (${sm} — Hopper, FA3 fully supported)"
                    FA3_AVAILABLE=1
                elif (( cc_major == 8 && cc_minor == 9 )); then
                    check_pass "GPU: ${gpu_name}  (${sm} — Ada Lovelace, FA2 supported; FA3 requires Hopper+)"
                elif (( cc_major >= 8 )); then
                    check_pass "GPU: ${gpu_name}  (${sm} — Ampere, FA2 supported; FA3 requires Hopper+)"
                else
                    check_warn "GPU: ${gpu_name}  (${sm} — older arch, flash-attn may not support)"
                fi
            done <<< "$GPU_INFO"
            if (( FA3_AVAILABLE == 0 )); then
                check_warn "No Hopper/Blackwell GPU (sm90+) detected — FA3 kernels will not be built"
            fi
        else
            check_fail "nvidia-smi found but no GPUs returned"
        fi
    else
        check_fail "nvidia-smi not found — NVIDIA driver not installed"
        CHECKS_PASSED=0
    fi

    # C++ compiler
    if command -v g++ &>/dev/null; then
        GXX_VER=$(g++ --version | head -1)
        check_pass "g++: $GXX_VER"
    else
        check_fail "g++ not found — required to compile flash-attention"
        CHECKS_PASSED=0
    fi

    # ninja (optional, speeds up build significantly)
    if command -v ninja &>/dev/null; then
        check_pass "ninja: $(ninja --version)"
    else
        check_warn "ninja not found — build will use make (slower); install with: pip install ninja"
    fi

    # Existing env
    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        check_warn "Conda env '${ENV_NAME}' already exists — packages will be added/updated"
    else
        check_pass "Conda env '${ENV_NAME}' does not exist — will create fresh"
    fi

    echo ""
    if (( CHECKS_PASSED == 0 )); then
        error "One or more required tools are missing. Fix the issues above and re-run."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Create conda environment
# ---------------------------------------------------------------------------
create_env() {
    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        info "Conda env '${ENV_NAME}' already exists — skipping creation"
    else
        info "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} ..."
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi
}

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
install_system_packages() {
    info "Installing system packages: pigz pv ..."
    sudo apt-get install -y pigz pv
}

# ---------------------------------------------------------------------------
# Install packages
# ---------------------------------------------------------------------------
install_packages() {
    info "Installing PyTorch ${PYTORCH_VERSION}+${CUDA_LABEL} ..."
    conda run -n "${ENV_NAME}" pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
        "torch==${PYTORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_LABEL}"

    info "Installing core scientific packages ..."
    conda run -n "${ENV_NAME}" pip install \
        numpy \
        scipy \
        scikit-learn \
        pandas \
        pyarrow \
        tqdm \
        joblib

    info "Installing training / search packages ..."
    conda run -n "${ENV_NAME}" pip install \
        optuna \
        psycopg2-binary \
        einops \
        numba

    info "Installing build tools for flash-attention ..."
    conda run -n "${ENV_NAME}" pip install \
        ninja \
        packaging \
        wheel \
        setuptools

    info "Installing huggingface_hub ..."
    conda run -n "${ENV_NAME}" pip install huggingface_hub
}

# ---------------------------------------------------------------------------
# Build flash-attention from source, optionally exporting a reusable wheel
# ---------------------------------------------------------------------------
_nvcc_mm() {
    # Print major.minor of the nvcc binary at $1, or empty if missing/unrunnable.
    local nvcc_bin="$1"
    [[ -x "${nvcc_bin}" ]] || return 0
    "${nvcc_bin}" --version 2>/dev/null \
        | grep "release" | awk '{print $5}' | cut -d. -f1,2 | tr -d ',' || true
}

_cuda_version_ok() {
    # Return 0 if the candidate CUDA_HOME has nvcc matching expected major.minor,
    # or has no nvcc at all (version unverifiable — headers-only install).
    local home="$1" expected_mm="$2"
    local nvcc_bin="${home}/bin/nvcc"
    [[ -x "${nvcc_bin}" ]] || return 0   # no nvcc here; accept headers-only
    local ver; ver=$(_nvcc_mm "${nvcc_bin}")
    [[ "${ver}" == "${expected_mm}" ]]
}

_fix_targets_layout() {
    # NVIDIA conda packages put headers under targets/x86_64-linux/include/ instead of include/.
    # Create symlinks so CUDA_HOME/include/cuda_runtime_api.h resolves correctly.
    local prefix="$1"
    local src="${prefix}/targets/x86_64-linux/include"
    [[ -d "${src}" ]] || return 0
    [[ -f "${prefix}/include/cuda_runtime_api.h" ]] && return 0
    info "Symlinking conda CUDA headers: ${src} → ${prefix}/include/ ..."
    mkdir -p "${prefix}/include" 2>/dev/null || return 0  # skip silently if no write access
    for item in "${src}"/*; do
        [[ -e "${item}" ]] || continue
        local base; base=$(basename "${item}")
        [[ -e "${prefix}/include/${base}" ]] || ln -sf "${item}" "${prefix}/include/${base}" 2>/dev/null || true
    done
    # Also symlink any subdirectories (e.g. cuda/std/, cub/)
    for item in "${src}"/*/; do
        [[ -d "${item}" ]] || continue
        local base; base=$(basename "${item}")
        [[ -e "${prefix}/include/${base}" ]] || ln -sf "${item%/}" "${prefix}/include/${base}" 2>/dev/null || true
    done
}

_has_cuda_header() {
    # Return 0 (success) if prefix has cuda_runtime_api.h reachable via include/.
    # Handles both standard layout and NVIDIA conda package layout (targets/).
    local prefix="$1"
    [[ -n "${prefix}" ]] || return 1
    [[ -f "${prefix}/include/cuda_runtime_api.h" ]] && return 0
    if [[ -f "${prefix}/targets/x86_64-linux/include/cuda_runtime_api.h" ]]; then
        _fix_targets_layout "${prefix}"
        # Verify the symlink was actually created (may fail if no write access)
        [[ -f "${prefix}/include/cuda_runtime_api.h" ]] || return 1
        return 0
    fi
    return 1
}

_resolve_cuda_home() {
    # Returns a directory whose include/cuda_runtime_api.h matches torch.version.cuda.
    # Strategy:
    #   1. Ask PyTorch's cpp_extension where it would look (most reliable).
    #   2. Search common system CUDA install paths (both standard and conda layouts).
    #   3. Check the conda env prefix (may differ from venv sys.prefix).
    #   4. Install the matching cuda-toolkit into the conda env and re-check.
    local torch_cuda_ver torch_mm env_prefix cuda_home conda_pfx candidate

    torch_cuda_ver=$(conda run -n "${ENV_NAME}" python -c \
        "import torch; print(torch.version.cuda)" 2>/dev/null || true)
    [[ -z "${torch_cuda_ver}" ]] && die "Cannot read torch.version.cuda — is torch installed?"
    info "PyTorch CUDA version: ${torch_cuda_ver}"
    torch_mm=$(echo "${torch_cuda_ver}" | cut -d. -f1,2)

    env_prefix=$(conda run -n "${ENV_NAME}" python -c \
        "import sys; print(sys.prefix)" 2>/dev/null || true)
    conda_pfx=$(conda env list 2>/dev/null \
        | grep -E "^${ENV_NAME}[[:space:]]" | awk '{print $NF}' || true)

    # 1. Ask torch itself — but validate the version, as it may point to the
    #    system CUDA even when a different version was used to compile torch.
    cuda_home=$(conda run -n "${ENV_NAME}" python -c \
        "from torch.utils import cpp_extension; h=cpp_extension.CUDA_HOME; print(h or '')" \
        2>/dev/null | tail -1 || true)
    if _has_cuda_header "${cuda_home}" && _cuda_version_ok "${cuda_home}" "${torch_mm}"; then
        info "CUDA_HOME (torch): ${cuda_home}"; echo "${cuda_home}"; return
    elif [[ -n "${cuda_home}" ]]; then
        warn "torch CUDA_HOME=${cuda_home} has wrong nvcc version (need ${torch_mm}) — searching further"
    fi

    # 2. Version-specific system paths, then conda env paths (preferred over generic
    #    /usr/local/cuda which may be a different version), then generic fallbacks.
    for candidate in \
            "/usr/local/cuda-${torch_cuda_ver}" \
            "/usr/local/cuda-${torch_mm}" \
            "${conda_pfx}" \
            "${env_prefix}" \
            "/usr/local/cuda" \
            "/usr/lib/cuda" \
            "/opt/cuda"; do
        if _has_cuda_header "${candidate}" && _cuda_version_ok "${candidate}" "${torch_mm}"; then
            info "CUDA_HOME (found): ${candidate}"; echo "${candidate}"; return
        fi
    done

    # 3. Install matching toolkit, then re-check.
    local nvcc_ver=""
    command -v nvcc &>/dev/null && \
        nvcc_ver=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',' || true)
    warn "cuda_runtime_api.h not found (system nvcc: ${nvcc_ver:-none}, torch CUDA: ${torch_cuda_ver})"
    info "Installing CUDA ${torch_mm} toolkit into conda env '${ENV_NAME}'..."
    conda install -y -n "${ENV_NAME}" \
        -c "nvidia/label/cuda-${torch_mm}.0" \
        cuda-toolkit >/dev/null 2>&1 \
    || conda install -y -n "${ENV_NAME}" \
        -c nvidia -c conda-forge \
        "cuda-toolkit=${torch_mm}.*" >/dev/null 2>&1

    for candidate in "${env_prefix}" "${conda_pfx}"; do
        if _has_cuda_header "${candidate}"; then
            info "CUDA_HOME (after install): ${candidate}"; echo "${candidate}"; return
        fi
    done

    die "Cannot find cuda_runtime_api.h for CUDA ${torch_mm}. \
Install the CUDA ${torch_mm} toolkit manually."
}

_detect_arch_flags() {
    local flags=""
    if command -v nvidia-smi &>/dev/null; then
        while IFS=',' read -r _name cc; do
            cc=$(echo "$cc" | xargs)
            local cc_nodot="${cc//./}"
            case "$cc_nodot" in
                75)   flags="${flags};7.5"   ;;
                80)   flags="${flags};8.0"   ;;
                86)   flags="${flags};8.6"   ;;
                89)   flags="${flags};8.9"   ;;
                90)   flags="${flags};9.0"   ;;
                90a)  flags="${flags};9.0a"  ;;
                100)  flags="${flags};10.0"  ;;
                100a) flags="${flags};10.0a" ;;
                120)  flags="${flags};12.0"  ;;
                120a) flags="${flags};12.0a" ;;
            esac
        done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)
    fi
    echo "${flags#;}"
}

# ---------------------------------------------------------------------------
# Show meaningful build errors from the log (compiler errors, not generic tail)
# ---------------------------------------------------------------------------
_show_build_error() {
    local log_file="$1"
    # Look for actual compiler errors first: "file:line: error:" or "fatal error:"
    # Skip pip packaging lines like "error: subprocess-exited-with-error"
    local first_err_line
    first_err_line=$(grep -n "fatal error:\|: error:" "${log_file}" 2>/dev/null \
        | grep -v "subprocess-exited\|error: subprocess" \
        | grep -v "^Binary" | head -1 | cut -d: -f1 || true)

    if [[ -n "${first_err_line}" ]]; then
        local ctx_start=$(( first_err_line > 10 ? first_err_line - 10 : 1 ))
        error "Compiler error at log line ${first_err_line} (context):" >&2
        sed -n "${ctx_start},$((first_err_line + 30))p" "${log_file}" >&2
        echo "" >&2
        local all_errors
        all_errors=$(grep "fatal error:\|: error:" "${log_file}" 2>/dev/null \
            | grep -v "subprocess-exited\|error: subprocess" | grep -v "^Binary" || true)
        local err_count
        err_count=$(echo "${all_errors}" | grep -c . || true)
        if (( err_count > 1 )); then
            error "All ${err_count} compiler error lines:" >&2
            echo "${all_errors}" | head -40 >&2
        fi
    else
        error "Last 100 lines of build log:" >&2
        tail -100 "${log_file}" >&2
    fi
    error "Full log: ${log_file}" >&2
}

# ---------------------------------------------------------------------------
# Progress bar — monitors a background build by parsing ninja's [X/N] output
# ---------------------------------------------------------------------------
_run_with_progress() {
    # Run "$@" in the background, stream a progress bar by tailing the log,
    # then wait for completion and return the exit code.
    # Usage: _run_with_progress LOG_FILE CMD [ARGS...]
    local log_file="$1"; shift
    local bar_w=45 pid start elapsed current total filled pct numbers progress

    # Launch the build. Trap ensures Ctrl+C kills the background job too.
    ( "$@" ) >"${log_file}" 2>&1 &
    pid=$!
    trap "kill '${pid}' 2>/dev/null; exit 130" INT TERM
    start=${SECONDS}

    printf "\n"
    while kill -0 "${pid}" 2>/dev/null; do
        elapsed=$(( SECONDS - start ))

        # Ninja emits lines like "[42/1234] Building CUDA object ..."
        # Search only the last 200 lines to keep grep fast on large logs.
        progress=$(tail -200 "${log_file}" 2>/dev/null \
                   | grep -oE '\[[0-9]+/[0-9]+\]' | tail -1 || true)

        if [[ -n "${progress}" ]]; then
            numbers="${progress//[\[\]]/}"      # "42/1234"
            current="${numbers%/*}"             # "42"
            total="${numbers#*/}"               # "1234"
            if (( total > 0 )); then
                pct=$(( current * 100 / total ))
                filled=$(( current * bar_w / total ))
                printf "\r  \033[0;32m%s\033[0m\033[2m%s\033[0m  %3d%%  %d/%d  %02d:%02d" \
                    "$(printf '█%.0s' $(seq 1 $(( filled > 0 ? filled : 0 ))))" \
                    "$(printf '░%.0s' $(seq 1 $(( bar_w - filled ))))" \
                    "${pct}" "${current}" "${total}" \
                    "$(( elapsed / 60 ))" "$(( elapsed % 60 ))"
            fi
        else
            # Compilation hasn't started yet (setup / submodule clone phase)
            local spinner='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
            local spin_char="${spinner:$(( elapsed % ${#spinner} )):1}"
            printf "\r  %s  Preparing build...  %02d:%02d" \
                "${spin_char}" "$(( elapsed / 60 ))" "$(( elapsed % 60 ))"
        fi
        sleep 2
    done

    printf "\r%-80s\n" ""   # clear the progress line
    wait "${pid}"
    return $?
}

build_flash_attn() {
    # --- Fast path: install from a pre-built wheel ---
    if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
        info "Installing flash-attn from pre-built wheel: ${FLASH_ATTN_WHEEL}"
        conda run -n "${ENV_NAME}" pip install "${FLASH_ATTN_WHEEL}"
        FA_VER=$(conda run -n "${ENV_NAME}" python -c \
            "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
        info "flash-attn installed: version ${FA_VER}"
        return
    fi

    # --- Clone / update source ---
    info "Cloning flash-attention from ${FLASH_ATTN_REPO} ..."
    if [[ -d "${FLASH_ATTN_DIR}/.git" ]]; then
        warn "Source dir ${FLASH_ATTN_DIR} exists — pulling latest"
        git -C "${FLASH_ATTN_DIR}" fetch origin
        git -C "${FLASH_ATTN_DIR}" reset --hard origin/main
    else
        rm -rf "${FLASH_ATTN_DIR}"
        git clone --depth 1 "${FLASH_ATTN_REPO}" "${FLASH_ATTN_DIR}"
    fi

    local cuda_home arch_flags build_log pip_cmd env_prefix
    cuda_home=$(_resolve_cuda_home)
    arch_flags=$(_detect_arch_flags)
    build_log=$(mktemp /tmp/flash_attn_build.XXXXXX.log)

    # pip lives in the Python env prefix, not in the CUDA toolkit directory.
    # Use `conda run` to locate it reliably.
    env_prefix=$(conda run -n "${ENV_NAME}" python -c \
        "import sys; print(sys.prefix)" 2>/dev/null || true)
    pip_cmd="${env_prefix}/bin/pip"
    [[ -x "${pip_cmd}" ]] || die "pip not found at ${pip_cmd} — is the '${ENV_NAME}' env set up?"

    info "Building flash-attention (MAX_JOBS=${MAX_JOBS}, ~15–60 min)"
    [[ -n "${arch_flags}" ]] && info "Targeting GPU arch(s): ${arch_flags}"
    info "Full build log: ${build_log}"

    # PYTHONUNBUFFERED forces line-by-line flushing so the log is live.
    local -a build_env=(
        env
        PYTHONUNBUFFERED=1
        FLASH_ATTENTION_FORCE_BUILD=TRUE
        MAX_JOBS="${MAX_JOBS}"
        CUDA_HOME="${cuda_home}"
        TORCH_CUDA_ARCH_LIST="${arch_flags}"
    )

    cd "${FLASH_ATTN_DIR}"

    if [[ -n "${FLASH_ATTN_WHEEL_OUT}" ]]; then
        mkdir -p "${FLASH_ATTN_WHEEL_OUT}"
        info "Will save wheel to ${FLASH_ATTN_WHEEL_OUT}"
        _run_with_progress "${build_log}" \
            "${build_env[@]}" "${pip_cmd}" \
                wheel . --no-build-isolation -w "${FLASH_ATTN_WHEEL_OUT}" \
        || { _show_build_error "${build_log}"; die "flash-attn build failed"; }

        local wheel_path
        wheel_path=$(ls "${FLASH_ATTN_WHEEL_OUT}"/flash_attn-*.whl 2>/dev/null | tail -1)
        [[ -z "${wheel_path}" ]] && die "Wheel not found in ${FLASH_ATTN_WHEEL_OUT} after build"
        info "Wheel saved: ${wheel_path}"
        info "Reuse on other servers with the same Python/CUDA/arch:"
        info "  FLASH_ATTN_WHEEL=${wheel_path} bash setup_env.sh"
        "${pip_cmd}" install "${wheel_path}"
    else
        _run_with_progress "${build_log}" \
            "${build_env[@]}" "${pip_cmd}" \
                install . --no-build-isolation \
        || { _show_build_error "${build_log}"; die "flash-attn build failed"; }
    fi

    local fa_ver
    fa_ver=$("${env_prefix}/bin/python" -c \
        "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
    info "flash-attn installed: version ${fa_ver}"
}

# ---------------------------------------------------------------------------
# PostgreSQL coordinator setup (--coordinator only)
# ---------------------------------------------------------------------------
_PG_USER="optuna"
_PG_PASS="choroshps"
_PG_DB="optuna_choros"

setup_postgres_server() {
    info "Setting up PostgreSQL coordinator ..."

    # Install server if missing
    if ! command -v pg_lsclusters &>/dev/null; then
        info "Installing PostgreSQL ..."
        sudo apt-get install -y postgresql postgresql-contrib
    fi

    # Detect active cluster version
    local pg_ver
    pg_ver=$(pg_lsclusters -h | awk 'NR==1{print $1}')
    [[ -z "${pg_ver}" ]] && die "No PostgreSQL cluster found after install"
    local pg_conf_dir="/etc/postgresql/${pg_ver}/main"

    # Ensure service is started and enabled
    sudo systemctl enable postgresql
    sudo pg_ctlcluster "${pg_ver}" main start 2>/dev/null || true

    # Create role (idempotent)
    if ! sudo -u postgres psql -tAc \
            "SELECT 1 FROM pg_roles WHERE rolname='${_PG_USER}'" | grep -q 1; then
        sudo -u postgres psql -c \
            "CREATE USER ${_PG_USER} WITH PASSWORD '${_PG_PASS}';"
        info "Created PostgreSQL role '${_PG_USER}'"
    else
        info "Role '${_PG_USER}' already exists — skipping"
    fi

    # Create database (idempotent)
    if ! sudo -u postgres psql -tAc \
            "SELECT 1 FROM pg_database WHERE datname='${_PG_DB}'" | grep -q 1; then
        sudo -u postgres psql -c \
            "CREATE DATABASE ${_PG_DB} OWNER ${_PG_USER};"
        info "Created database '${_PG_DB}'"
    else
        info "Database '${_PG_DB}' already exists — skipping"
    fi

    # Configure listen_addresses = '*'  (idempotent)
    if ! grep -qE "^listen_addresses\s*=\s*'\*'" "${pg_conf_dir}/postgresql.conf"; then
        sudo sed -i \
            "s/^#\?listen_addresses\s*=.*/listen_addresses = '*'/" \
            "${pg_conf_dir}/postgresql.conf"
        info "Set listen_addresses = '*'"
    fi

    # Add LAN rule to pg_hba.conf (idempotent, derives /24 from this machine's IP)
    local local_ip local_subnet pg_hba
    local_ip=$(hostname -I | awk '{print $1}')
    local_subnet="${local_ip%.*}.0/24"
    pg_hba="${pg_conf_dir}/pg_hba.conf"
    if ! grep -qE "^host\s+${_PG_DB}\s+${_PG_USER}\s+${local_subnet}" "${pg_hba}"; then
        echo "host  ${_PG_DB}  ${_PG_USER}  ${local_subnet}  md5" | \
            sudo tee -a "${pg_hba}" >/dev/null
        info "Added pg_hba rule for subnet ${local_subnet}"
    fi

    # Restart to apply config changes
    sudo pg_ctlcluster "${pg_ver}" main restart
    info "PostgreSQL coordinator ready"
    info "  Storage URL: postgresql://${_PG_USER}:${_PG_PASS}@app.arcadea.us/${_PG_DB}"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Setup complete"
    echo "════════════════════════════════════════════════════════"
    local _pfx; _pfx=$(conda run -n "${ENV_NAME}" python -c "import sys;print(sys.prefix)" 2>/dev/null)
    "${_pfx}/bin/python" - <<'EOF'
import sys, torch
print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}  (available={torch.cuda.is_available()})")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}   : {p.name}  sm{p.major}{p.minor}  {p.total_memory/1024**3:.1f} GB")
try:
    import flash_attn
    print(f"  flash-attn: {flash_attn.__version__}")
except ImportError:
    print("  flash-attn: NOT installed")
EOF
    echo ""
    echo "  Activate with:  conda activate ${ENV_NAME}"
    echo "════════════════════════════════════════════════════════"
}

# ---------------------------------------------------------------------------
# Download and extract data
# ---------------------------------------------------------------------------
download_data() {
    info "Running download_data.py from ${PROJECT_ROOT} ..."
    cd "${PROJECT_ROOT}"
    conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/download_data.py"

    info "Extracting data/CHOROS.tar.gz ..."
    pv data/CHOROS.tar.gz | pigz -dc | tar -xf -
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run_checks || exit 1

if (( CHECK_ONLY == 1 )); then
    info "--check mode: exiting without making changes."
    exit 0
fi

install_system_packages
create_env
install_packages
build_flash_attn
if (( COORDINATOR == 1 )); then
    setup_postgres_server
fi
print_summary
download_data
