#!/usr/bin/env bash
# =============================================================================
#  🧠  B R A I N B R E W  —  Bulletproof Installer
#  Version : 1.2.0
#  Supports: Linux (Ubuntu/Debian/Arch/RHEL) · macOS · WSL2
#  Author  : Yog-Sotho  |  MIT License
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & COLOUR PALETTE
# ---------------------------------------------------------------------------
readonly SCRIPT_VERSION="1.2.0"
readonly REQUIRED_PYTHON_MAJOR=3
readonly REQUIRED_PYTHON_MINOR=12
readonly VENV_DIR=".venv"
readonly ENV_FILE=".env"
readonly ENV_SAMPLE=".env.sample"
readonly REQUIREMENTS="requirements.txt"
readonly LOG_FILE="install.log"

# ANSI colours (disabled automatically when not a terminal)
if [[ -t 1 ]]; then
  RED='\033[0;31m';  GREEN='\033[0;32m';  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'; CYAN='\033[0;36m';   BOLD='\033[1m';  RESET='\033[0m'
  MAGENTA='\033[0;35m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; RESET=''; MAGENTA=''
fi

# ---------------------------------------------------------------------------
# 1.  LOGGING HELPERS
# ---------------------------------------------------------------------------
log()     { echo -e "${BOLD}${BLUE}[INFO]${RESET}  $*" | tee -a "$LOG_FILE"; }
ok()      { echo -e "${BOLD}${GREEN}[ OK ]${RESET}  $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${BOLD}${YELLOW}[WARN]${RESET}  $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${BOLD}${RED}[FAIL]${RESET}  $*" | tee -a "$LOG_FILE" >&2; }
step()    { echo -e "\n${BOLD}${CYAN}---  $*  ---${RESET}" | tee -a "$LOG_FILE"; }
banner()  {
  echo -e "${BOLD}${MAGENTA}"
  echo "  +------------------------------------------------------+"
  echo "  |   Brainbrew -- Synthetic Dataset Generator            |"
  echo "  |       Production Installer  v${SCRIPT_VERSION}                   |"
  echo "  +------------------------------------------------------+"
  echo -e "${RESET}"
}

die() {
  error "$1"
  echo ""
  error "Installation aborted. Check ${LOG_FILE} for full output."
  exit 1
}

# ---------------------------------------------------------------------------
# 2.  ARGUMENT PARSING
# ---------------------------------------------------------------------------
MODE="interactive"      # interactive | docker | ci
SKIP_VENV=false
SKIP_GPU_CHECK=false
SKIP_TESTS=false
NO_COLOUR=false

usage() {
  cat << EOF
Usage: bash install.sh [OPTIONS]

Options:
  --docker          Build & start via Docker instead of local venv
  --ci              Non-interactive mode (reads env vars; skips prompts)
  --no-venv         Skip virtual-environment creation (use system Python)
  --no-gpu-check    Skip CUDA / GPU detection
  --skip-tests      Skip the post-install import smoke tests
  --no-colour       Disable ANSI colours
  -h, --help        Show this help message

Environment variables (CI mode):
  BB_OPENAI_KEY     OpenAI API key
  BB_HF_TOKEN       Hugging Face token
  BB_HF_USERNAME    Hugging Face username

Examples:
  bash install.sh                     # Standard interactive install
  bash install.sh --docker            # Docker install
  BB_OPENAI_KEY=sk-... bash install.sh --ci
EOF
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --docker)        MODE="docker"          ;;
    --ci)            MODE="ci"              ;;
    --no-venv)       SKIP_VENV=true         ;;
    --no-gpu-check)  SKIP_GPU_CHECK=true    ;;
    --skip-tests)    SKIP_TESTS=true        ;;
    --no-colour)     NO_COLOUR=true; RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; RESET=''; MAGENTA='' ;;
    -h|--help)       usage                  ;;
    *)               warn "Unknown option: $arg -- ignoring" ;;
  esac
done

# ---------------------------------------------------------------------------
# 3.  INITIALISE LOG
# ---------------------------------------------------------------------------
: > "$LOG_FILE"
echo "BrainBrew install started at $(date)" >> "$LOG_FILE"

banner

# ---------------------------------------------------------------------------
# 4.  OPERATING SYSTEM DETECTION
# ---------------------------------------------------------------------------
step "Detecting operating system"

OS_TYPE="unknown"
PKG_MANAGER="unknown"

if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
  OS_TYPE="linux"
  if grep -qi "microsoft" /proc/version 2>/dev/null; then
    OS_TYPE="wsl"
    warn "WSL2 detected -- GPU passthrough requires WSL2 + CUDA drivers on host."
  fi
  if command -v apt-get &>/dev/null; then   PKG_MANAGER="apt";   fi
  if command -v dnf     &>/dev/null; then   PKG_MANAGER="dnf";   fi
  if command -v pacman  &>/dev/null; then   PKG_MANAGER="pacman"; fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
  PKG_MANAGER="brew"
  warn "macOS detected -- vLLM is Linux-only. You must use OpenAI API mode."
  SKIP_GPU_CHECK=true
else
  warn "Unknown OS type: $OSTYPE -- proceeding with best effort."
fi

ok "OS: ${OS_TYPE}  |  Package manager: ${PKG_MANAGER}"

# ---------------------------------------------------------------------------
# 5.  CORE DEPENDENCY CHECKS
# ---------------------------------------------------------------------------
step "Checking core system dependencies"

# -- git -------------------------------------------------------------------
if ! command -v git &>/dev/null; then
  die "git is not installed. Install it with your package manager and re-run."
fi
ok "git $(git --version | awk '{print $3}')"

# -- curl ------------------------------------------------------------------
if ! command -v curl &>/dev/null; then
  warn "curl not found -- some network checks will be skipped."
fi

# -- Python version --------------------------------------------------------
PYTHON_CMD=""
for cmd in python3.12 python3.13 python3.14 python3; do
  if command -v "$cmd" &>/dev/null; then
    PY_MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    PY_MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [[ "$PY_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ]] && \
       [[ "$PY_MINOR" -ge "$REQUIRED_PYTHON_MINOR" ]]; then
      PYTHON_CMD="$cmd"
      break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  error "Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ is required but was not found."
  echo ""
  case "$PKG_MANAGER" in
    apt)    echo "  -> sudo apt install python3.12 python3.12-venv python3.12-dev" ;;
    dnf)    echo "  -> sudo dnf install python3.12" ;;
    pacman) echo "  -> sudo pacman -S python" ;;
    brew)   echo "  -> brew install python@3.12" ;;
  esac
  die "Please install Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ and re-run."
fi

PY_VERSION=$("$PYTHON_CMD" --version 2>&1)
ok "Python: $PY_VERSION  (${PYTHON_CMD})"

# -- pip -------------------------------------------------------------------
if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
  die "pip not found for ${PYTHON_CMD}. Install python3-pip and re-run."
fi
ok "pip: $("$PYTHON_CMD" -m pip --version | awk '{print $2}')"

# -- venv module -----------------------------------------------------------
if [[ "$SKIP_VENV" == false ]]; then
  if ! "$PYTHON_CMD" -m venv --help &>/dev/null; then
    error "python3-venv module not found."
    case "$PKG_MANAGER" in
      apt) echo "  -> sudo apt install python3.12-venv" ;;
      *)   echo "  -> Install the venv module for your Python distribution." ;;
    esac
    die "Cannot create virtual environment."
  fi
fi

# ---------------------------------------------------------------------------
# 6.  GPU / CUDA DETECTION
# ---------------------------------------------------------------------------
step "Detecting GPU / CUDA"

HAS_GPU=false
GPU_VRAM_GB=0
RECOMMENDED_MODE="openai"

if [[ "$SKIP_GPU_CHECK" == false ]]; then
  if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ -n "$GPU_INFO" ]]; then
      GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
      VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
      GPU_VRAM_GB=$(( VRAM_MIB / 1024 ))
      HAS_GPU=true
      ok "GPU detected: ${GPU_NAME}  |  VRAM: ${GPU_VRAM_GB} GB"

      if [[ "$GPU_VRAM_GB" -ge 24 ]]; then
        RECOMMENDED_MODE="vllm"
        ok "VRAM >= 24 GB -- vLLM mode recommended"
      elif [[ "$GPU_VRAM_GB" -ge 8 ]]; then
        RECOMMENDED_MODE="openai_with_lora"
        warn "VRAM ${GPU_VRAM_GB} GB -- Enough for LoRA training, but not vLLM inference."
      else
        warn "VRAM ${GPU_VRAM_GB} GB -- Below recommended 24 GB. Using OpenAI API mode."
      fi
    fi
  fi

  if [[ "$HAS_GPU" == false ]]; then
    warn "No NVIDIA GPU detected. Brainbrew will run in OpenAI API mode (no local inference)."
    RECOMMENDED_MODE="openai"
  fi
fi

# ---------------------------------------------------------------------------
# 7.  DOCKER PATH
# ---------------------------------------------------------------------------
if [[ "$MODE" == "docker" ]]; then
  step "Docker installation mode"

  if ! command -v docker &>/dev/null; then
    die "Docker is not installed. See: https://docs.docker.com/get-docker/"
  fi
  ok "Docker: $(docker --version)"

  if ! docker info &>/dev/null; then
    die "Docker daemon is not running. Start it and re-run."
  fi

  if [[ ! -f "Dockerfile" ]]; then
    die "Dockerfile not found at repo root. Cannot build."
  fi

  # Ensure .env exists before docker build
  if [[ ! -f "$ENV_FILE" ]] && [[ -f "$ENV_SAMPLE" ]]; then
    cp "$ENV_SAMPLE" "$ENV_FILE"
    warn "Copied .env.sample -> .env. Edit it with real API keys before running the container."
  fi

  log "Building Docker image: brainbrew ..."
  docker build -t brainbrew . 2>&1 | tee -a "$LOG_FILE"

  ok "Docker image built successfully."
  echo ""
  echo -e "${BOLD}${GREEN}Run with:${RESET}"
  if [[ "$HAS_GPU" == true ]]; then
    echo "  docker run --gpus all -p 8501:8501 --env-file .env brainbrew"
  else
    echo "  docker run -p 8501:8501 --env-file .env brainbrew"
  fi
  echo "  Then open: http://localhost:8501"
  exit 0
fi

# ---------------------------------------------------------------------------
# 8.  PROJECT STRUCTURE VALIDATION
# ---------------------------------------------------------------------------
step "Validating project structure"

REQUIRED_FILES=("app.py" "config.py" "orchestrator.py" "requirements.txt")
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    die "Required file missing: ${f}. Make sure you are running this script from the Brainbrew repo root."
  fi
done
ok "Core project files present."

# Ensure Python package __init__.py files exist
for pkg_dir in "pipeline" "training" "publish" "tests"; do
  if [[ -d "$pkg_dir" ]] && [[ ! -f "${pkg_dir}/__init__.py" ]]; then
    log "Creating missing ${pkg_dir}/__init__.py"
    touch "${pkg_dir}/__init__.py"
    ok "Created ${pkg_dir}/__init__.py"
  fi
done

# ---------------------------------------------------------------------------
# 9.  VIRTUAL ENVIRONMENT
# ---------------------------------------------------------------------------
step "Setting up Python virtual environment"

if [[ "$SKIP_VENV" == true ]]; then
  warn "--no-venv specified -- using system Python. Not recommended for production."
  PIP_CMD="$PYTHON_CMD -m pip"
  PYTHON_ACTIVE="$PYTHON_CMD"
else
  if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment '${VENV_DIR}' already exists -- reusing it."
    warn "To force a clean install: rm -rf ${VENV_DIR} && bash install.sh"
  else
    log "Creating virtual environment at: ${VENV_DIR}"
    "$PYTHON_CMD" -m venv "$VENV_DIR" 2>&1 | tee -a "$LOG_FILE"
    ok "Virtual environment created."
  fi

  # Activate
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
  elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/Scripts/activate"
  else
    die "Cannot find activation script in ${VENV_DIR}. Virtual environment may be corrupt."
  fi

  PYTHON_ACTIVE="${VENV_DIR}/bin/python"
  PIP_CMD="${VENV_DIR}/bin/pip"
  ok "Virtual environment activated."
fi

# ---------------------------------------------------------------------------
# 10.  PIP UPGRADE
# ---------------------------------------------------------------------------
step "Upgrading pip, setuptools, wheel"
"$PIP_CMD" install --quiet --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"
ok "pip upgraded to $("$PIP_CMD" --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 11.  DEPENDENCY INSTALLATION
# ---------------------------------------------------------------------------
step "Installing Python dependencies from requirements.txt"

if [[ "$HAS_GPU" == false ]]; then
  warn "No GPU detected. GPU packages (vLLM, Unsloth, bitsandbytes) will be installed"
  warn "but may fail at runtime without CUDA. That is expected."
fi

# Count packages for progress feedback
TOTAL_PKGS=$(grep -cE '^\s*[^#]' "$REQUIREMENTS" 2>/dev/null || echo "?")
log "Installing ${TOTAL_PKGS} packages (this may take 5-15 minutes on first run)..."

# Install with error isolation
if "$PIP_CMD" install --quiet -r "$REQUIREMENTS" 2>&1 | tee -a "$LOG_FILE"; then
  ok "All dependencies installed successfully."
else
  warn "Batch install encountered errors. Attempting package-by-package install..."
  FAILED_PKGS=()
  while IFS= read -r pkg; do
    # Skip blank lines and comments
    [[ -z "$pkg" || "$pkg" =~ ^[[:space:]]*# ]] && continue
    if ! "$PIP_CMD" install --quiet "$pkg" 2>>"$LOG_FILE"; then
      warn "  x Failed: $pkg"
      FAILED_PKGS+=("$pkg")
    fi
  done < "$REQUIREMENTS"

  if [[ ${#FAILED_PKGS[@]} -gt 0 ]]; then
    warn "The following packages failed to install:"
    for pkg in "${FAILED_PKGS[@]}"; do
      warn "  - $pkg"
    done
    warn "Common reasons: vllm/unsloth/bitsandbytes require CUDA + Linux."
    warn "Check ${LOG_FILE} for full error output."
  fi
fi

# ---------------------------------------------------------------------------
# 12.  .ENV FILE SETUP
# ---------------------------------------------------------------------------
step "Configuring environment variables (.env)"

_write_env() {
  local openai_key="$1"
  local hf_token="$2"
  local hf_username="$3"
  cat > "$ENV_FILE" << ENVEOF
# Brainbrew Environment Configuration
# Generated by install.sh on $(date)

OPENAI_API_KEY=${openai_key}
HF_TOKEN=${hf_token}
HF_USERNAME=${hf_username}
ENVEOF
  ok ".env written successfully."
}

if [[ "$MODE" == "ci" ]]; then
  OPENAI_KEY="${BB_OPENAI_KEY:-}"
  HF_TOKEN_VAL="${BB_HF_TOKEN:-}"
  HF_USERNAME="${BB_HF_USERNAME:-yourusername}"

  if [[ -z "$OPENAI_KEY" ]] && [[ "$RECOMMENDED_MODE" == "openai" ]]; then
    warn "BB_OPENAI_KEY is not set. Set it before running Brainbrew."
  fi
  _write_env "$OPENAI_KEY" "$HF_TOKEN_VAL" "$HF_USERNAME"

else
  _do_prompt=false
  if [[ -f "$ENV_FILE" ]]; then
    warn ".env already exists."
    read -rp "  Overwrite it? [y/N]: " OVERWRITE_ENV
    if [[ "$OVERWRITE_ENV" =~ ^[Yy]$ ]]; then
      _do_prompt=true
    else
      log "Keeping existing .env."
    fi
  else
    _do_prompt=true
  fi

  if [[ "$_do_prompt" == true ]]; then
    echo ""
    echo -e "${BOLD}Enter your API credentials.${RESET}"
    echo -e "  Press ${BOLD}Enter${RESET} to leave blank (you can edit .env later)."
    echo ""

    # FIX H-06: read API keys silently so they don't appear in terminal history
    read -rsp "  OpenAI API Key (sk-...): " OPENAI_KEY
    echo ""

    read -rsp "  Hugging Face Token (hf_...): " HF_TOKEN_VAL
    echo ""

    HF_USERNAME_DEFAULT="yourusername"
    read -rp "  Hugging Face Username [${HF_USERNAME_DEFAULT}]: " HF_USERNAME_INPUT
    HF_USERNAME="${HF_USERNAME_INPUT:-$HF_USERNAME_DEFAULT}"

    _write_env "${OPENAI_KEY:-}" "${HF_TOKEN_VAL:-}" "$HF_USERNAME"
  fi

  # Fallback: copy sample if nothing was generated and no .env exists
  if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_SAMPLE" ]]; then
      cp "$ENV_SAMPLE" "$ENV_FILE"
      warn "Copied .env.sample -> .env. Edit it with your real keys before running."
    else
      warn "Neither .env nor .env.sample found. Creating a blank .env template."
      _write_env "" "" "yourusername"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 13.  IMPORT SMOKE TEST
# ---------------------------------------------------------------------------
if [[ "$SKIP_TESTS" == false ]]; then
  step "Verifying critical imports"

  _test_import() {
    local module="$1"
    local label="${2:-$1}"
    if "$PYTHON_ACTIVE" -c "import ${module}" 2>>"$LOG_FILE"; then
      ok "  + ${label}"
      return 0
    else
      warn "  x ${label} -- import failed (check ${LOG_FILE})"
      return 1
    fi
  }

  IMPORT_FAILURES=0

  _test_import "streamlit"                 "streamlit"                || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "pydantic"                  "pydantic"                 || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "structlog"                 "structlog"                || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "dotenv"                    "python-dotenv"            || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "langchain_text_splitters"  "langchain-text-splitters" || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "pdfminer"                  "pdfminer.six"             || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "datasets"                  "datasets"                 || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))
  _test_import "huggingface_hub"           "huggingface_hub"          || IMPORT_FAILURES=$(( IMPORT_FAILURES + 1 ))

  # GPU-only imports -- warn but don't count as failure
  if [[ "$HAS_GPU" == true ]]; then
    _test_import "distilabel" "distilabel" || warn "distilabel import failed -- check CUDA setup."
    _test_import "vllm"       "vllm"       || warn "vLLM import failed -- ensure CUDA drivers are installed."
  else
    "$PYTHON_ACTIVE" -c "import distilabel" 2>>"$LOG_FILE" \
      && ok "  + distilabel" \
      || warn "  x distilabel -- may need GPU drivers at runtime."
  fi

  if [[ "$IMPORT_FAILURES" -gt 0 ]]; then
    warn "${IMPORT_FAILURES} core import(s) failed. Review ${LOG_FILE} for details."
  else
    ok "All critical imports verified."
  fi
else
  log "Skipping import smoke tests (--skip-tests)."
fi

# ---------------------------------------------------------------------------
# 14.  CREATE LAUNCHER SCRIPTS
# ---------------------------------------------------------------------------
step "Creating convenience launcher scripts"

cat > run.sh << 'RUNSCRIPT'
#!/usr/bin/env bash
# Brainbrew -- local launcher
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
  source "${VENV_DIR}/Scripts/activate"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 "$@"
RUNSCRIPT
chmod +x run.sh
ok "Created run.sh"

cat > run_docker.sh << 'DOCKERSCRIPT'
#!/usr/bin/env bash
# Brainbrew -- Docker launcher
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".env" ]]; then
  echo "ERROR: .env file not found. Create it from .env.sample first."
  exit 1
fi

GPU_FLAG=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  GPU_FLAG="--gpus all"
  echo "[INFO] GPU detected -- enabling GPU passthrough."
fi

docker run ${GPU_FLAG} -p 8501:8501 --env-file .env brainbrew "$@"
DOCKERSCRIPT
chmod +x run_docker.sh
ok "Created run_docker.sh"

# ---------------------------------------------------------------------------
# 15.  FINAL SUMMARY
# ---------------------------------------------------------------------------
step "Installation complete"

echo ""
echo -e "${BOLD}${GREEN}+--------------------------------------------------+"
echo -e "|   Brainbrew is ready to brew!                     |"
echo -e "+--------------------------------------------------+${RESET}"
echo ""
echo -e "${BOLD}Quick start:${RESET}"
echo ""

if [[ "$SKIP_VENV" == false ]]; then
  echo -e "  ${CYAN}# Activate the virtual environment${RESET}"
  echo -e "  source ${VENV_DIR}/bin/activate"
  echo ""
fi

echo -e "  ${CYAN}# Option 1 -- Use the launcher script${RESET}"
echo -e "  bash run.sh"
echo ""
echo -e "  ${CYAN}# Option 2 -- Run directly${RESET}"
echo -e "  streamlit run app.py"
echo ""
echo -e "  ${CYAN}# Option 3 -- Docker${RESET}"
echo -e "  docker build -t brainbrew . && bash run_docker.sh"
echo ""
echo -e "  ${CYAN}# Then open in your browser:${RESET}"
echo -e "  http://localhost:8501"
echo ""

echo -e "${BOLD}Configuration:${RESET}"
echo -e "  .env file  ->  edit your OpenAI / HF keys"
echo ""

echo -e "${BOLD}Recommended mode for your hardware:${RESET}"
case "$RECOMMENDED_MODE" in
  vllm)               echo -e "  vLLM mode  (GPU: ${GPU_VRAM_GB} GB VRAM detected)" ;;
  openai_with_lora)   echo -e "  OpenAI API + local LoRA training (GPU: ${GPU_VRAM_GB} GB)" ;;
  openai)             echo -e "  OpenAI API mode  (no GPU / insufficient VRAM)" ;;
esac

echo ""
echo -e "Full install log saved to: ${LOG_FILE}"
echo ""
echo -e "${BOLD}${MAGENTA}Now go brew some brains.${RESET}"
echo ""
