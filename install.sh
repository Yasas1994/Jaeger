#!/usr/bin/env bash
#
# One-liner install script for Jaeger.
# Usage: curl -sSL https://raw.githubusercontent.com/MGXlab/Jaeger/main/install.sh | bash
#        bash <(curl -sSL https://raw.githubusercontent.com/MGXlab/Jaeger/main/install.sh)
#
# This script detects your platform and installs Jaeger with the appropriate
# dependencies (GPU, CPU, or Apple Silicon).

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { printf "${BLUE}[INFO]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }

# Detect platform
detect_platform() {
    local os arch
    os=$(uname -s)
    arch=$(uname -m)

    if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
        echo "darwin-arm"
    elif command -v nvidia-smi &>/dev/null; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# Detect conda/mamba
has_conda() { command -v conda &>/dev/null; }
has_mamba() { command -v mamba &>/dev/null; }

# Main install logic
main() {
    local platform pkg_extra env_cmd

    platform=$(detect_platform)
    case "$platform" in
        gpu)
            pkg_extra="[gpu]"
            info "NVIDIA GPU detected — installing with GPU support"
            ;;
        cpu)
            pkg_extra="[cpu]"
            info "No NVIDIA GPU detected — installing CPU-only version"
            ;;
        darwin-arm)
            pkg_extra="[darwin-arm]"
            info "Apple Silicon (ARM) detected — installing ARM-optimized version"
            ;;
    esac

    # Prefer conda/mamba if available
    if has_mamba || has_conda; then
        info "Conda detected — creating 'jaeger' environment"

        if has_mamba; then
            env_cmd="mamba"
        else
            env_cmd="conda"
            warn "Mamba not found, falling back to conda (slower)"
        fi

        # Remove existing environment if present
        if conda env list | grep -q "^jaeger "; then
            warn "Existing 'jaeger' environment found — removing it"
            conda env remove -n jaeger -y
        fi

        # Create environment
        info "Creating conda environment 'jaeger' with Python 3.11–3.12"
        $env_cmd create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<=3.12" pip -y

        # Activate and install
        info "Activating environment and installing jaeger-bio${pkg_extra}"
        # shellcheck source=/dev/null
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate jaeger
        pip install "jaeger-bio${pkg_extra}"

    else
        info "Conda not detected — using Python venv"

        # Check Python version
        local py_version
        py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$py_version" != "3.11" && "$py_version" != "3.12" ]]; then
            error "Python ${py_version} detected. Jaeger requires Python 3.11 or 3.12."
            exit 1
        fi

        # Remove existing venv if present
        if [[ -d "jaeger" ]]; then
            warn "Existing 'jaeger' virtual environment found — removing it"
            rm -rf jaeger
        fi

        # Create venv and install
        python3 -m venv jaeger
        # shellcheck source=/dev/null
        source jaeger/bin/activate
        pip install --upgrade pip
        pip install "jaeger-bio${pkg_extra}"
    fi

    # Verify installation
    ok "Installation complete!"
    info "Running health check..."
    if jaeger health; then
        ok "Jaeger is ready to use!"
        info "Activate the environment with: ${YELLOW}conda activate jaeger${NC}"
    else
        error "Health check failed. Please check the error messages above."
        exit 1
    fi
}

main "$@"
