#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENVIRONMENT_FILE="${1:-$SCRIPT_DIR/environment_experot.yaml}"
ENV_NAME="tfew"
RECREATE_ENV="0"

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [environment.yaml] [options]

Arguments:
  environment.yaml          Path to environment yaml file (default: ./environment_experot.yaml)

Options:
  -m, --miniforge-dir PATH  Path to existing Miniforge install (default: \$HOME/miniforge3)
      --recreate            Remove and recreate the environment if it already exists
  -h, --help                Show this help message

Examples:
  bash $(basename "$0")
  bash $(basename "$0") environment_experot.yaml --recreate
  bash $(basename "$0") --miniforge-dir /opt/miniforge3
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--miniforge-dir)
            MINIFORGE_DIR="$2"
            shift 2
            ;;
        --recreate)
            RECREATE_ENV="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ENVIRONMENT_FILE="$1"
            shift
            ;;
    esac
done

# Validate files and paths
if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
    echo "Environment file not found: $ENVIRONMENT_FILE" >&2
    exit 1
fi

if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
    echo "Miniforge not found at: $MINIFORGE_DIR" >&2
    echo "Set MINIFORGE_DIR or pass --miniforge-dir to the existing install location." >&2
    exit 1
fi

# Load conda
if [[ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
    echo "Unable to load conda from $MINIFORGE_DIR" >&2
    exit 1
fi

# Check if environment exists
env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

# Remove environment if recreating
if [[ "$RECREATE_ENV" == "1" ]] && env_exists; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

# Create environment from yaml
echo "Creating/updating environment from: $ENVIRONMENT_FILE"
conda env create -f "$ENVIRONMENT_FILE" --name "$ENV_NAME" -y

conda activate "$ENV_NAME"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment in a new shell session:"
echo "  source $MINIFORGE_DIR/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Quick validation:"
echo "  python -c \"import torch, transformers, datasets, pytorch_lightning; print('✓ All packages imported successfully')\""
echo ""
