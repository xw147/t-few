#!/bin/bash

# Setup script for T-Few on macOS with Python 3.10
# Environment name: tfew_mac310

echo "========================================="
echo "T-Few macOS Setup (Python 3.10)"
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment: tfew_mac310 with Python 3.10..."
conda create -n tfew_mac310 python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate tfew_mac310

# Verify Python version
echo ""
echo "Python version:"
python --version

# Install PyTorch with MPS support
echo ""
echo "Installing PyTorch with MPS (GPU) support..."
pip install torch torchvision torchaudio

# Install promptsource
echo ""
echo "Installing promptsource..."
pip install promptsource --no-deps

# Install other dependencies
echo ""
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify MPS availability
echo ""
echo "Verifying MPS (Metal Performance Shaders) support..."
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"

# Check Python environment
echo ""
echo "Checking Python environment..."
python -c "from src.path_config import DATASETS_OFFLINE, TEMPLATES_BASE; print('DATASETS_OFFLINE:', DATASETS_OFFLINE); print('TEMPLATES_BASE:', TEMPLATES_BASE)"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate tfew_mac310"
echo ""
echo "To verify your setup, run:"
echo "  python -c 'import torch; print(\"MPS:\", torch.backends.mps.is_available())'"
echo ""
echo "For more information, see:"
echo "  - MACOS_SETUP.md"
echo "  - APPLE_SILICON_GUIDE.md"
echo ""
