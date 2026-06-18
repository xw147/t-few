#!/bin/bash
# ===========================================================================
# setup_tfew.sh  —  build the T-Few environment on HPC
#
# Target GPU : NVIDIA L40S (Ada, sm_89)  -> needs a CUDA >= 11.8 torch build
# WHERE      : run this ON A LOGIN NODE. NOT on a GPU node.
# RUN WITH   : bash setup_tfew.sh
#
# The conda env lands in ~/.conda/envs/tfew. It is ~8 GB. Re-use it from any job with:
#     module load miniforge && conda activate tfew
# ===========================================================================
set -eo pipefail

ENV_NAME=tfew
PY_VERSION=3.9            # 3.9 (not 3.13): required by promptsource, fits the stack
REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements.txt"

# Make sure the 'module' (Lmod) command exists in this shell
if ! command -v module &>/dev/null; then
  source /etc/profile.d/lmod.sh 2>/dev/null \
    || source /etc/profile.d/modules.sh 2>/dev/null || true
fi

echo ">>> [1/6] Loading Miniforge"
module load miniforge

echo ">>> [2/6] Creating conda env '$ENV_NAME' (python $PY_VERSION)"
conda create -y -n "$ENV_NAME" python="$PY_VERSION"

# Enable 'conda activate' inside a (non-interactive) script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- CUDA 11.8 toolkit for DeepSpeed --------------------------------------
# HPC only ships CUDA 12.x modules, which don't match our cu118 torch build,
# so we install a matching CUDA 11.8 toolkit (incl. nvcc) INSIDE the env.
# DeepSpeed uses it to compile its ops at first run.
#   (If you decide to run single-GPU WITHOUT DeepSpeed, you can delete this
#    step and remove 'deepspeed' from requirements.txt.)
echo ">>> [3/6] Installing CUDA 11.8 toolkit (for DeepSpeed)"
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"

# --- PyTorch (CUDA 11.8 build) --------------------------------------------
# The +cu118 wheel has native sm_89 (L40S) kernels and bundles its own runtime,
# so no system CUDA module is needed to RUN it.
echo ">>> [4/6] Installing PyTorch 2.0.1 (cu118)"
pip install --upgrade pip
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# --- The rest of the stack -------------------------------------------------
echo ">>> [5/6] Installing remaining packages from requirements.txt"
pip install -r "$REQ_FILE"

# --- promptsource (no deps) ------------------------------------------------
# Installed last and WITHOUT deps so its old pins can't downgrade datasets etc.
echo ">>> [6/6] Installing promptsource (--no-deps)"
pip install --no-deps --ignore-requires-python promptsource==0.2.3

# --- Sanity check ----------------------------------------------------------
echo ">>> Verifying the install"
python - <<'PY'
import torch, pytorch_lightning as pl, transformers, datasets, deepspeed, promptsource
print("torch          :", torch.__version__)
print("  built for CUDA:", torch.version.cuda)
print("  cuda available:", torch.cuda.is_available(),
      "(False here is normal — the login node has no GPU)")
print("lightning      :", pl.__version__)
print("transformers   :", transformers.__version__)
print("datasets       :", datasets.__version__)
print("deepspeed      :", deepspeed.__version__)
print("promptsource   : import OK")
PY

echo ""
echo ">>> Done. To use the environment in a job:"
echo "      module load miniforge"
echo "      conda activate $ENV_NAME"
echo "      export CUDA_HOME=\$CONDA_PREFIX   # only needed if you use DeepSpeed"
