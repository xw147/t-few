#!/bin/bash
# ===========================================================================
# download_models.sh  —  pre-fetch HuggingFace models into the Scratch cache
#
# WHERE : run on a LOGIN NODE 
# RUN   : bash download_models.sh                 # downloads T0_3B and T0
#         bash download_models.sh bigscience/T0   # or pass specific id(s)
#
# Uses snapshot_download with resume_download=True, wrapped in a retry loop, so
# a dropped/timed-out connection just resumes from the partial file instead of
# restarting the 44 GB download. Safe to re-run any time.
# ===========================================================================
set -eo pipefail

# --- Environment -----------------------------------------------------------
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tfew

: "${SCRATCH:=/mnt/scratch/$USER}"          # fallback if $SCRATCH isn't set
export HF_HOME="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"

# --- Which models to download ----------------------------------------------
# Defaults match T-Few's t03b.json / t011b.json origin_model fields.
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("bigscience/T0_3B" "bigscience/T0")
fi

# --- Download (resume + retry) ---------------------------------------------
for m in "${MODELS[@]}"; do
  echo ">>> Downloading: $m"
  attempts=0; max_attempts=100
  until python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$m', resume_download=True, max_workers=4)"; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge "$max_attempts" ]; then
      echo ">>> Gave up on $m after $attempts attempts. Check your connection / the repo id."
      exit 1
    fi
    echo ">>> '$m' interrupted (network timeout). Resuming (attempt $attempts)..."
    sleep 10
  done
  echo ">>> '$m' complete."
done

# --- Verify ----------------------------------------------------------------
echo ""
echo ">>> Cache location: $HF_HOME"
du -sh "$HF_HOME"
echo ""
huggingface-cli scan-cache
echo ""
echo ">>> Done. If scan-cache lists your models with no .incomplete files, you're set."
