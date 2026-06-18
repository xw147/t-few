#!/bin/bash
#SBATCH --job-name=tfew
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tfew_ico_%j.out
#SBATCH --error=logs/tfew_ico_%j.err
# ===========================================================================
# T-Few few-shot training on HPC
# Submit from a login node with:   sbatch few-shot-pretrained-100k.sh
#
# BEFORE the first submit, pre-download the model ON THE LOGIN NODE 
# ===========================================================================
set -eo pipefail

# --- Environment -----------------------------------------------------------
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tfew
# export CUDA_HOME=$CONDA_PREFIX     # uncomment ONLY if you switch to DeepSpeed

# --- Paths ------------------------------------------------
REPO=$HOME/work/t-few
export CONFIG_PATH=$REPO/configs
export HF_HOME=$SCRATCH/hf_cache            # 40GB+ model cache -> Scratch
export OUTPUT_PATH=$HOME/t-few/exp_out
mkdir -p "$HF_HOME" "$OUTPUT_PATH" logs
cd "$REPO"                                  # so 'python -m src.pl_train' finds src/

# --- Confirm we're really on the L40S --------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# After pre-downloading the model, you can force offline loading (recommended):
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Experiment settings ---------------------------------------------------
allow_skip_exp=True
train_batch_size=4          # for T0-11B, drop to 1 if you hit CUDA OOM
grad_accum_factor=1
lr=0.003
re='^[0-9]+$'

num_steps=0
eval_epoch_interval=0

# Start with t03b to validate the whole pipeline cheaply (~minutes/exp),
# then change to 't011b' for the real runs.
for model in 't03b'         # <-- change to 't011b' once a t03b run works
do
  # for num_shot in 4 8 16 32 64 128
  for num_shot in 4
  do
    for dataset in ico
    do
      eval_before_training=False
      num_steps=$(( 30 * ($num_shot / $train_batch_size) ))
      eval_epoch_interval=30

      # for seed in 42 1024 0 1 32
      for seed in 42 
      do
        for ico_label_strategy in all
        do
          python -m src.pl_train \
            -c ${model}.json+ia3.json+global.json \
            -k dataset=${dataset} \
               load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" \
               num_steps=${num_steps} num_shot=${num_shot} \
               exp_name=${model}_${dataset}_${ico_label_strategy}_numshot${num_shot}_seed${seed}_ia3_pretrained100k \
               few_shot_random_seed=${seed} seed=${seed} \
               allow_skip_exp=${allow_skip_exp} \
               eval_before_training=${eval_before_training} \
               eval_epoch_interval=${eval_epoch_interval} \
               batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} \
               lr=${lr} compute_strategy="none" \
               compute_precision=bf16 \
               ico_label_strategy=${ico_label_strategy}
        done
      done
    done
  done
done

echo "All experiments finished."
