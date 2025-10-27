#!/bin/bash
# =============================================================================
# SMART TRAINING SCRIPT - Three Modes in One
# =============================================================================
# 
# USAGE:
# [1] Office Computer (Debug):     Set DEBUG_MODE=true
# [2] HPC Simple Test:             Set SIMPLE_HPC_TEST=true  
# [3] HPC Comprehensive:          Set COMPREHENSIVE_MODE=true
#
# WORKFLOW:
# 1. Test locally:     DEBUG_MODE=true       (2 steps, 1 seed, 1 model)
# 2. Validate on HPC:  SIMPLE_HPC_TEST=true (100 steps, 1 seed, 1 model)
# 3. Full experiment:  COMPREHENSIVE_MODE=true (full steps, all seeds/shots)
#
# To expand experiments, just uncomment lines in the respective sections!
# =============================================================================

# =============================================================================
# CONFIGURATION: Uncomment EXACTLY ONE section you want to run
# =============================================================================

# [1] LOCAL DEBUG MODE (Office Computer - Windows)
# Uncomment for quick testing on your office computer
DEBUG_MODE=true
# DEBUG_MODE=false

# [2] SIMPLE HPC TEST (One model, one seed, small shots)
# Uncomment for initial HPC testing
# SIMPLE_HPC_TEST=true
SIMPLE_HPC_TEST=false

# [3] COMPREHENSIVE EXPERIMENTS (Full grid search)
# Uncomment for complete experiments
# COMPREHENSIVE_MODE=true
COMPREHENSIVE_MODE=false

# =============================================================================
# SCRIPT LOGIC (Don't modify below unless needed)
# =============================================================================

# Basic settings
lr=0.003
re='^[0-9]+$'
cuda_device=0

# Configure based on mode
if [ "$DEBUG_MODE" = true ]; then
    echo "[DEBUG MODE] Quick testing for office computer"
    allow_skip_exp=False  # Set to False to re-run experiment
    eval_before_training=True
    train_batch_size=2
    grad_accum_factor=1
    export TFEW_DEBUG=true
    
elif [ "$SIMPLE_HPC_TEST" = true ]; then
    echo "[SIMPLE HPC TEST] Initial validation on HPC"
    allow_skip_exp=True
    eval_before_training=False
    train_batch_size=4
    grad_accum_factor=1
    export TFEW_PRODUCTION=true
    
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    echo "[COMPREHENSIVE MODE] Full experimental grid"
    allow_skip_exp=True
    eval_before_training=False
    train_batch_size=4
    grad_accum_factor=1
    export TFEW_PRODUCTION=true
    
else
    echo "ERROR: No mode selected!"
    echo "Please uncomment one of the mode flags at the top of the script:"
    echo "  - DEBUG_MODE=true (for office computer testing)"
    echo "  - SIMPLE_HPC_TEST=true (for initial HPC validation)"
    echo "  - COMPREHENSIVE_MODE=true (for full experiments)"
    exit 1
fi

# Model selection based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug: single model
    models="t03b"
elif [ "$SIMPLE_HPC_TEST" = true ]; then
    # Simple test: single model
    models="t03b"
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    # Comprehensive: multiple models (uncomment as needed)
    models="t03b"  # Add t011b when ready
    # models="t03b t011b"  # <-- Uncomment for both models
fi

# Shot selection based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug: minimal shots
    shot_numbers="2"
elif [ "$SIMPLE_HPC_TEST" = true ]; then
    # Simple test: one shot number
    shot_numbers="4"
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    # Comprehensive: full grid (uncomment lines below to expand)
    shot_numbers="4"  
    # shot_numbers="4 8 16 32 64 128 256 512"  # <-- Uncomment for all shots
fi

# Seed selection based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug: single seed
    seeds="42"
elif [ "$SIMPLE_HPC_TEST" = true ]; then
    # Simple test: single seed
    seeds="42"
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    # Comprehensive: multiple seeds (uncomment lines below to expand)
    seeds="42"  
    # seeds="42 1024 0 1 32"    # <-- Uncomment for all 5 seeds
fi

# Dataset selection (same for all modes, but you can modify)
datasets="ico_list"


# =============================================================================
# MAIN TRAINING LOOPS
# =============================================================================

for model in $models
do
  for num_shot in $shot_numbers
  do
    for dataset in $datasets
    do
      # Configure training steps based on mode and dataset
      if [ "$DEBUG_MODE" = true ]; then
          # Debug mode: minimal steps
          num_steps=2
          eval_epoch_interval=1
          eval_before_training=True
          
      elif [ "$SIMPLE_HPC_TEST" = true ]; then
          # Simple HPC test: moderate steps for validation
          if [[ $dataset = *"ico_list"* ]]; then
              num_steps=100  # Quick but meaningful test
          else
              num_steps=100  # Default for other datasets
          fi
          eval_epoch_interval=20
          eval_before_training=False
          
      elif [ "$COMPREHENSIVE_MODE" = true ]; then
          # Production mode: full training steps
          if ! [[ $num_shot =~ $re ]]; then
              # Handle "all" case with dataset-specific steps
              if [[ $dataset = *"income"* ]]; then
                  num_steps=295000
              elif [[ $dataset = *"car"* ]]; then
                  num_steps=10500
              elif [[ $dataset = *"heart"* ]]; then
                  num_steps=5600
              elif [[ $dataset = *"diabetes"* ]]; then
                  num_steps=4700
              elif [[ $dataset = *"bank"* ]]; then
                  num_steps=272000
              elif [[ $dataset = *"blood"* ]]; then
                  num_steps=4520
              elif [[ $dataset = *"calhousing"* ]]; then
                  num_steps=124000
              elif [[ $dataset = *"creditg"* ]]; then
                  num_steps=6000
              elif [[ $dataset = *"jungle"* ]]; then
                  num_steps=270000
              elif [[ $dataset = *"ico_list"* ]]; then
                  num_steps=15000
              else
                  num_steps=10000  # Default for unknown datasets
              fi
          else
              # Few-shot case: calculate based on shots and batch size
              num_steps=$(( 30 * ($num_shot / $train_batch_size)))
          fi
          eval_epoch_interval=30
          eval_before_training=False
      fi

      for seed in $seeds
      do
        echo "=============================================="
        echo "Running: Model=$model, Dataset=$dataset, Shots=$num_shot, Seed=$seed"
        echo "Mode: $([ "$DEBUG_MODE" = true ] && echo "DEBUG" || ([ "$SIMPLE_HPC_TEST" = true ] && echo "SIMPLE HPC" || echo "COMPREHENSIVE"))"
        echo "Steps: $num_steps, Eval interval: $eval_epoch_interval"
        echo "=============================================="
        
        # Set environment variables
        export CUDA_VISIBLE_DEVICES=${cuda_device}
        export CONFIG_PATH="${PWD}/configs"
        export HF_HOME="${PWD}/.cache/huggingface"
        
        # Choose experiment name based on mode
        if [ "$DEBUG_MODE" = true ]; then
            exp_suffix="debug"
        elif [ "$SIMPLE_HPC_TEST" = true ]; then
            exp_suffix="hpc_test"
        else
            exp_suffix="pretrained100k"
        fi
        
        python -m src.pl_train \
          -c ${model}.json+ia3.json+global.json \
          -k dataset=${dataset} \
             load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" \
             num_steps=${num_steps} \
             num_shot=${num_shot} \
             exp_name=${model}_${dataset}_numshot${num_shot}_seed${seed}_ia3_${exp_suffix} \
             few_shot_random_seed=${seed} \
             seed=${seed} \
             allow_skip_exp=${allow_skip_exp} \
             eval_before_training=${eval_before_training} \
             eval_epoch_interval=${eval_epoch_interval} \
             batch_size=${train_batch_size} \
             grad_accum_factor=${grad_accum_factor} \
             lr=${lr}
      done
    done
  done
done

echo "=============================================="
echo "Training completed for all configurations!"
echo "=============================================="
