#!/bin/bash
allow_skip_exp=True
eval_before_training=True
balanced_ibc=True

# ===== QUICK TEST PARAMETERS (comment these lines for production) =====
# train_batch_size=1  # QUICK_TEST: change to 4 for production
train_batch_size=4
grad_accum_factor=1

lr=0.003
re='^[0-9]+$'

# macOS: No CUDA, use MPS or CPU (handled automatically by PyTorch Lightning)
# cuda_device is not used on macOS

# Few-shot semantics (num_shot):
# - num_shot is the TOTAL number of training examples used for fine-tuning.
# - For custom categorical tasks (income, car, heart, diabetes, creditg, bank, blood, jungle, calhousing, ico),
#   sampling is BALANCED across labels: approximately num_shot/num_labels per class (remainder goes to one class).
# - For other datasets (e.g., GLUE/T0-style), selection is unbalanced: first num_shot shuffled examples from train.
# - Special values: num_shot='all' uses the full train split; num_shot=0 or '0' uses none.
# - Few-shot selections are cached under data/few_shot/<dataset>/<num_shot>_shot/<seed>_seed.jsonl.

# Set adaptively
# num_steps=1  # QUICK_TEST: change to 0 or appropriate value for production
num_steps=0
# eval_epoch_interval=1  # QUICK_TEST: change to 0 or appropriate value for production
eval_epoch_interval=0
# ===== END QUICK TEST PARAMETERS =====

for model in 't03b' # 't011b'
do
  # For zero-shot set to '0', for all to 'all'
  # for num_shot in 4 8 16 32 64 128 256 512
  for num_shot in 0
  do
    # Datasets: car, income, heart, diabetes, jungle, bank, blood, calhousing, creditg, jungle
    # Run all serializations for car
    # for dataset in car car_list car_list_permuted car_list_shuffled car_list_values car_gpt car_t0 car_ttt ico
    for dataset in ico
    do
      # Zero-shot
      # eval_before_training=True
      # num_steps=0
      # Few-shot
      eval_before_training=False
      # QUICK_TEST: override to 1 step (comment out these 2 lines for production)
      # num_steps=1
      # eval_epoch_interval=1
      # Production versions (uncomment below):
      num_steps=$(( 30 * ($num_shot / $train_batch_size)))
      eval_epoch_interval=30


      # For all run
      if ! [[ $num_shot =~ $re ]]; then
        if [[ $dataset = *"income"* ]]; then
          num_steps=295000
        fi
        if [[ $dataset = *"car"* ]]; then
          num_steps=10500
        fi
        if [[ $dataset = *"heart"* ]]; then
          num_steps=5600
        fi
        if [[ $dataset = *"diabetes"* ]]; then
          num_steps=4700
        fi
        if [[ $dataset = *"bank"* ]]; then
          num_steps=272000
        fi
        if [[ $dataset = *"blood"* ]]; then
          num_steps=4520
        fi
        if [[ $dataset = *"calhousing"* ]]; then
          num_steps=124000
        fi
        if [[ $dataset = *"creditg"* ]]; then
          num_steps=6000
        fi
        if [[ $dataset = *"jungle"* ]]; then
          num_steps=270000
        fi
      fi

      # for seed in 42 1024 0 1 32
      for seed in 42 1024 0 1 32
      do
        # ICO label strategies: all (default) | high_only | low_only
        # Each strategy filters rows and redefines the positive class, so N and
        # class balance differ — the exp_name encodes the strategy for traceability.
        for ico_label_strategy in all
        do
          # macOS setup: Use local paths, no CUDA_VISIBLE_DEVICES
          CONFIG_PATH=/Users/work/t-few/configs HF_HOME=/Users/work/.cache/huggingface \
          python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
          exp_name=${model}_${dataset}_${ico_label_strategy}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
          batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr} compute_strategy="none" ico_label_strategy=${ico_label_strategy}
        done
      done
    done
  done
done
