#!/bin/bash
# =============================================================
# HP-tuned few-shot fine-tuning (t03b, IA3, pretrained 100k)
#
# This script is identical to few-shot-pretrained-100k.sh except
# that lr, steps_mult, and unlikely_loss are loaded automatically
# per num_shot from the HP search results file produced by:
#
#   bash bin/few-shot-hp-100k.sh            # run search
#   bash bin/few-shot-hp-100k.sh --summarize  # save best params
#
# If configs/hp_best_params.json does not exist, the defaults below
# are used and a warning is printed.
# =============================================================

allow_skip_exp=True
eval_before_training=True
balanced_ibc=True

train_batch_size=4
grad_accum_factor=1

re='^[0-9]+$'

# Path written by hp-search-pretrained-100k.sh --summarize
HP_PARAMS_FILE="/Users/work/t-few/configs/hp_best_params.json"

# Fallback defaults (same as few-shot-pretrained-100k.sh)
DEFAULT_LR="3e-3"
DEFAULT_STEPS_MULT=30
DEFAULT_UNLIKELY_LOSS=1

if [[ ! -f "${HP_PARAMS_FILE}" ]]; then
  echo "WARNING: ${HP_PARAMS_FILE} not found — using defaults (lr=${DEFAULT_LR}, mult=${DEFAULT_STEPS_MULT}, ul=${DEFAULT_UNLIKELY_LOSS})"
  echo "         Run bin/hp-search-pretrained-100k.sh first to generate it."
fi

# Set adaptively
num_steps=0
eval_epoch_interval=0

for model in 't03b' # 't011b'
do
  # for num_shot in 4 8 16 32 64 128
  for num_shot in 4 8 16 32 64 128
  do
    for dataset in ico
    do
      eval_before_training=False

      # ---- Load per-num_shot HP params ----
      if [[ -f "${HP_PARAMS_FILE}" ]]; then
        lr=$(python3 -c "
import json
d = json.load(open('${HP_PARAMS_FILE}'))
p = d.get(str(${num_shot}), {})
print(p.get('lr', '${DEFAULT_LR}'))
")
        steps_mult=$(python3 -c "
import json
d = json.load(open('${HP_PARAMS_FILE}'))
p = d.get(str(${num_shot}), {})
print(p.get('steps_mult', ${DEFAULT_STEPS_MULT}))
")
        unlikely_loss=$(python3 -c "
import json
d = json.load(open('${HP_PARAMS_FILE}'))
p = d.get(str(${num_shot}), {})
print(p.get('unlikely_loss', ${DEFAULT_UNLIKELY_LOSS}))
")
      else
        lr=${DEFAULT_LR}
        steps_mult=${DEFAULT_STEPS_MULT}
        unlikely_loss=${DEFAULT_UNLIKELY_LOSS}
      fi

      num_steps=$(( steps_mult * ($num_shot / $train_batch_size) ))
      # Guard: ensure at least steps_mult steps when num_shot < train_batch_size
      if [[ $num_steps -lt $steps_mult ]]; then
        num_steps=$steps_mult
      fi
      eval_epoch_interval=${steps_mult}

      echo "num_shot=${num_shot}  lr=${lr}  steps_mult=${steps_mult}  num_steps=${num_steps}  unlikely_loss=${unlikely_loss}"

      # for seed in 42 1024 0 1 32
      for seed in 42 1024 0 1 32
      do
        for ico_label_strategy in low_only
        do
          CONFIG_PATH=/Users/work/t-few/configs HF_HOME=/Users/work/.cache/huggingface \
          python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
          exp_name=${model}_${dataset}_${ico_label_strategy}_numshot${num_shot}_seed${seed}_ia3_hptuned few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
          batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr} unlikely_loss=${unlikely_loss} compute_strategy="none" ico_label_strategy=${ico_label_strategy}
        done
      done
    done
  done
done
