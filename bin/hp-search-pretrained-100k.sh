#!/bin/bash
# =============================================================
# HP Search: find best lr, training steps, unlikely_loss
# for t03b + IA3 (pretrained 100k) on ICO binary classification.
# Optimises for PRAUC and f1_binary.
#
# USAGE
#   1. Run the search:
#        bash bin/hp-search-pretrained-100k.sh
#
#   2. After all runs finish, print the best params per num_shot:
#        bash bin/hp-search-pretrained-100k.sh --summarize
#
# Results land in exp_out/ (one sub-directory per run, same as the
# main script).  The --summarize mode ranks combos by mean PRAUC
# averaged over the two search seeds, and saves the winning combo
# per num_shot to configs/hp_best_params.json.
# =============================================================

# ============================================================
# FIXED SETTINGS  (keep these identical to few-shot-pretrained-100k.sh)
# ============================================================
allow_skip_exp=True
train_batch_size=4
grad_accum_factor=1
mc_loss=1        # keep contrastive loss on; not searched
length_norm=1    # short labels → length norm has little effect; not searched
dataset=ico
model=t03b

# ============================================================
# SEARCH GRID
# ============================================================

# Only search mid-range shots:
#   4/8/16  → params inherited from num_shot=4 result (see KNOWN_PARAMS below)
#   32/64   → already searched; winners hardcoded in KNOWN_PARAMS below
#   128     → searched here with reduced grid (steps_mult fixed at 60)
search_shots=(128)

# lr=1e-4 and lr=3e-3 eliminated by num_shot=4 result.
lr_values=("3e-4" "1e-3")

# steps_mult: 30→30→60 trend across shots 4→32→64 — fix at 60 for 128-shot.
# Saves half the runs vs searching both values.
steps_mult_values=(60)

# unlikely_loss=0 won at num_shot=4 and is consistent with class-imbalance
# hurting calibration — fixed at 0, not searched.
unlikely_loss=0

# Two seeds give enough signal without running the full 5-seed suite.
search_seeds=(42 0)

# ============================================================
# SUMMARIZE MODE  (bash bin/hp-search-pretrained-100k.sh --summarize)
# ============================================================
HP_PARAMS_FILE="/Users/work/t-few/configs/hp_best_params.json"

# Known best params for shots that are not searched.
# 4/8/16: inherited from num_shot=4 search result.
# 32/64:  already searched; winners hardcoded here so --summarize preserves them.
# 128:    will be filled by the current search run.
KNOWN_PARAMS='{
  "4":  {"lr": "3e-4", "steps_mult": 30, "unlikely_loss": 0, "avg_prauc": 0.3243, "avg_f1_binary": 0.3326, "num_seeds": 2, "source": "searched"},
  "8":  {"lr": "3e-4", "steps_mult": 30, "unlikely_loss": 0, "avg_prauc": null,   "avg_f1_binary": null,   "num_seeds": 0, "source": "inherited_from_4"},
  "16": {"lr": "3e-4", "steps_mult": 30, "unlikely_loss": 0, "avg_prauc": null,   "avg_f1_binary": null,   "num_seeds": 0, "source": "inherited_from_4"},
  "32": {"lr": "1e-3", "steps_mult": 30, "unlikely_loss": 0, "avg_prauc": 0.3506, "avg_f1_binary": 0.3726, "num_seeds": 2, "source": "searched"},
  "64": {"lr": "3e-4", "steps_mult": 60, "unlikely_loss": 0, "avg_prauc": 0.3538, "avg_f1_binary": 0.3767, "num_seeds": 2, "source": "searched"}
}'

if [[ "$1" == "--summarize" ]]; then
  export KNOWN_PARAMS="${KNOWN_PARAMS}"
  python3 - "${HP_PARAMS_FILE}" <<'PYEOF'
import json, os, sys, glob, re
from collections import defaultdict

OUT_FILE = sys.argv[1]
EXP_BASE = "/Users/work/t-few/exp_out"

pattern = os.path.join(EXP_BASE, "t03b_ico_hpsearch_*", "dev_scores.json")
files = sorted(glob.glob(pattern))

if not files:
    print("No HP search results found in exp_out/. Run the search first.")
    sys.exit(1)

print(f"Found {len(files)} completed run(s).\n")

# Exp-name format:
#   t03b_ico_hpsearch_lr{lr_tag}_mult{mult}_ul{ul}_numshot{ns}_seed{s}_ia3_pretrained100k
combo_by_shot = defaultdict(lambda: defaultdict(list))   # shot → combo → [(prauc, f1b)]

for fpath in files:
    exp_name = os.path.basename(os.path.dirname(fpath))
    m = re.search(
        r'_lr([^_]+)_mult(\d+)_ul(\d+)_numshot(\d+)_seed(\d+)_', exp_name
    )
    if not m:
        print(f"  [skip] could not parse: {exp_name}")
        continue
    lr_tag, mult, ul, ns, sd = m.groups()
    with open(fpath) as fh:
        try:
            d = json.load(fh)
        except json.JSONDecodeError:
            print(f"  [skip] bad JSON: {fpath}")
            continue
    prauc = d.get("PR", 0.0)
    f1b   = d.get("f1_binary", d.get("macro_f1", 0.0))  # fallback for older runs
    combo_by_shot[ns][(lr_tag, mult, ul)].append((prauc, f1b))

def decode_lr(tag):
    # "1em4" → "1e-4",  "3em3" → "3e-3"
    return tag.replace("em", "e-")

# merge known params, then fill searched shots, then inherit 128 from 64
best_params = json.loads(os.environ.get('KNOWN_PARAMS', '{}'))
for ns in combo_by_shot:
    combos = combo_by_shot[ns]
    best = max(combos,
               key=lambda k: sum(p for p, f in combos[k]) / len(combos[k]))
    vals = combos[best]
    avg_p = sum(v[0] for v in vals) / len(vals)
    avg_f = sum(v[1] for v in vals) / len(vals)
    lr_t, mult_t, ul_t = best
    best_params[ns] = {
        "lr":            decode_lr(lr_t),
        "steps_mult":    int(mult_t),
        "unlikely_loss": int(ul_t),
        "avg_prauc":     round(avg_p, 4),
        "avg_f1_binary": round(avg_f, 4),
        "num_seeds":     len(vals),
    }

with open(OUT_FILE, "w") as fh:
    json.dump(best_params, fh, indent=2)
print(f"Best params saved to: {OUT_FILE}\n")

# ---- per-num_shot winner table (printed for reference) ----
print(f"{'num_shot':<10} {'best_lr':<8} {'mult':<6} {'ul':<4} "
      f"{'PRAUC':>7}  {'f1_binary':>9}  {'seeds':>5}")
print("-" * 58)
for ns in sorted(best_params, key=lambda x: int(x)):
    p = best_params[ns]
    prauc_str = f"{p['avg_prauc']:>7.4f}" if p['avg_prauc'] is not None else f"{'n/a':>7}"
    f1_str    = f"{p['avg_f1_binary']:>9.4f}" if p['avg_f1_binary'] is not None else f"{'n/a':>9}"
    print(f"{ns:<10} {p['lr']:<8} {p['steps_mult']:<6} {p['unlikely_loss']:<4} "
          f"{prauc_str}  {f1_str}  {p['num_seeds']:>5}")

# ---- full ranking per shot (for manual inspection) ----
print("\n-- Full ranking per num_shot (best → worst, by avg PRAUC) --")
for ns in sorted(combo_by_shot, key=lambda x: int(x)):
    combos = combo_by_shot[ns]
    ranked = sorted(
        combos.items(),
        key=lambda kv: sum(p for p, f in kv[1]) / len(kv[1]),
        reverse=True
    )
    print(f"\n  num_shot={ns}:")
    print(f"  {'lr':<8} {'mult':<6} {'ul':<4} {'PRAUC':>7}  {'f1_binary':>9}  {'seeds':>5}")
    for (lr_t, mult_t, ul_t), vals in ranked[:8]:   # top 8
        avg_p = sum(v[0] for v in vals) / len(vals)
        avg_f = sum(v[1] for v in vals) / len(vals)
        print(f"  {decode_lr(lr_t):<8} {mult_t:<6} {ul_t:<4} "
              f"{avg_p:>7.4f}  {avg_f:>9.4f}  {len(vals):>5}")

print()
print("f1_binary is shown for reference only — ranking is by PRAUC.")
PYEOF
  exit 0
fi

# ============================================================
# TRAINING LOOP
# ============================================================
total_runs=$(( ${#search_shots[@]} \
             * ${#lr_values[@]} \
             * ${#steps_mult_values[@]} \
             * 2 ))
# (× 2 seeds; unlikely_loss is fixed at 0)

echo "HP search starting — ${total_runs} total runs"
echo "  shots     : ${search_shots[*]}"
echo "  lr        : ${lr_values[*]}"
echo "  steps_mult: ${steps_mult_values[*]}"
echo "  ul        : ${unlikely_loss} (fixed)"
echo "  seeds     : ${search_seeds[*]}"
echo ""

run_count=0

for num_shot in "${search_shots[@]}"
do
  for lr in "${lr_values[@]}"
  do
    # Filesystem-safe lr tag: replace 'e-' with 'em' (e.g. 1e-4 → 1em4)
    lr_tag="${lr//e-/em}"
    for steps_mult in "${steps_mult_values[@]}"
    do
      num_steps=$(( steps_mult * (num_shot / train_batch_size) ))
      # Guard: ensure at least steps_mult steps when num_shot < train_batch_size
      if [[ $num_steps -lt $steps_mult ]]; then
        num_steps=$steps_mult
      fi
      eval_epoch_interval=$steps_mult

      for unlikely_loss_val in 0
      do
        for seed in "${search_seeds[@]}"
        do
          run_count=$(( run_count + 1 ))
          exp_name="${model}_${dataset}_hpsearch_lr${lr_tag}_mult${steps_mult}_ul${unlikely_loss_val}_numshot${num_shot}_seed${seed}_ia3_pretrained100k"
          echo "[${run_count}/${total_runs}] ${exp_name}"

          CONFIG_PATH=/Users/work/t-few/configs HF_HOME=/Users/work/.cache/huggingface \
          python -m src.pl_train \
            -c ${model}.json+ia3.json+global.json \
            -k dataset=${dataset} \
               load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" \
               num_steps=${num_steps} \
               num_shot=${num_shot} \
               exp_name=${exp_name} \
               few_shot_random_seed=${seed} \
               seed=${seed} \
               allow_skip_exp=${allow_skip_exp} \
               eval_before_training=False \
               eval_epoch_interval=${eval_epoch_interval} \
               batch_size=${train_batch_size} \
               grad_accum_factor=${grad_accum_factor} \
               lr=${lr} \
               mc_loss=${mc_loss} \
               unlikely_loss=${unlikely_loss_val} \
               length_norm=${length_norm} \
               compute_strategy="none"
        done
      done
    done
  done
done

echo ""
echo "All ${total_runs} HP search runs complete."
echo "Run the following to see best params per num_shot:"
echo "  bash bin/hp-search-pretrained-100k.sh --summarize"
