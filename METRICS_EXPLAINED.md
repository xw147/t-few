# T-Few Metrics Explanation

This document explains the evaluation metrics computed during training and validation in the T-Few project.

## Metrics Overview

All metrics are computed in `src/data/dataset_readers.py` (specifically in `CustomCategoricalReader.compute_metric()`) and written to `dev_scores.json` files.

### Classification Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **AUC** | ROC-AUC (Receiver Operating Characteristic - Area Under Curve) score | `sklearn.metrics.roc_auc_score` |
| **PR** | Precision-Recall AUC | Custom `pr_auc_score()` function (line 345 in dataset_readers.py) |
| **micro_f1** | Micro-averaged F1 score (aggregates contributions of all classes) | `sklearn.metrics.f1_score` with `average='micro'` |
| **macro_f1** | Macro-averaged F1 score (unweighted mean of per-class F1) | `sklearn.metrics.f1_score` with `average='macro'` |
| **accuracy** | Simple classification accuracy (correct predictions / total examples) | Computed directly from predictions |
| **sensitivity** | True positive rate (recall) — TP / (TP + FN). For multi-class: macro-averaged per-class recall | `sklearn.metrics.recall_score` |
| **specificity** | True negative rate — TN / (TN + FP). For multi-class: macro-averaged per-class specificity | Computed from `sklearn.metrics.confusion_matrix` |
| **precision** | Positive predictive value — TP / (TP + FP). For multi-class: macro-averaged per-class precision | `sklearn.metrics.precision_score` |

### Metadata Metrics

| Metric | Description |
|--------|-------------|
| **num** | Number of examples evaluated in this run |
| **num_steps** | Global training step at which the best validation model was saved |

### Model Confidence Metrics

These metrics provide insight into the model's confidence when making predictions for multiple-choice tasks.

#### **score_gt** (Ground Truth Score)
- **Definition**: Average cross-entropy loss when the model generates the **correct answer**
- **Range**: Lower is better (typically 0.5-2.0 for well-trained models)
- **Interpretation**: 
  - Low score (~0.5-1.0): Model is very confident in correct answers
  - High score (~2.0+): Model struggles to generate correct answers
- **Calculation**: For each example, compute the sum of token-level cross-entropy losses for generating the ground truth answer choice (optionally length-normalized)

#### **score_cand** (Best Candidate Score)
- **Definition**: Average cross-entropy loss for the **best incorrect answer** (the most plausible wrong choice)
- **Range**: Should be higher than `score_gt` for good performance
- **Interpretation**:
  - High score (>>score_gt): Model clearly distinguishes correct from incorrect answers
  - Low score (≈score_gt): Model is confused and considers wrong answers plausible
- **Calculation**: After excluding the ground truth, find the minimum (best) score among all remaining wrong answer choices

#### **Confidence Margin**
The gap between `score_cand` and `score_gt` indicates the model's ability to discriminate:
- **Large gap** (`score_cand - score_gt > 1.0`): Strong discrimination, high confidence
- **Small gap** (`score_cand - score_gt < 0.5`): Weak discrimination, low confidence
- **Negative gap** (`score_cand < score_gt`): Model prefers wrong answers (problematic)


## Data Splits: Understanding "dev" vs "test"

### Why the Confusing Naming?

The code uses confusing terminology. Here's what actually happens:

**For custom tasks (ico, car, heart, diabetes, etc.):**

In `CustomCategoricalReader.read_orig_dataset()` (lines 254-268 in dataset_readers.py):

```python
data = orig_data.train_test_split(test_size=0.20, seed=self.config.seed)
data2 = data['test'].train_test_split(test_size=0.50, seed=self.config.seed)
dataset_dict = DatasetDict({
    'train': data['train'],              # 80% of original data
    'validation': concatenate_datasets([data2['train'], data2['test']]),  # 20% of original
    'test': Dataset.from_dict({'note': [], 'label': []})  # Empty!
})
```

### Actual Data Split Usage

| Split Name | How It's Used | What It Contains |
|-----------|---------------|-----------------|
| **Train** | Used to train the model | 80% of original data (or few-shot subset if `num_shot` specified) |
| **Validation** | Held-out evaluation (called "dev") | 20% of original data, completely held-out |
| **Test** | Reserved but empty | Empty dataset for custom tasks |

### Key Facts

✅ **The "validation" split (reported in `dev_scores.json`) IS your actual test set:**
- Never used during training
- Never used for hyperparameter tuning (all hyperparameters are fixed: learning rate, batch size, num_steps)
- Completely held-out before model training begins
- Represents true unseen performance

✅ **No hyperparameter optimization occurs:**
- All configurations are set in advance (shell script passes fixed values)
- The `best_eval_model_metric` in `EncoderDecoder.on_validation_epoch_end()` (line 318) tracks the best checkpoint but doesn't tune hyperparameters
- The best model is selected based on validation AUC, but the hyperparameters themselves remain fixed

### Bottom Line

**`dev_scores.json` reports legitimate held-out test set results.** The metrics from `get_results_table.py` are valid final evaluation metrics, not validation metrics in the traditional ML sense.

The terminology is confusing, but functionally:
- **"validation" set** = Your actual test set (20% held-out, never trained on, not used for tuning)
- **"test" set** = Empty for custom tasks
- **"train" set** = Either 80% of data OR your few-shot subset

## Generating Summary Tables with `get_result_table.py`

After running experiments, use `src/scripts/get_result_table.py` to aggregate results across multiple runs into a CSV summary.

### Usage

```bash
python -m src.scripts.get_result_table \
  -e "EXPERIMENT_NAME_PATTERN" \
  -d "DATASET_NAMES" \
  -m "METRIC_NAME"
```

### Parameters

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--exp_name_templates` | `-e` | **Required.** Glob pattern(s) to match experiment directories. Use wildcards (`*`) to match multiple experiments. Multiple patterns can be comma-separated. | `"t03b_ico_list_*_ia3_pretrained100k"` |
| `--datasets` | `-d` | Dataset name(s) to include in the summary. Must match the dataset names in your experiment directory names. Comma-separated for multiple datasets. | `"ico_list"` or `"copa,rte,wic"` |
| `--metric` | `-m` | The metric to aggregate and report. Must match a key in `dev_scores.json`. Default: `AUC` | `"AUC"` or `"accuracy"` or `"macro_f1"` |

### Example Usage

**Single dataset with AUC metric:**
```bash
python -m src.scripts.get_result_table \
  -e "t03b_ico_list_*_ia3_pretrained100k" \
  -d "ico_list" \
  -m "PR"
```

**Multiple datasets with accuracy metric:**
```bash
python -m src.scripts.get_result_table \
  -e "t03b_*_finetune" \
  -d "copa,rte,wic,cb" \
  -m "accuracy"
```






