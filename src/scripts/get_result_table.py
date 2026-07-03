from glob import glob
import json
from collections import defaultdict
from numpy import mean, std
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from path_config import EXP_OUTPUT_PATH, SUMMARY_OUTPUT_PATH


METRIC_NAMES = [
    'precision', 'recall', 'f1_binary', 'micro_f1', 'auprc',
    'specificity', 'auroc', 'macro_f1', 'accuracy'
]

# Maps dev_scores.json keys → unified internal metric names
TFEW_METRIC_MAP = {
    'precision': 'precision',
    'recall': 'recall',
    'sensitivity': 'recall',   # sensitivity == recall
    'PR': 'auprc',
    'AUC': 'auroc',
    'specificity': 'specificity',
    'f1_binary': 'f1_binary',
    'micro_f1': 'micro_f1',
    'macro_f1': 'macro_f1',
    'accuracy': 'accuracy',
}


def make_result_table(args):
    metrics_to_show = args.metrics if args.metrics else METRIC_NAMES

    def parse_expname(fname):
        expname = fname.split("/")[-2]
        parts = expname.split("_")
        seed_idx = numshot_idx = None
        for i, part in enumerate(parts):
            if part.startswith("seed"):
                seed_idx = i
            if part.startswith("numshot"):
                numshot_idx = i
        if seed_idx is None:
            seed_idx = len(parts) - 1
        if numshot_idx is None:
            numshot_idx = seed_idx
        dataset = "_".join(parts[1:numshot_idx])
        numshot = parts[numshot_idx].replace("numshot", "") or "unknown"
        return dataset, numshot

    def collect_exp_scores(exp_name_template):
        print("=" * 80)
        all_files = glob(
            os.path.join(EXP_OUTPUT_PATH, exp_name_template, "dev_scores.json")
        )
        print(f"Find {len(all_files)} experiments fit into {exp_name_template}")

        # raw_by_shot[numshot][metric] = [values across seeds]
        raw_by_shot = defaultdict(lambda: defaultdict(list))
        dataset_name = None

        for fname in all_files:
            with open(fname) as f:
                lines = f.readlines()
                if not lines:
                    continue
                result = json.loads(lines[-1])
            dataset, numshot = parse_expname(fname)
            if dataset_name is None:
                dataset_name = dataset
            seen_unified = set()
            for tfew_key, unified_key in TFEW_METRIC_MAP.items():
                if tfew_key in result and unified_key not in seen_unified:
                    raw_by_shot[numshot][unified_key].append(result[tfew_key])
                    seen_unified.add(unified_key)

        def numshot_key(x):
            try:
                return int(x)
            except Exception:
                return x

        numshot_values = sorted(raw_by_shot.keys(), key=numshot_key)
        label = exp_name_template

        # rows[metric][numshot] = formatted string
        rows = {}
        for metric in METRIC_NAMES:
            rows[metric] = {}
            for ns in numshot_values:
                vals = raw_by_shot[ns].get(metric, [])
                if len(vals) > 1:
                    rows[metric][ns] = f"{mean(vals):.4f} ({std(vals):.4f})"
                elif len(vals) == 1:
                    rows[metric][ns] = f"{vals[0]:.4f}"
                else:
                    rows[metric][ns] = ""

        return label, dataset_name, rows, numshot_values

    all_results = []
    all_numshots = set()
    for exp_name_template in args.exp_name_templates:
        label, dataset_name, rows, numshot_values = collect_exp_scores(exp_name_template)
        all_results.append((label, dataset_name, rows))
        all_numshots.update(numshot_values)

    def numshot_key(x):
        try:
            return int(x)
        except Exception:
            return x

    shot_order = sorted(all_numshots, key=numshot_key)
    col_headers = ["experiment", "dataset"] + [f"numshot{ns}" for ns in shot_order]

    output_fname = os.path.join(
        SUMMARY_OUTPUT_PATH,
        args.output if args.output else "summary_all_metrics.csv"
    )
    with open(output_fname, "w", encoding="utf-8") as f:
        for i, metric in enumerate(metrics_to_show):
            if i > 0:
                f.write("\n")
            f.write(f"# Metric: {metric}\n")
            f.write(",".join(col_headers) + "\n")
            for label, dataset_name, rows in all_results:
                row = [label, dataset_name or ""] + [rows[metric].get(ns, "") for ns in shot_order]
                f.write(",".join(row) + "\n")

    print(f"\nSaved summary to: {output_fname}")
    print(f"Metrics included: {', '.join(metrics_to_show)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary CSV from experiment results.")
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True,
                        help="Comma-separated glob patterns matching folders under exp_out/.")
    parser.add_argument(
        "-d", "--datasets", default="copa,h-swag,storycloze,winogrande,wsc,wic,rte,cb,anli-r1,anli-r2,anli-r3"
    )
    parser.add_argument(
        "-m", "--metrics", nargs="*", default=None,
        help=f"Which metrics to include. Default: all. Choices: {METRIC_NAMES}"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV filename (saved under SUMMARY_OUTPUT_PATH). Default: summary_all_metrics.csv"
    )
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_result_table(args)
