from glob import glob
import json
from collections import defaultdict
from scipy.stats import iqr
from numpy import median, mean, std
import os
import argparse


def make_result_table(args):
    def collect_exp_scores(exp_name_template, datasets):
        print("=" * 80)
        all_files = glob(
            os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), exp_name_template, "dev_scores.json")
        )
        print(f"Find {len(all_files)} experiments fit into {exp_name_template}")

        def read_last_eval(fname):
            with open(fname) as f:
                e = json.loads(f.readlines()[-1])
            return e

        acc_by_dataset = defaultdict(lambda: list())

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
            model = parts[0]
            dataset = "_".join(parts[1:numshot_idx])
            numshot = parts[numshot_idx].replace("numshot", "") or "unknown"
            return model, dataset, numshot, parts[seed_idx], "_".join(parts[seed_idx + 1 :])

        for fname in all_files:
            result = read_last_eval(fname)
            model, dataset, numshot, seed, spec = parse_expname(fname)
            if args.metric not in result:
                continue
            acc_by_dataset[(dataset, numshot)].append(result[args.metric])

        def result_str(acc_list):
            if len(acc_list) > 1:
                return f"{mean(acc_list):.2f} ({std(acc_list):.2f})"
            else:
                return f"{acc_list[0]:.2f}"

        def numshot_key(x):
            try:
                return int(x)
            except:
                return x
        
        numshot_values = sorted(set(k[1] for k in acc_by_dataset.keys()), key=numshot_key)
        dataset_keys = [(d, ns) for d in datasets for ns in numshot_values]
        outputs = [result_str(acc_by_dataset.get(k, [])) if acc_by_dataset.get(k) else "NA" for k in dataset_keys]
        print(", ".join([f"{ds}_numshot{ns}: {val}" for (ds, ns), val in zip(dataset_keys, outputs)]))
        return ",".join(outputs), dataset_keys

    header = ["template"]
    all_rows = []
    for exp_name_template in args.exp_name_templates:
        row_vals, dataset_keys = collect_exp_scores(exp_name_template, args.datasets)
        if len(header) == 1:
            header.extend([f"{ds}_numshot{ns}" for ds, ns in dataset_keys])
        all_rows.append(f"{exp_name_template}," + row_vals)
    
    csv_lines = [",".join(header)] + all_rows

    output_fname = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), f"summary_{args.metric}.csv")
    with open(output_fname, "w") as f:
        for line in csv_lines:
            f.write(line + "\n")
    print(f"Save result to {output_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True)
    parser.add_argument(
        "-d", "--datasets", default="copa,h-swag,storycloze,winogrande,wsc,wic,rte,cb,anli-r1,anli-r2,anli-r3"
    )
    parser.add_argument(
        "-m", "--metric", default="AUC", help="Metric to report (AUC, accuracy, macro_f1, micro_f1, PR)"
    )
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_result_table(args)
