import argparse
import csv
import json
import os
from statistics import mean

from config import GRID_SEARCH_DIR


NUMERIC_FIELDS = [
    "epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "best_val_accuracy",
    "best_train_accuracy",
]

GROUP_FIELDS = [
    "epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze grid search results and summarize the best hyperparameters."
    )
    parser.add_argument("--results-csv")
    parser.add_argument("--output-root")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def get_results_csv(args):
    if args.results_csv:
        return args.results_csv
    if args.output_root:
        return os.path.join(args.output_root, "grid_search", "grid_search_results.csv")
    return os.path.join(GRID_SEARCH_DIR, "grid_search_results.csv")


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if value in ("", None):
                row[field] = None
            elif field in ("epochs", "lora_r", "lora_alpha"):
                row[field] = int(float(value))
            else:
                row[field] = float(value)
    return rows


def completed_rows(rows):
    return [
        row for row in rows
        if row.get("status") in {"completed", "skipped_existing"}
        and row.get("best_val_accuracy") is not None
    ]


def sort_by_val_accuracy(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["best_val_accuracy"],
            row.get("best_train_accuracy") or -1,
        ),
        reverse=True,
    )


def summarize_group(rows, field):
    grouped = {}
    for row in rows:
        key = row[field]
        grouped.setdefault(key, []).append(row["best_val_accuracy"])

    summary = []
    for key, values in grouped.items():
        summary.append({
            "value": key,
            "runs": len(values),
            "avg_val_accuracy": mean(values),
            "max_val_accuracy": max(values),
            "min_val_accuracy": min(values),
        })
    return sorted(summary, key=lambda item: item["avg_val_accuracy"], reverse=True)


def save_top_runs(rows, output_dir, top_k):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "grid_search_top_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "best_val_accuracy",
                "best_train_accuracy",
                "epochs",
                "learning_rate",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "best_checkpoint",
                "best_val_error_file",
                "loss_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows[:top_k])
    return path


def save_summary(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "grid_search_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def main():
    args = parse_args()
    results_csv = get_results_csv(args)
    rows = load_rows(results_csv)
    good_rows = sort_by_val_accuracy(completed_rows(rows))

    if not good_rows:
        raise SystemExit(f"No completed runs with valid accuracy found in {results_csv}")

    output_dir = os.path.dirname(results_csv)
    top_path = save_top_runs(good_rows, output_dir, args.top_k)

    summary = {
        "results_csv": results_csv,
        "total_runs": len(rows),
        "completed_runs": len(good_rows),
        "best_run": good_rows[0],
        "top_k": good_rows[:args.top_k],
        "parameter_effects": {
            field: summarize_group(good_rows, field)
            for field in GROUP_FIELDS
        },
    }
    summary_path = save_summary(summary, output_dir)

    print(f"[INFO] Top runs saved to {top_path}")
    print(f"[INFO] Analysis summary saved to {summary_path}")
    print("[INFO] Best run:")
    print(
        f"  {good_rows[0]['experiment_name']} | "
        f"val_acc={good_rows[0]['best_val_accuracy']:.4f} | "
        f"train_acc={good_rows[0]['best_train_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
