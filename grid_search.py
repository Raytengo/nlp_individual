import argparse
import csv
import itertools
import json
import os
import subprocess
import sys

from config import GRID_SEARCH_DIR, build_run_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run grid search over training hyperparameters.")
    parser.add_argument("--output-root")
    parser.add_argument("--data-path")
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", default="1,2")
    parser.add_argument("--learning-rates", default="2e-4,1e-4,5e-5")
    parser.add_argument("--lora-r", default="8,16")
    parser.add_argument("--lora-alpha", default="16,32")
    parser.add_argument("--lora-dropout", default="0.05,0.1")
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def format_float_token(value):
    return str(value).replace(".", "p")


def build_experiment_name(params):
    return (
        f"ep{params['epochs']}"
        f"_lr{format_float_token(params['learning_rate'])}"
        f"_r{params['lora_r']}"
        f"_a{params['lora_alpha']}"
        f"_drop{format_float_token(params['lora_dropout'])}"
    )


def build_train_command(params, args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
        "--experiment-name",
        params["experiment_name"],
    ]

    for key, value in [
        ("data-path", args.data_path),
        ("output-root", args.output_root),
        ("model-name", args.model_name),
        ("batch-size-per-gpu", args.batch_size_per_gpu),
        ("gradient-accumulation-steps", args.gradient_accumulation_steps),
        ("max-seq-length", args.max_seq_length),
        ("max-new-tokens", args.max_new_tokens),
        ("inference-batch-size", args.inference_batch_size),
        ("val-ratio", args.val_ratio),
        ("seed", args.seed),
    ]:
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    cmd.extend([
        "--epochs", str(params["epochs"]),
        "--learning-rate", str(params["learning_rate"]),
        "--lora-r", str(params["lora_r"]),
        "--lora-alpha", str(params["lora_alpha"]),
        "--lora-dropout", str(params["lora_dropout"]),
    ])
    return cmd


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(search_dir, rows, best_row):
    os.makedirs(search_dir, exist_ok=True)
    csv_path = os.path.join(search_dir, "grid_search_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "status",
                "epochs",
                "learning_rate",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "best_checkpoint",
                "best_val_accuracy",
                "best_train_accuracy",
                "best_val_error_file",
                "loss_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_path = os.path.join(search_dir, "best_config.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print(f"[INFO] Grid search results saved to {csv_path}")
    print(f"[INFO] Best config saved to {best_path}")


def main():
    args = parse_args()
    search_dir = GRID_SEARCH_DIR
    if args.output_root:
        search_dir = os.path.join(args.output_root, "grid_search")

    epochs_list = parse_int_list(args.epochs)
    lr_list = parse_float_list(args.learning_rates)
    lora_r_list = parse_int_list(args.lora_r)
    lora_alpha_list = parse_int_list(args.lora_alpha)
    lora_dropout_list = parse_float_list(args.lora_dropout)

    rows = []
    best_row = None

    for epochs, lr, lora_r, lora_alpha, lora_dropout in itertools.product(
        epochs_list,
        lr_list,
        lora_r_list,
        lora_alpha_list,
        lora_dropout_list,
    ):
        params = {
            "epochs": epochs,
            "learning_rate": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
        params["experiment_name"] = build_experiment_name(params)
        run_config = build_run_config(
            experiment_name=params["experiment_name"],
            output_root=args.output_root,
        )

        summary_path = run_config.evaluation_summary_path
        if args.skip_existing and os.path.exists(summary_path):
            print(f"[INFO] Skipping existing experiment: {params['experiment_name']}")
            summary = load_summary(summary_path)
            status = "skipped_existing"
        else:
            cmd = build_train_command(params, args)
            print(f"\n[INFO] Running experiment: {params['experiment_name']}")
            print("[INFO] Command:", " ".join(cmd))
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                rows.append({
                    "experiment_name": params["experiment_name"],
                    "status": f"failed_{result.returncode}",
                    "epochs": epochs,
                    "learning_rate": lr,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "best_checkpoint": None,
                    "best_val_accuracy": None,
                    "best_train_accuracy": None,
                    "best_val_error_file": None,
                    "loss_plot": None,
                })
                continue
            if not os.path.exists(summary_path):
                rows.append({
                    "experiment_name": params["experiment_name"],
                    "status": "missing_summary",
                    "epochs": epochs,
                    "learning_rate": lr,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "best_checkpoint": None,
                    "best_val_accuracy": None,
                    "best_train_accuracy": None,
                    "best_val_error_file": None,
                    "loss_plot": None,
                })
                continue
            summary = load_summary(summary_path)
            status = "completed"

        row = {
            "experiment_name": params["experiment_name"],
            "status": status,
            "epochs": epochs,
            "learning_rate": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "best_checkpoint": summary["best_checkpoint"],
            "best_val_accuracy": summary["best_val_accuracy"],
            "best_train_accuracy": summary["best_train_accuracy"],
            "best_val_error_file": summary["best_val_error_file"],
            "loss_plot": run_config.loss_plot_path,
        }
        rows.append(row)

        if best_row is None or (
            row["best_val_accuracy"] is not None
            and row["best_val_accuracy"] > best_row["best_val_accuracy"]
        ):
            best_row = row

    if best_row is None:
        best_row = {"status": "no_successful_runs"}
    save_results(search_dir, rows, best_row)


if __name__ == "__main__":
    main()
