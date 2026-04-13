import argparse
import json
import os
import subprocess
import sys

from config import GRID_SEARCH_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run grid search, summarize results, and analyze errors for the best experiment."
    )
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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_script_command(script_name, args_map):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)]
    for key, value in args_map:
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_step(title, cmd):
    print(f"\n[PIPELINE] {title}")
    print("[PIPELINE] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_search_dir(output_root):
    if output_root:
        return os.path.join(output_root, "grid_search")
    return GRID_SEARCH_DIR


def load_best_experiment(search_dir):
    best_config_path = os.path.join(search_dir, "best_config.json")
    with open(best_config_path, "r", encoding="utf-8") as f:
        best = json.load(f)

    experiment_name = best.get("experiment_name")
    if not experiment_name:
        raise RuntimeError(f"No best experiment found in {best_config_path}")
    return experiment_name


def main():
    args = parse_args()

    grid_cmd = build_script_command("grid_search.py", [
        ("output-root", args.output_root),
        ("data-path", args.data_path),
        ("model-name", args.model_name),
        ("epochs", args.epochs),
        ("learning-rates", args.learning_rates),
        ("lora-r", args.lora_r),
        ("lora-alpha", args.lora_alpha),
        ("lora-dropout", args.lora_dropout),
        ("batch-size-per-gpu", args.batch_size_per_gpu),
        ("gradient-accumulation-steps", args.gradient_accumulation_steps),
        ("max-seq-length", args.max_seq_length),
        ("max-new-tokens", args.max_new_tokens),
        ("inference-batch-size", args.inference_batch_size),
        ("val-ratio", args.val_ratio),
        ("seed", args.seed),
        ("skip-existing", args.skip_existing),
    ])
    run_step("Running grid search", grid_cmd)

    analyze_grid_cmd = build_script_command("analyze_grid_search.py", [
        ("output-root", args.output_root),
        ("top-k", args.top_k),
    ])
    run_step("Analyzing grid search results", analyze_grid_cmd)

    best_experiment = load_best_experiment(get_search_dir(args.output_root))
    analyze_errors_cmd = build_script_command("analyze_errors.py", [
        ("experiment-name", best_experiment),
        ("output-root", args.output_root),
        ("top-k", args.top_k),
    ])
    run_step(f"Analyzing errors for best experiment: {best_experiment}", analyze_errors_cmd)

    print("\n[PIPELINE] Done")
    print(f"[PIPELINE] Best experiment: {best_experiment}")


if __name__ == "__main__":
    main()
