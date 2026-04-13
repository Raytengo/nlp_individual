import argparse
import csv
import os

import matplotlib.pyplot as plt

from config import build_run_config


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves for one experiment.")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--output-root")
    return parser.parse_args()


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_loss_curves(run_config):
    """Plot training + validation loss curves and save as PNG."""
    train_rows = load_csv(os.path.join(run_config.log_dir, "train_loss.csv"))
    val_rows = load_csv(os.path.join(run_config.log_dir, "val_loss.csv"))

    train_epochs = [float(r["epoch"]) for r in train_rows]
    train_losses = [float(r["loss"]) for r in train_rows]

    val_epochs = [float(r["epoch"]) for r in val_rows]
    val_losses = [float(r["eval_loss"]) for r in val_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_losses, label="Training Loss", alpha=0.7)
    ax.plot(
        val_epochs,
        val_losses,
        "o-",
        label="Validation Loss",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training and Validation Loss Curve ({run_config.experiment_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(run_config.plot_dir, exist_ok=True)
    fig.savefig(run_config.loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve saved to {run_config.loss_plot_path}")


if __name__ == "__main__":
    args = parse_args()
    run_config = build_run_config(
        experiment_name=args.experiment_name,
        output_root=args.output_root,
    )
    plot_loss_curves(run_config)
