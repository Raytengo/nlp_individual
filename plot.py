import os
import csv
import matplotlib.pyplot as plt

from config import LOG_DIR, PLOT_DIR


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def plot_loss_curves():
    """Plot training + validation loss curves and save as PNG."""
    train_rows = load_csv(os.path.join(LOG_DIR, "train_loss.csv"))
    val_rows = load_csv(os.path.join(LOG_DIR, "val_loss.csv"))

    train_epochs = [float(r["epoch"]) for r in train_rows]
    train_losses = [float(r["loss"]) for r in train_rows]

    val_epochs = [float(r["epoch"]) for r in val_rows]
    val_losses = [float(r["eval_loss"]) for r in val_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_losses, label="Training Loss", alpha=0.7)
    ax.plot(val_epochs, val_losses, "o-", label="Validation Loss",
            linewidth=2, markersize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = os.path.join(PLOT_DIR, "loss_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve saved to {save_path}")


if __name__ == "__main__":
    plot_loss_curves()
