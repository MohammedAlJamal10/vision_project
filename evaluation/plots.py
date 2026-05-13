import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _ensure_parent_dir(save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _save_or_show(fig, save_path):
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _history_to_dict(history):
    if history is None:
        return {}

    if isinstance(history, dict):
        return history

    if hasattr(history, "history"):
        return history.history

    return {
        key: value
        for key, value in vars(history).items()
        if not key.startswith("_")
    }


def _first_available(history, keys):
    for key in keys:
        values = history.get(key)
        if values is not None:
            return values
    return None


def _curve_save_paths(save_path):
    if save_path is None:
        return None, None

    path = Path(save_path)
    if path.suffix:
        return (
            str(path.with_name(f"{path.stem}_accuracy{path.suffix}")),
            str(path.with_name(f"{path.stem}_loss{path.suffix}")),
        )

    os.makedirs(path.parent if path.parent != Path("") else ".", exist_ok=True)
    return f"{save_path}_accuracy.png", f"{save_path}_loss.png"


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    save_path=None
):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def plot_training_curves(
    history,
    title="Training Curves",
    save_path=None
):
    history = _history_to_dict(history)

    train_acc = _first_available(history, ["train_acc", "accuracy", "acc"])
    val_acc = _first_available(history, ["val_acc", "val_accuracy"])
    train_loss = _first_available(history, ["train_loss", "loss"])
    val_loss = _first_available(history, ["val_loss"])

    accuracy_path, loss_path = _curve_save_paths(save_path)

    if train_acc is not None or val_acc is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        if train_acc is not None:
            ax.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy")
        if val_acc is not None:
            ax.plot(range(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
        ax.set_title(f"{title} - Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, accuracy_path)

    if train_loss is not None or val_loss is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        if train_loss is not None:
            ax.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
        if val_loss is not None:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
        ax.set_title(f"{title} - Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, loss_path)


def plot_k_sweep(
    results,
    title="Validation Accuracy vs K",
    save_path=None
):
    if isinstance(results, dict):
        k_values = list(results.keys())
        val_accuracies = list(results.values())
    else:
        k_values = results["k"]
        val_accuracies = results["val_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, val_accuracies, marker="o")
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def plot_model_comparison(
    model_names,
    accuracies,
    title="Model Test Accuracy Comparison",
    save_path=None
):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(model_names, accuracies)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, max(1.0, float(np.max(accuracies)) * 1.15))

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{accuracy:.4f}",
            ha="center",
            va="bottom"
        )

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    _save_or_show(fig, save_path)
