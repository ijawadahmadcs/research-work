"""
plots.py
========
Training/validation accuracy and loss curves across both training
stages (frozen warmup + fine-tuning), stitched together into one
continuous x-axis of epochs -- mirroring Figs. 5 & 6 of the base paper.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt

from src.utils.config import OUTPUTS_DIR


def _concat_history(hist1, hist2, key):
    values = list(hist1.history.get(key, []))
    if hist2 is not None:
        values += list(hist2.history.get(key, []))
    return values


def plot_training_curves(history_stage1, history_stage2: Optional[object], backbone_name: str) -> str:
    acc = _concat_history(history_stage1, history_stage2, "accuracy")
    val_acc = _concat_history(history_stage1, history_stage2, "val_accuracy")
    loss = _concat_history(history_stage1, history_stage2, "loss")
    val_loss = _concat_history(history_stage1, history_stage2, "val_loss")
    stage1_len = len(history_stage1.history.get("loss", []))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(stage1_len - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune start")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(stage1_len - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune start")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(f"Training history — {backbone_name}")
    fig.tight_layout()

    out_dir = os.path.join(OUTPUTS_DIR, "training_curves")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backbone_name}_training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
