"""
confusion_matrix.py
====================
Computes and plots the confusion matrix -- essential for a 3-class
medical problem because overall accuracy alone can hide a model that,
say, never correctly identifies PD (the minority class, only 31 videos
total in the base dataset) while still scoring well by being good at
the majority classes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.utils.config import OUTPUTS_DIR


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred, labels=range(len(class_names)))


def plot_confusion_matrix(
    cm: np.ndarray, class_names, backbone_name: str, normalize: bool = False,
) -> str:
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
        title = f"Normalized Confusion Matrix — {backbone_name}"
    else:
        fmt = "d"
        title = f"Confusion Matrix — {backbone_name}"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()

    out_dir = os.path.join(OUTPUTS_DIR, "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "normalized" if normalize else "counts"
    out_path = os.path.join(out_dir, f"{backbone_name}_confusion_matrix_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
