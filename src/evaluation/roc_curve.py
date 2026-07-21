import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.config import OUTPUTS_DIR


def plot_multiclass_roc(
    y_true: np.ndarray, y_pred_proba: np.ndarray, class_names, backbone_name: str,
) -> str:
    """
    y_true: 1D array of integer class indices.
    y_pred_proba: (N, num_classes) array of predicted probabilities (softmax output).
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, class_name in enumerate(class_names):
        ax.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
    ax.plot(
        fpr["micro"], tpr["micro"], linestyle="--", color="black",
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve — {backbone_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_dir = os.path.join(OUTPUTS_DIR, "roc_curves")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backbone_name}_roc_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
