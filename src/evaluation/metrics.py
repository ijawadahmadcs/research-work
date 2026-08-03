"""
metrics.py
==========
Evaluation metrics for the MHI gait classifier.

Metrics reported
----------------
- Accuracy
- Weighted Precision, Recall, F1  (appropriate for imbalanced datasets;
  weights each class by its support in the true labels)
- Macro Precision, Recall, F1     (treats every class equally; useful for
  detecting collapse on minority classes)
- Per-class Precision, Recall, F1, Support
- Macro ROC-AUC (one-vs-rest, requires probability outputs, not hard labels)
- Cohen's Kappa κ                 (agreement corrected for chance; useful
  for very small test sets where accuracy is noisy)
- Classification report (sklearn text format)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute the full suite of classification metrics.

    Parameters
    ----------
    y_true       : 1-D integer array of true class indices.
    y_pred       : 1-D integer array of predicted class indices.
    class_names  : sequence of class name strings (used for the report).
    y_pred_proba : (N, C) float array of softmax probabilities.  If provided,
                   macro ROC-AUC is included in the output; otherwise it is
                   omitted (set to NaN).

    Returns
    -------
    dict with scalar metrics and a per-class DataFrame.
    """
    n_classes = len(class_names)
    accuracy = accuracy_score(y_true, y_pred)

    # ── Macro averages ─────────────────────────────────────────────────────
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ── Weighted averages (accounts for class imbalance) ───────────────────
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ── Per-class metrics ──────────────────────────────────────────────────
    prec_pc, rec_pc, f1_pc, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": prec_pc,
            "recall": rec_pc,
            "f1_score": f1_pc,
            "support": support,
        }
    )

    # ── Cohen's Kappa ──────────────────────────────────────────────────────
    kappa = cohen_kappa_score(y_true, y_pred)

    # ── ROC-AUC (requires probability outputs) ────────────────────────────
    roc_auc = float("nan")
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(
                y_true, y_pred_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            # Can happen if a class has no positive examples in y_true
            roc_auc = float("nan")

    # ── Classification report (sklearn text) ──────────────────────────────
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": accuracy,
        # Macro
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        # Weighted
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        # Other scalars
        "roc_auc_macro": roc_auc,
        "cohen_kappa": kappa,
        # Structured
        "per_class": per_class_df,
        "report_text": report_text,
    }


def metrics_to_percent_string(metrics: Dict) -> str:
    """One-line summary suitable for logging."""
    roc = metrics.get("roc_auc_macro", float("nan"))
    roc_str = f"{roc * 100:.2f}%" if not np.isnan(roc) else "N/A"
    kappa = metrics.get("cohen_kappa", float("nan"))
    kappa_str = f"{kappa:.4f}" if not np.isnan(kappa) else "N/A"
    return (
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | "
        f"F1 (weighted): {metrics['f1_weighted'] * 100:.2f}% | "
        f"F1 (macro): {metrics['f1_macro'] * 100:.2f}% | "
        f"ROC-AUC (macro): {roc_str} | "
        f"Kappa: {kappa_str}"
    )
