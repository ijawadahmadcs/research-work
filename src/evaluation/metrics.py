from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> Dict:
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    per_class_df = pd.DataFrame({
        "class": class_names,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1_score": f1_per_class,
        "support": support,
    })

    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class_df,
        "report_text": report_text,
    }


def metrics_to_percent_string(metrics: Dict) -> str:
    return (
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | "
        f"Precision (macro): {metrics['precision_macro'] * 100:.2f}% | "
        f"Recall (macro): {metrics['recall_macro'] * 100:.2f}% | "
        f"F1 (macro): {metrics['f1_macro'] * 100:.2f}%"
    )
