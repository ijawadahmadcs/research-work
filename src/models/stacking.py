"""
stacking.py
===========
Stacking meta-classifier for the MHI gait classifier ensemble.

Architecture
------------
Base models (MobileNet, EfficientNet, DenseNet, VGG16, NASNet, Swin)
each output a (N, 3) probability matrix.  These matrices are concatenated
horizontally → a (N, 3*M) feature matrix fed to a meta-classifier.

Meta-classifier options
-----------------------
- LogisticRegression (default, robust, low variance)
- SVM with RBF kernel (sometimes better with a small feature space)
- MLP (more flexible, may need more validation samples)

Data hygiene
------------
- The meta-classifier is fitted ONLY on validation-set probability outputs.
- It is evaluated ONLY on test-set probability outputs.
- No test-set data is seen during meta-classifier training.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.models.ensemble import load_all_predictions
from src.utils.config import DEFAULT_CONFIG, OUTPUTS_DIR, RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
VAL_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, "val")
ENSEMBLE_RESULTS_DIR = os.path.join(RESULTS_DIR, "ensemble")


def build_stacking_features(
    data: Dict[str, Dict],
    backbones: List[str],
) -> tuple:
    """
    Concatenate per-model probability outputs into a stacking feature matrix.

    Parameters
    ----------
    data      : dict backbone -> {"y_true", "y_pred", "proba"}
    backbones : ordered list of backbone names to include.

    Returns
    -------
    X : (N, C*M) feature matrix
    y : (N,) integer labels
    """
    X_parts = [data[b]["proba"] for b in backbones if b in data]
    if not X_parts:
        raise ValueError("No backbone data available for the given backbone list.")
    X = np.hstack(X_parts)
    y = data[backbones[0]]["y_true"]
    return X, y


class StackingClassifier:
    """
    Thin wrapper around sklearn meta-classifiers for use with the MHI pipeline.
    """

    def __init__(self, meta_type: str = "logreg", C: float = 1.0):
        """
        Parameters
        ----------
        meta_type : ``'logreg'`` (LogisticRegression) or ``'svm'`` (SVM-RBF).
        C         : regularisation inverse strength (sklearn convention).
        """
        self.meta_type = meta_type
        self.C = C
        self.scaler = StandardScaler()
        if meta_type == "logreg":
            self.meta = LogisticRegression(
                C=C, max_iter=2000, random_state=42, solver="lbfgs"
            )
        elif meta_type == "svm":
            self.meta = SVC(
                C=C, kernel="rbf", probability=True, random_state=42
            )
        else:
            raise ValueError(f"Unknown meta_type '{meta_type}'.")

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "StackingClassifier":
        X_val_scaled = self.scaler.fit_transform(X_val)
        self.meta.fit(X_val_scaled, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def run_stacking_evaluation(
    backbones: Optional[List[str]] = None,
    cfg=DEFAULT_CONFIG,
):
    """
    Full stacking evaluation pipeline:
    1. Load validation and test predictions.
    2. Fit meta-classifiers on validation features.
    3. Evaluate on test set.
    4. Save results.
    """
    os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)

    if backbones is None:
        backbones = ["mobilenet", "efficientnet", "densenet", "vgg16", "nasnet", "swin"]

    # Load predictions
    test_data = load_all_predictions(
        PREDICTIONS_DIR, cfg.class_names, backbones=backbones
    )
    val_data = load_all_predictions(
        VAL_PREDICTIONS_DIR, cfg.class_names, backbones=backbones
    )

    usable = [b for b in backbones if b in test_data and b in val_data]
    if len(usable) < 2:
        logger.error(
            "Need at least 2 backbones with both test and val predictions. "
            f"Usable: {usable}"
        )
        return

    logger.info(f"Running stacking evaluation with backbones: {usable}")

    X_val, y_val = build_stacking_features(val_data, usable)
    X_test, y_test = build_stacking_features(test_data, usable)

    results = []
    for meta_type in ["logreg", "svm"]:
        for C in [0.1, 1.0, 10.0]:
            clf = StackingClassifier(meta_type=meta_type, C=C)
            clf.fit(X_val, y_val)
            test_proba = clf.predict_proba(X_test)
            test_pred = clf.predict(X_test)

            m = compute_metrics(
                y_test, test_pred, cfg.class_names, y_pred_proba=test_proba
            )
            label = f"stack_{meta_type}_C{C}"
            logger.info(f"  {label}: {metrics_to_percent_string(m)}")
            results.append(
                {
                    "name": "+".join(usable),
                    "strategy": label,
                    "accuracy": m["accuracy"],
                    "f1_weighted": m["f1_weighted"],
                    "f1_macro": m["f1_macro"],
                    "roc_auc_macro": m.get("roc_auc_macro", float("nan")),
                    "cohen_kappa": m.get("cohen_kappa", float("nan")),
                }
            )

    df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    out_path = os.path.join(ENSEMBLE_RESULTS_DIR, "stacking_results.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Stacking results saved to {out_path}")
    return df


if __name__ == "__main__":
    run_stacking_evaluation()
