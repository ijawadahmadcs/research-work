"""
ensemble_eval.py
================
Post-hoc ensemble evaluation on the held-out test set.

This script:
  1. Loads saved softmax probability outputs for every trained model from
     outputs/predictions/*.csv.
  2. Tests every pairwise and full-set combination of models.
  3. For each combination, tests:
       a. Uniform soft voting
       b. Validation-tuned weighted soft voting (weights optimised on the
          VALIDATION split, then applied to the test set)
       c. Hard majority voting
  4. Runs a stacking meta-classifier (LogisticRegression trained on
     validation probabilities, evaluated on test probabilities).
  5. Reports all metrics in a summary table and saves it to
     results/ensemble/ensemble_comparison.csv.

Important: ONLY validation-set data is used to tune ensemble weights and
fit the stacking meta-classifier.  The test set is touched exactly once
per combination, at the very end.

Usage
-----
    python -m src.evaluation.ensemble_eval
"""

import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.models.ensemble import (
    load_all_predictions,
    soft_vote,
    weighted_soft_vote,
    majority_vote,
)
from src.utils.config import DEFAULT_CONFIG, OUTPUTS_DIR, RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
ENSEMBLE_RESULTS_DIR = os.path.join(RESULTS_DIR, "ensemble")

# Models to include in the ensemble search
ALL_BACKBONES = ["mobilenet", "efficientnet", "densenet", "vgg16", "nasnet", "swin"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation predictions loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_val_predictions(
    class_names: List[str],
    backbones: List[str],
) -> Dict[str, Dict]:
    """
    Load validation-split predictions.  These CSVs are written by
    run_val_predictions() (called at the bottom of this script), which runs
    each saved model over the validation split and saves the results.
    """
    val_dir = os.path.join(OUTPUTS_DIR, "predictions", "val")
    return load_all_predictions(val_dir, class_names, backbones=backbones)


# ─────────────────────────────────────────────────────────────────────────────
# Weight optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _optimise_weights(
    val_probas: List[np.ndarray],
    val_y_true: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Find ensemble weights that maximise validation-set accuracy using Nelder-Mead.

    The weights are constrained to be non-negative (via softmax inside the
    objective) so the optimisation is unconstrained in the raw parameter space.

    Parameters
    ----------
    val_probas  : list of (N_val, C) probability arrays, one per model.
    val_y_true  : (N_val,) integer true labels for the validation split.
    n_classes   : number of classes.

    Returns
    -------
    Normalised weight vector of length len(val_probas).
    """
    n_models = len(val_probas)

    def _neg_accuracy(raw_w):
        # Softmax ensures weights are positive and sum to 1
        w = np.exp(raw_w - raw_w.max())
        w = w / w.sum()
        avg = weighted_soft_vote(val_probas, w)
        preds = np.argmax(avg, axis=1)
        return -accuracy_score(val_y_true, preds)

    x0 = np.ones(n_models) / n_models
    result = minimize(
        _neg_accuracy,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
    )
    raw_best = result.x
    w_best = np.exp(raw_best - raw_best.max())
    w_best = w_best / w_best.sum()
    return w_best


# ─────────────────────────────────────────────────────────────────────────────
# Stacking meta-classifier
# ─────────────────────────────────────────────────────────────────────────────

def _run_stacking(
    val_data: Dict[str, Dict],
    test_data: Dict[str, Dict],
    backbones: List[str],
    class_names: List[str],
) -> Dict:
    """
    Train a Logistic Regression meta-classifier on concatenated validation
    probabilities and evaluate it on concatenated test probabilities.

    Returns a metrics dict (same format as compute_metrics()).
    """
    # Build feature matrices: concatenate per-model probabilities row-wise
    val_X = np.hstack([val_data[b]["proba"] for b in backbones if b in val_data])
    val_y = val_data[backbones[0]]["y_true"]  # all models share the same y_true

    test_X = np.hstack([test_data[b]["proba"] for b in backbones if b in test_data])
    test_y = test_data[backbones[0]]["y_true"]

    scaler = StandardScaler()
    val_X = scaler.fit_transform(val_X)
    test_X = scaler.transform(test_X)

    meta = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    )
    meta.fit(val_X, val_y)

    test_proba = meta.predict_proba(test_X)
    test_pred = np.argmax(test_proba, axis=1)

    return compute_metrics(test_y, test_pred, class_names, y_pred_proba=test_proba)


# ─────────────────────────────────────────────────────────────────────────────
# Validation prediction writer
# ─────────────────────────────────────────────────────────────────────────────

def run_val_predictions(cfg=DEFAULT_CONFIG, backbones=None):
    """
    Run every saved final model over the validation split and save the
    probability outputs to outputs/predictions/val/<backbone>_val_predictions.csv.

    This is needed for weight optimisation and stacking meta-classifier training.
    """
    import tensorflow as tf
    from src.preprocessing.video_loader import build_dataset
    from src.utils.config import MODELS_DIR

    if backbones is None:
        backbones = ALL_BACKBONES

    val_dir = os.path.join(OUTPUTS_DIR, "predictions", "val")
    os.makedirs(val_dir, exist_ok=True)

    val_cfg = DEFAULT_CONFIG
    val_cfg.use_augmentation = False

    val_ds, n_val = build_dataset("validation", val_cfg, shuffle=False)
    logger.info(f"Validation split: {n_val} samples.")

    for backbone in backbones:
        model_path = os.path.join(MODELS_DIR, f"{backbone}_final_model.keras")
        if not os.path.exists(model_path):
            logger.warning(f"No saved model for {backbone}, skipping.")
            continue

        csv_out = os.path.join(val_dir, f"{backbone}_val_predictions.csv")
        if os.path.exists(csv_out):
            logger.info(f"Val predictions already exist for {backbone}, skipping.")
            continue

        logger.info(f"Running validation inference for {backbone} …")
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load {backbone}: {e}")
            continue

        y_true_list, y_proba_list = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_proba_list.append(preds)
            y_true_list.append(np.argmax(labels.numpy(), axis=1))

        y_true = np.concatenate(y_true_list)
        y_proba = np.concatenate(y_proba_list)
        y_pred = np.argmax(y_proba, axis=1)

        df = pd.DataFrame(
            {
                "true_label": [cfg.class_names[i] for i in y_true],
                "predicted_label": [cfg.class_names[i] for i in y_pred],
                **{
                    f"prob_{name}": y_proba[:, i]
                    for i, name in enumerate(cfg.class_names)
                },
            }
        )
        df.to_csv(csv_out, index=False)
        logger.info(f"  Saved to {csv_out}")

        del model
        tf.keras.backend.clear_session()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)
    cfg = DEFAULT_CONFIG

    # ── Step 1: ensure validation predictions are available ────────────────
    logger.info("=== Step 1: Collecting validation-split predictions ===")
    run_val_predictions(cfg, backbones=ALL_BACKBONES)

    # ── Step 2: load test and validation predictions ───────────────────────
    logger.info("=== Step 2: Loading test predictions ===")
    test_data = load_all_predictions(
        PREDICTIONS_DIR, cfg.class_names, backbones=ALL_BACKBONES
    )
    available_backbones = list(test_data.keys())
    logger.info(f"Available backbones: {available_backbones}")

    val_data = _load_val_predictions(cfg.class_names, available_backbones)
    available_val = list(val_data.keys())
    logger.info(f"Available val-prediction backbones: {available_val}")

    # Use only backbones that have BOTH test and val predictions
    usable = [b for b in available_backbones if b in available_val]
    logger.info(f"Usable backbones (test + val): {usable}")

    # ── Step 3: Individual model baselines ────────────────────────────────
    rows = []

    logger.info("=== Step 3: Individual model metrics ===")
    for backbone in available_backbones:
        d = test_data[backbone]
        m = compute_metrics(
            d["y_true"], d["y_pred"], cfg.class_names, y_pred_proba=d["proba"]
        )
        row = _metrics_to_row(m, backbone, "individual", None)
        rows.append(row)
        logger.info(f"  {backbone}: {metrics_to_percent_string(m)}")

    # ── Step 4: Ensemble combinations ────────────────────────────────────
    logger.info("=== Step 4: Ensemble combinations ===")

    for r in range(2, len(usable) + 1):
        for combo in itertools.combinations(usable, r):
            combo = list(combo)
            combo_name = "+".join(combo)
            test_probas = [test_data[b]["proba"] for b in combo]
            val_probas = [val_data[b]["proba"] for b in combo]
            y_true = test_data[combo[0]]["y_true"]
            val_y_true = val_data[combo[0]]["y_true"]

            # a) Uniform soft voting
            avg_proba = soft_vote(test_probas)
            avg_pred = np.argmax(avg_proba, axis=1)
            m_soft = compute_metrics(
                y_true, avg_pred, cfg.class_names, y_pred_proba=avg_proba
            )
            rows.append(
                _metrics_to_row(m_soft, combo_name, "soft_vote_uniform", None)
            )
            logger.info(f"  {combo_name} [uniform]: {metrics_to_percent_string(m_soft)}")

            # b) Validation-optimised weighted soft voting
            if len(val_probas[0]) >= 2:
                opt_weights = _optimise_weights(val_probas, val_y_true, cfg.num_classes)
                wavg_proba = weighted_soft_vote(test_probas, opt_weights)
                wavg_pred = np.argmax(wavg_proba, axis=1)
                m_wsoft = compute_metrics(
                    y_true, wavg_pred, cfg.class_names, y_pred_proba=wavg_proba
                )
                rows.append(
                    _metrics_to_row(
                        m_wsoft, combo_name, "soft_vote_weighted",
                        dict(zip(combo, opt_weights.tolist()))
                    )
                )
                logger.info(
                    f"  {combo_name} [weighted, w={np.round(opt_weights, 3)}]: "
                    f"{metrics_to_percent_string(m_wsoft)}"
                )

            # c) Hard majority voting
            preds_list = [test_data[b]["y_pred"] for b in combo]
            hard_pred = majority_vote(preds_list)
            m_hard = compute_metrics(y_true, hard_pred, cfg.class_names)
            rows.append(
                _metrics_to_row(m_hard, combo_name, "majority_vote", None)
            )

            # d) Stacking (only if all combo members have val predictions)
            if all(b in val_data for b in combo):
                m_stack = _run_stacking(
                    val_data, test_data, combo, cfg.class_names
                )
                rows.append(
                    _metrics_to_row(m_stack, combo_name, "stacking_logreg", None)
                )
                logger.info(
                    f"  {combo_name} [stacking]: {metrics_to_percent_string(m_stack)}"
                )

    # ── Step 5: Save & print summary ─────────────────────────────────────
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("f1_weighted", ascending=False)

    out_csv = os.path.join(ENSEMBLE_RESULTS_DIR, "ensemble_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"\nFull results saved to {out_csv}")

    # Print top-10 by weighted F1
    logger.info("\n=== TOP 10 CONFIGURATIONS BY WEIGHTED F1 ===")
    top10 = summary_df.head(10)
    for _, row in top10.iterrows():
        logger.info(
            f"  [{row['strategy']}] {row['name']:45s}  "
            f"Acc={row['accuracy']*100:.1f}%  "
            f"F1w={row['f1_weighted']*100:.1f}%  "
            f"F1m={row['f1_macro']*100:.1f}%  "
            f"AUC={row['roc_auc_macro']:.3f}"
        )

    # Recommendation
    best = summary_df.iloc[0]
    logger.info(
        f"\n[RECOMMENDATION] {best['name']} via {best['strategy']}\n"
        f"  Accuracy={best['accuracy']*100:.2f}%  "
        f"  F1-weighted={best['f1_weighted']*100:.2f}%  "
        f"  ROC-AUC={best['roc_auc_macro']:.3f}  "
        f"  Kappa={best['cohen_kappa']:.4f}"
    )


def _metrics_to_row(m: Dict, name: str, strategy: str, weights) -> Dict:
    return {
        "name": name,
        "strategy": strategy,
        "weights": str(weights) if weights else None,
        "accuracy": m["accuracy"],
        "f1_weighted": m["f1_weighted"],
        "f1_macro": m["f1_macro"],
        "precision_weighted": m["precision_weighted"],
        "recall_weighted": m["recall_weighted"],
        "roc_auc_macro": m.get("roc_auc_macro", float("nan")),
        "cohen_kappa": m.get("cohen_kappa", float("nan")),
    }


if __name__ == "__main__":
    main()
