"""
evaluate.py
===========
Standalone evaluation script: loads a saved model, runs it on the
held-out test split, and writes a full suite of metrics, confusion
matrices, ROC curves, and a predictions CSV.

Usage
-----
    python -m src.evaluation.evaluate --backbone mobilenet
    python -m src.evaluation.evaluate --backbone efficientnet --model-path /path/to/custom.keras
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.evaluation.roc_curve import plot_multiclass_roc
from src.preprocessing.video_loader import build_dataset
from src.training.trainer import WarmupCosineDecaySchedule
from src.utils.config import BACKBONE_CHOICES, DEFAULT_CONFIG, MODELS_DIR, OUTPUTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collect_predictions(model, test_ds):
    """
    Run the model over the entire test dataset and collect true labels and
    predicted probabilities.

    Returns
    -------
    y_true       : 1-D int array of true class indices.
    y_pred_proba : (N, C) float array of softmax probabilities.
    """
    y_true_list, y_pred_proba_list = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba_list.append(preds)
        y_true_list.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_list)
    y_pred_proba = np.concatenate(y_pred_proba_list)
    return y_true, y_pred_proba


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MHI gait classifier on the test split."
    )
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default=DEFAULT_CONFIG.backbone)
    parser.add_argument(
        "--model-path", default=None,
        help="Override path to a .keras model file.",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.backbone = args.backbone
    # Disable augmentation at evaluation time (should already be False in cfg,
    # but explicitly set it here as a safeguard).
    cfg.use_augmentation = False

    model_path = args.model_path or os.path.join(
        MODELS_DIR, f"{cfg.backbone}_final_model.keras"
    )
    logger.info(f"Loading model from {model_path}")

    custom_objects = {
        "preprocess_input": lambda x: x,
        "WarmupCosineDecaySchedule": WarmupCosineDecaySchedule,
    }
    try:
        model = tf.keras.models.load_model(
            model_path, compile=False, safe_mode=False, custom_objects=custom_objects
        )
        logger.info("Model loaded via tf.keras.models.load_model().")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise e

    test_ds, n_test = build_dataset("test", cfg, shuffle=False)
    logger.info(f"Evaluating on {n_test} held-out test samples.")

    y_true, y_pred_proba = collect_predictions(model, test_ds)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ── Full metric suite ─────────────────────────────────────────────────
    metrics = compute_metrics(
        y_true, y_pred, cfg.class_names, y_pred_proba=y_pred_proba
    )
    logger.info(metrics_to_percent_string(metrics))
    logger.info("\n" + metrics["report_text"])

    # ── Confusion matrices ────────────────────────────────────────────────
    cm = compute_confusion_matrix(y_true, y_pred, cfg.class_names)
    plot_confusion_matrix(cm, cfg.class_names, cfg.backbone, normalize=False)
    plot_confusion_matrix(cm, cfg.class_names, cfg.backbone, normalize=True)

    # ── ROC curves ────────────────────────────────────────────────────────
    plot_multiclass_roc(y_true, y_pred_proba, cfg.class_names, cfg.backbone)

    # ── Predictions CSV ───────────────────────────────────────────────────
    predictions_dir = os.path.join(OUTPUTS_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "true_label": [cfg.class_names[i] for i in y_true],
            "predicted_label": [cfg.class_names[i] for i in y_pred],
            **{
                f"prob_{name}": y_pred_proba[:, i]
                for i, name in enumerate(cfg.class_names)
            },
        }
    )
    csv_path = os.path.join(
        predictions_dir, f"{cfg.backbone}_test_predictions.csv"
    )
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Predictions written to {csv_path}")


if __name__ == "__main__":
    main()
