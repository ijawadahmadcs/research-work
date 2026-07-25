import argparse
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.evaluation.roc_curve import plot_multiclass_roc
from src.preprocessing.video_loader import build_dataset
from src.utils.config import BACKBONE_CHOICES, DEFAULT_CONFIG, MODELS_DIR, OUTPUTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_model_from_keras_archive(model_path: str, cfg):
    """Build the model architecture, then restore weights from the .keras zip archive.

    The saved .keras file serialised a Lambda layer referencing a Python
    function by name, which cannot be round-tripped by tf_keras's deserialiser.
    We therefore rebuild the identical architecture with model_factory and load
    only the weights from the ``model.weights.h5`` entry inside the archive.
    """
    from src.models.model_factory import build_model

    logger.info("Rebuilding model architecture for weight loading …")
    model, _ = build_model(cfg)

    with zipfile.ZipFile(model_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract("model.weights.h5", tmp)
            weights_path = os.path.join(tmp, "model.weights.h5")
            model.load_weights(weights_path)

    logger.info("Weights loaded successfully.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MHI gait classifier on the test split.")
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default=DEFAULT_CONFIG.backbone)
    parser.add_argument("--model-path", default=None, help="Override path to a .keras model file.")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.backbone = args.backbone

    model_path = args.model_path or os.path.join(MODELS_DIR, f"{cfg.backbone}_final_model.keras")
    logger.info(f"Loading model from {model_path}")
    model = _load_model_from_keras_archive(model_path, cfg)

    test_ds, n_test = build_dataset("test", cfg, shuffle=False)
    logger.info(f"Evaluating on {n_test} held-out test samples.")

    y_true, y_pred_proba = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba.append(preds)
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred_proba = np.concatenate(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)

    metrics = compute_metrics(y_true, y_pred, cfg.class_names)
    logger.info(metrics_to_percent_string(metrics))
    logger.info("\n" + metrics["report_text"])

    cm = compute_confusion_matrix(y_true, y_pred, cfg.class_names)
    plot_confusion_matrix(cm, cfg.class_names, cfg.backbone, normalize=False)
    plot_confusion_matrix(cm, cfg.class_names, cfg.backbone, normalize=True)
    plot_multiclass_roc(y_true, y_pred_proba, cfg.class_names, cfg.backbone)

    predictions_dir = os.path.join(OUTPUTS_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        "true_label": [cfg.class_names[i] for i in y_true],
        "predicted_label": [cfg.class_names[i] for i in y_pred],
        **{f"prob_{name}": y_pred_proba[:, i] for i, name in enumerate(cfg.class_names)},
    })
    pred_df.to_csv(os.path.join(predictions_dir, f"{cfg.backbone}_test_predictions.csv"), index=False)
    logger.info(f"Wrote predictions to {predictions_dir}")


if __name__ == "__main__":
    main()
