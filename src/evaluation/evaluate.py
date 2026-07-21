import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.evaluation.roc_curve import plot_multiclass_roc
from src.preprocessing.video_loader import build_dataset
from src.utils.config import DEFAULT_CONFIG, MODELS_DIR, OUTPUTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MHI gait classifier on the test split.")
    parser.add_argument("--backbone", choices=["vgg16", "mobilenet", "densenet"], default=DEFAULT_CONFIG.backbone)
    parser.add_argument("--model-path", default=None, help="Override path to a .keras model file.")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.backbone = args.backbone

    model_path = args.model_path or os.path.join(MODELS_DIR, f"{cfg.backbone}_final_model.keras")
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

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
