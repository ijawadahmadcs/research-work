"""
main.py
=======
Single entry point running the full pipeline end-to-end:

    python -m src.main --stage all --backbone mobilenet
    python -m src.main --stage preprocess
    python -m src.main --stage train --backbone vgg16
    python -m src.main --stage evaluate --backbone vgg16
"""

import argparse

from src.preprocessing.preprocessing import run_full_preprocessing
from src.preprocessing.video_loader import build_dataset
from src.evaluation.plots import plot_training_curves
from src.evaluation.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import compute_metrics, metrics_to_percent_string
from src.evaluation.roc_curve import plot_multiclass_roc
from src.training.trainer import Trainer
from src.utils.config import BACKBONE_CHOICES, DEFAULT_CONFIG, MODELS_DIR, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

import os
import numpy as np

logger = get_logger("main")


def run_preprocess(cfg):
    logger.info("### STAGE: preprocessing (videos -> MHI -> subject-wise split) ###")
    run_full_preprocessing(cfg)


def run_train(cfg):
    logger.info(f"### STAGE: training ({cfg.backbone}) ###")
    train_ds, n_train = build_dataset("train", cfg)
    val_ds, n_val = build_dataset("validation", cfg)
    logger.info(f"{n_train} train samples, {n_val} validation samples.")

    trainer = Trainer(cfg)
    result = trainer.fit(train_ds, val_ds)

    final_model_path = os.path.join(MODELS_DIR, f"{cfg.backbone}_final_model.keras")
    result.model.save(final_model_path)
    logger.info(f"Saved final model -> {final_model_path}")
    plot_training_curves(result.history_stage1, result.history_stage2, cfg.backbone)
    return result.model


def run_evaluate(cfg, model=None):
    import tensorflow as tf
    logger.info(f"### STAGE: evaluation ({cfg.backbone}) ###")
    if model is None:
        model_path = os.path.join(MODELS_DIR, f"{cfg.backbone}_final_model.keras")
        model = tf.keras.models.load_model(model_path)

    test_ds, n_test = build_dataset("test", cfg, shuffle=False)
    logger.info(f"Evaluating on {n_test} test samples.")

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

    cm = compute_confusion_matrix(y_true, y_pred, cfg.class_names)
    plot_confusion_matrix(cm, cfg.class_names, cfg.backbone, normalize=True)
    plot_multiclass_roc(y_true, y_pred_proba, cfg.class_names, cfg.backbone)


def main():
    parser = argparse.ArgumentParser(description="MHI-based gait disease classification pipeline.")
    parser.add_argument("--stage", choices=["preprocess", "train", "evaluate", "all"], default="all")
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default=DEFAULT_CONFIG.backbone)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.backbone = args.backbone
    set_global_seed(cfg.random_seed)
    ensure_dirs(cfg)

    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg)
    if args.stage in ("train", "all"):
        model = run_train(cfg)
    else:
        model = None
    if args.stage in ("evaluate", "all"):
        run_evaluate(cfg, model=model)


if __name__ == "__main__":
    main()
