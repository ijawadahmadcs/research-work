"""
train.py
========
CLI entry point for training one backbone end-to-end:
    python -m src.training.train --backbone mobilenet

Loads the already-preprocessed train/validation MHI datasets, runs the
two-stage Trainer, saves the final model, and plots training curves.
"""

import argparse
import os

from src.evaluation.plots import plot_training_curves
from src.preprocessing.video_loader import build_dataset
from src.training.trainer import Trainer
from src.utils.config import DEFAULT_CONFIG, MODELS_DIR, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train an MHI-based gait classifier.")
    parser.add_argument("--backbone", choices=["vgg16", "mobilenet", "densenet"], default=DEFAULT_CONFIG.backbone)
    parser.add_argument("--epochs-stage1", type=int, default=DEFAULT_CONFIG.epochs_stage1)
    parser.add_argument("--epochs-stage2", type=int, default=DEFAULT_CONFIG.epochs_stage2)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--no-fine-tune", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.backbone = args.backbone
    cfg.epochs_stage1 = args.epochs_stage1
    cfg.epochs_stage2 = args.epochs_stage2
    cfg.batch_size = args.batch_size
    cfg.fine_tune = not args.no_fine_tune

    set_global_seed(cfg.random_seed)
    ensure_dirs(cfg)

    logger.info(f"Config: backbone={cfg.backbone}, batch_size={cfg.batch_size}, "
                f"epochs=({cfg.epochs_stage1}+{cfg.epochs_stage2}), fine_tune={cfg.fine_tune}")

    train_ds, n_train = build_dataset("train", cfg)
    val_ds, n_val = build_dataset("validation", cfg)
    logger.info(f"Loaded {n_train} training samples and {n_val} validation samples.")

    trainer = Trainer(cfg)
    result = trainer.fit(train_ds, val_ds)

    final_model_path = os.path.join(MODELS_DIR, f"{cfg.backbone}_final_model.keras")
    result.model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    plot_training_curves(result.history_stage1, result.history_stage2, cfg.backbone)


if __name__ == "__main__":
    main()
