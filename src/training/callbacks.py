"""
callbacks.py
============
Standard training callbacks, justified individually:

- ModelCheckpoint: always keep the weights from the epoch with the best
  validation loss, not just whatever the last epoch happened to be --
  final-epoch weights are frequently worse than an earlier checkpoint
  once the model starts overfitting.
- EarlyStopping: stop wasting compute (and risking overfitting) once
  validation loss stops improving for `patience` epochs; restores the
  best weights automatically.
- ReduceLROnPlateau: when validation loss plateaus, shrink the learning
  rate rather than stopping outright -- often lets the optimizer settle
  into a sharper minimum before early stopping finally kicks in.
- CSVLogger: durable per-epoch record of every metric, so training
  curves (accuracy/loss plots) can be reproduced later without re-running
  training.
"""

import os
from typing import List

import tensorflow as tf

from src.utils.config import Config, CHECKPOINTS_DIR, OUTPUTS_DIR


def build_callbacks(cfg: Config, stage_name: str) -> List[tf.keras.callbacks.Callback]:
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{cfg.backbone}_{stage_name}_best.keras")
    log_path = os.path.join(OUTPUTS_DIR, "logs", f"{cfg.backbone}_{stage_name}_history.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=False),
    ]
