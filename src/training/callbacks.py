"""
callbacks.py
============
Training callbacks for both stages of the MHI classifier.
"""

import os
from typing import List

import tensorflow as tf

from src.utils.config import Config, CHECKPOINTS_DIR, OUTPUTS_DIR


def build_callbacks(cfg: Config, stage_name: str) -> List[tf.keras.callbacks.Callback]:
    """
    Build the callback list for one training stage.

    Parameters
    ----------
    cfg        : Config dataclass.
    stage_name : ``'stage1'`` or ``'stage2'``; used to namespace checkpoint
                 and log filenames.

    Returns
    -------
    List of Keras Callback instances.
    """
    ckpt_dir = os.path.join(CHECKPOINTS_DIR)
    log_dir = os.path.join(OUTPUTS_DIR, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Best-accuracy checkpoint (primary — accuracy is the publication metric)
    ckpt_acc = os.path.join(
        ckpt_dir, f"{cfg.backbone}_{stage_name}_best_acc.keras"
    )
    # Best-loss checkpoint (secondary — sometimes more stable under label smoothing)
    ckpt_loss = os.path.join(
        ckpt_dir, f"{cfg.backbone}_{stage_name}_best_loss.keras"
    )
    csv_log = os.path.join(log_dir, f"{cfg.backbone}_{stage_name}_history.csv")

    callbacks = [
        # ── Primary checkpoint: best validation accuracy ──────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_acc,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # ── Secondary checkpoint: best validation loss ────────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_loss,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        # ── Early stopping (monitors val_loss; gives the cosine schedule   ──
        # ── time to complete before pulling the plug)                      ──
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # ── CSV history log ────────────────────────────────────────────────
        tf.keras.callbacks.CSVLogger(csv_log, append=False),
        # ── Guard against NaN losses crashing training silently ───────────
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks
