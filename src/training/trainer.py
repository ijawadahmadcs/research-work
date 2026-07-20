"""
trainer.py
==========
Two-stage transfer-learning training loop:

  Stage 1 (feature extraction / warmup): backbone frozen, only the new
  FC head trains, with a relatively large learning rate. This lets the
  randomly-initialized head settle into a reasonable region without
  corrupting the pretrained ImageNet weights via large, noisy gradients.

  Stage 2 (fine-tuning): the top `fine_tune_at_layer` backbone layers are
  unfrozen and the WHOLE thing trains together with a much smaller
  learning rate, letting the backbone's high-level features adapt
  specifically to MHI image statistics (which look nothing like natural
  photos: mostly black background, a single bright blob-and-trail shape).

This staged approach is standard practice for transfer learning on small
datasets (see config.py comments) and avoids the common mistake of
fine-tuning the whole network from epoch 1 with a large learning rate,
which tends to destroy the pretrained weights before the new head has
learned anything useful to backpropagate.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from src.models.model_factory import build_model, unfreeze_for_fine_tuning
from src.training.callbacks import build_callbacks
from src.utils.config import Config, DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model: tf.keras.Model
    history_stage1: tf.keras.callbacks.History
    history_stage2: Optional[tf.keras.callbacks.History]


class Trainer:
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.model, self.backbone = build_model(cfg)

    def compile_stage1(self) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate_stage1),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def compile_stage2(self) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate_stage2),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def fit(self, train_ds, val_ds) -> TrainingResult:
        cfg = self.cfg

        logger.info("=== Stage 1: training head only (backbone frozen) ===")
        self.compile_stage1()
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs_stage1,
            callbacks=build_callbacks(cfg, "stage1"),
        )

        history2 = None
        if cfg.fine_tune:
            logger.info("=== Stage 2: fine-tuning top layers of backbone ===")
            unfreeze_for_fine_tuning(self.backbone, cfg.fine_tune_at_layer)
            self.compile_stage2()
            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.epochs_stage2,
                callbacks=build_callbacks(cfg, "stage2"),
            )

        return TrainingResult(model=self.model, history_stage1=history1, history_stage2=history2)
