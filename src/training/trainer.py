"""
trainer.py
==========
Trains a two-stage (head-only → partial fine-tune) MHI gait classifier.

Stage 1 — Head-only warmup (backbone frozen):
  - Optimizer: AdamW with weight decay and gradient clipping.
  - Schedule: linear warmup for `warmup_epochs` epochs, then cosine decay
    to `learning_rate_stage1 * 0.01` by the end of stage 1.
  - Loss: categorical cross-entropy with label smoothing (0.08).
  - Class weights applied to the loss to handle KOA/PD imbalance.

Stage 2 — Partial fine-tuning (top N backbone layers unfrozen):
  - Lower learning rate, shorter cosine decay.
  - ReduceLROnPlateau as a safety net in addition to cosine decay.
  - BatchNorm layers remain frozen throughout to preserve ImageNet
    running statistics.

Why AdamW over Adam
-------------------
Adam accumulates a per-parameter adaptive scale that interacts with L2
weight decay: the effective regularisation depends on the gradient
magnitude, meaning parameters with large gradients receive almost no L2
penalty. AdamW decouples weight decay from the adaptive term so every
parameter is penalised by the same fraction each step, giving proper
and predictable regularisation — especially important for a small dataset
where overfitting is the dominant failure mode.

Why label smoothing
-------------------
Hard one-hot targets force the softmax output to 1.0, driving the logits
for the correct class to +∞ and producing overconfident models that
generalise poorly. Label smoothing (ε = 0.08) replaces p(correct) = 1
with p(correct) = 1 - ε and p(other) = ε/(K-1), capping over-confidence
and improving calibration without changing which class has the highest
probability.

Why class weights
-----------------
Training distribution: KOA ≈ 57%, NM ≈ 26%, PD ≈ 17%.  PD is the
hardest to classify and the rarest class.  Inverse-frequency class
weights scale the loss so that a PD mistake costs ~3× more than a KOA
mistake, steering the optimiser to prioritise the minority classes.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup from `warmup_lr` to `peak_lr` over `warmup_steps` steps,
    then cosine decay from `peak_lr` to `min_lr` over the remaining steps.

    This is the standard recipe used by ViT / EfficientNet fine-tuning:
    the warmup avoids large early gradient updates (which can damage the
    pretrained weights) and the cosine tail gives a smooth annealing.
    """

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-8,
        name: str = "warmup_cosine",
    ):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr = min_lr
        self._name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        peak_lr = tf.cast(self.peak_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)

        # Warmup phase
        warmup_lr = peak_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        cosine_lr = min_lr + 0.5 * (peak_lr - min_lr) * (
            1.0 + tf.cos(tf.constant(math.pi) * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "name": self._name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_ds: tf.data.Dataset, num_classes: int
) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights from the training dataset.

    Weights are normalised so that the mean weight across all classes is 1.0,
    preventing the learning rate from being implicitly scaled by the weighting.

    Returns
    -------
    dict mapping class index (int) → float weight.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, labels in train_ds:
        class_indices = np.argmax(labels.numpy(), axis=1)
        for idx in class_indices:
            counts[idx] += 1

    # Guard against zero counts (should not happen but prevents /0)
    counts = np.maximum(counts, 1)
    total = counts.sum()
    # Inverse-frequency, normalised so mean weight = 1.0
    weights = (total / (num_classes * counts))
    return {i: float(w) for i, w in enumerate(weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg

        # Enable mixed precision if requested (requires GPU with Tensor Cores)
        if cfg.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (float16 compute, float32 weights).")

        self.model, self.backbone = build_model(cfg)

    def _make_optimizer(self, lr_schedule, clip_norm: float = 1.0):
        """Return an AdamW optimizer with gradient clipping."""
        return tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.cfg.weight_decay,
            clipnorm=clip_norm,
        )

    def _make_loss(self):
        """Categorical cross-entropy with label smoothing."""
        return tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.cfg.label_smoothing,
        )

    def _make_metrics(self):
        return [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ]

    def compile_stage1(self, steps_per_epoch: int) -> None:
        """Compile for head-only training with cosine LR + warmup."""
        total_steps = self.cfg.epochs_stage1 * steps_per_epoch
        warmup_steps = self.cfg.warmup_epochs * steps_per_epoch

        schedule = WarmupCosineDecaySchedule(
            peak_lr=self.cfg.learning_rate_stage1,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=self.cfg.min_lr,
            name="warmup_cosine_s1",
        )

        self.model.compile(
            optimizer=self._make_optimizer(schedule),
            loss=self._make_loss(),
            metrics=self._make_metrics(),
        )
        logger.info(
            "Stage 1 compiled: AdamW + cosine-warmup (peak_lr=%.4e, "
            "warmup=%d epochs, total=%d epochs)",
            self.cfg.learning_rate_stage1,
            self.cfg.warmup_epochs,
            self.cfg.epochs_stage1,
        )

    def compile_stage2(self, steps_per_epoch: int) -> None:
        """Compile for fine-tuning with a lower cosine LR schedule."""
        total_steps = self.cfg.epochs_stage2 * steps_per_epoch

        schedule = WarmupCosineDecaySchedule(
            peak_lr=self.cfg.learning_rate_stage2,
            total_steps=total_steps,
            warmup_steps=min(2 * steps_per_epoch, total_steps // 10),
            min_lr=self.cfg.min_lr,
            name="warmup_cosine_s2",
        )

        self.model.compile(
            optimizer=self._make_optimizer(schedule),
            loss=self._make_loss(),
            metrics=self._make_metrics(),
        )
        logger.info(
            "Stage 2 compiled: AdamW + cosine (peak_lr=%.4e, total=%d epochs)",
            self.cfg.learning_rate_stage2,
            self.cfg.epochs_stage2,
        )

    def fit(
        self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset
    ) -> TrainingResult:
        cfg = self.cfg
        n_train_batches = sum(1 for _ in train_ds)  # count steps per epoch

        # ── Class weights ──────────────────────────────────────────────────
        class_weight = None
        if cfg.use_class_weights:
            class_weight = compute_class_weights(train_ds, cfg.num_classes)
            logger.info("Class weights: %s", class_weight)

        # ── Stage 1: head-only warmup ──────────────────────────────────────
        logger.info("=== Stage 1: training head only (backbone frozen) ===")
        self.compile_stage1(steps_per_epoch=n_train_batches)

        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs_stage1,
            callbacks=build_callbacks(cfg, "stage1"),
            class_weight=class_weight,
        )

        # ── Stage 2: partial fine-tuning ──────────────────────────────────
        history2 = None
        if cfg.fine_tune:
            logger.info("=== Stage 2: fine-tuning top layers of backbone ===")
            unfreeze_for_fine_tuning(
                self.backbone,
                fine_tune_at_layer=cfg.fine_tune_at_layer,
                backbone_key=cfg.backbone,
                cfg=cfg,
            )
            self.compile_stage2(steps_per_epoch=n_train_batches)

            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.epochs_stage2,
                callbacks=build_callbacks(cfg, "stage2"),
                class_weight=class_weight,
            )

        return TrainingResult(
            model=self.model,
            history_stage1=history1,
            history_stage2=history2,
        )
