"""
_shared.py
==========
Shared helpers for MHI backbone builders.

Key changes vs. original
-------------------------
- Augmentation is NO LONGER embedded in the model graph.  It runs in the
  tf.data pipeline (video_loader.py).  This ensures validation/test data
  are never augmented, which was a correctness bug in the original code.
- The Rescaling(255.0) → preprocess_input round-trip has been replaced by
  a direct preprocess_input call on [0, 1] inputs:
    * EfficientNet's preprocess_input maps [0,255] to [-1,1]; we multiply
      by 255 first so the math is correct.
    * All other Keras apps' preprocess_input functions also expect [0,255]
      range, so the same multiply-then-call pattern is applied uniformly.
- The classifier head is upgraded:
    GAP → Dense(256, GELU) → BatchNorm → Dropout(0.30)
        → Dense(128, GELU) → Dropout(0.20) → Softmax(num_classes)
  GELU is smoother than ReLU and typically superior for transfer-learning
  classification heads. Batch normalisation after the first dense layer
  stabilises training under AdamW's weight decay.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.utils.config import Config, DEFAULT_CONFIG


def build_application_classifier(
    cfg: Config,
    application_builder,
    preprocess_input,
    model_name: str,
    backbone_name: str,
) -> tuple[Model, tf.keras.Model]:
    """
    Build a [Input → preprocess_input → frozen backbone → improved head] model.

    Parameters
    ----------
    cfg                 : Config dataclass.
    application_builder : e.g. tf.keras.applications.MobileNetV2
    preprocess_input    : e.g. tf.keras.applications.mobilenet_v2.preprocess_input
    model_name          : string name for the full Keras model.
    backbone_name       : string name assigned to the backbone sub-model.

    Returns
    -------
    (full_model, backbone_submodel)
    """
    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    # ── Preprocessing: scale [0,1] → [0,255] → backbone's preprocess_input ──
    # All Keras pretrained applications expect [0, 255] float32 as input to
    # their preprocess_input function (which then applies model-specific
    # normalisation such as mean subtraction or [-1, 1] rescaling).
    x = layers.Rescaling(255.0, name="rescale_to_255")(inputs)
    x = layers.Lambda(preprocess_input, name="backbone_preprocess")(x)

    # ── Backbone ─────────────────────────────────────────────────────────────
    base_model = application_builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = not cfg.freeze_backbone
    base_model._name = backbone_name

    # training=False keeps BatchNorm layers in inference mode during stage 1
    # (backbone frozen).  This prevents the BN running statistics from being
    # corrupted by the small MHI mini-batches while the backbone weights are
    # still frozen.
    x = base_model(x, training=False)

    # ── Spatial pooling ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Improved classification head ──────────────────────────────────────────
    # Layer 1: Dense + GELU + BatchNorm + Dropout
    x = layers.Dense(cfg.dense_units, name="fc1")(x)
    x = layers.Activation("gelu", name="fc1_gelu")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout1")(x)

    # Layer 2: Dense + GELU + Dropout
    x = layers.Dense(cfg.dense_units2, name="fc2")(x)
    x = layers.Activation("gelu", name="fc2_gelu")(x)
    x = layers.Dropout(cfg.dropout_rate2, name="dropout2")(x)

    # Output
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=model_name)
    model.backbone = base_model
    return model, base_model
