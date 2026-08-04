"""
swin_transformer.py
===================
Swin Transformer V2 Tiny backbone hook.

Uses SwinTransformerV2Tiny_ns from keras-cv-attention-models.
Note: If keras-cv-attention-models is incompatible with Keras 3.x symbolic tensors
in the current environment, this module gracefully catches the error and reports it.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.utils.config import Config, DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_swin_v2_backbone(input_shape):
    try:
        import os
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        from keras_cv_attention_models import swin_transformer_v2
        return swin_transformer_v2.SwinTransformerV2Tiny_ns(
            input_shape=input_shape,
            num_classes=0,
            pretrained="imagenet",
            classifier_activation=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Swin Transformer backbone failed to initialize under Keras 3 ({exc}). "
            "Skipping Swin backbone."
        ) from exc


def build_swin_transformer_model(cfg: Config = DEFAULT_CONFIG):
    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    # Preprocess: [0, 1] → PyTorch ImageNet mean/std normalisation via Keras Normalization layer
    mean = [0.485, 0.456, 0.406]
    variance = [0.229**2, 0.224**2, 0.225**2]
    norm_layer = layers.Normalization(mean=mean, variance=variance, axis=-1, name="swin_torch_preprocess")
    x = norm_layer(inputs)

    backbone = _build_swin_v2_backbone(input_shape)
    backbone.trainable = not cfg.freeze_backbone
    backbone._name = "swin_transformer_v2_tiny_ns_backbone"

    x = backbone(x, training=False)
    if hasattr(x, "shape") and len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Improved classification head (matches _shared.py) ────────────────────
    x = layers.Dense(cfg.dense_units, name="fc1")(x)
    x = layers.Activation("gelu", name="fc1_gelu")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout1")(x)

    x = layers.Dense(cfg.dense_units2, name="fc2")(x)
    x = layers.Activation("gelu", name="fc2_gelu")(x)
    x = layers.Dropout(cfg.dropout_rate2, name="dropout2")(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="SwinTransformerV2Tiny_ns_MHI_Classifier")
    model.backbone = backbone
    return model, backbone
