"""
swin_transformer.py
===================
Swin Transformer V2 Tiny backbone hook.

This uses SwinTransformerV2Tiny_ns from keras-cv-attention-models.
The backbone is built as a feature extractor and the classification head
is added in this repository so stage-1 training still works when the
backbone is frozen.

Key changes vs. original
-------------------------
- Augmentation removed from model graph (now in tf.data pipeline).
- Classifier head upgraded to match _shared.py (two Dense layers, GELU,
  BatchNorm, two Dropout layers).
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.utils.config import Config, DEFAULT_CONFIG


class SwinTorchPreprocess(layers.Layer):
    """Normalise inputs with ImageNet-style mean/std (PyTorch convention).

    Replaces the former ``Lambda(_torch_preprocess)`` layer so that the
    layer can be serialised and deserialised by Keras 3.x without requiring
    ``safe_mode=False`` or pickling a Python lambda.

    Input is expected to be float32 in [0, 1].
    """

    def call(self, inputs):
        mean = tf.constant([0.485, 0.456, 0.406], dtype=inputs.dtype)
        std = tf.constant([0.229, 0.224, 0.225], dtype=inputs.dtype)
        return (inputs - mean) / std

    def get_config(self):
        return super().get_config()


def _build_swin_v2_backbone(input_shape):
    try:
        from keras_cv_attention_models import swin_transformer_v2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Swin Transformer support requires the optional dependency "
            "'keras-cv-attention-models'. "
            "Install it with `pip install keras-cv-attention-models` "
            "and then rerun training."
        ) from exc

    return swin_transformer_v2.SwinTransformerV2Tiny_ns(
        input_shape=input_shape,
        num_classes=0,
        pretrained="imagenet",
        classifier_activation=None,
    )


def build_swin_transformer_model(cfg: Config = DEFAULT_CONFIG):
    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    # Augmentation is now in the tf.data pipeline; do NOT apply it here.
    # Preprocess: [0, 1] → ImageNet mean/std normalisation
    x = SwinTorchPreprocess(name="swin_torch_preprocess")(inputs)

    backbone = _build_swin_v2_backbone(input_shape)
    backbone.trainable = not cfg.freeze_backbone
    backbone._name = "swin_transformer_v2_tiny_ns_backbone"

    x = backbone(x, training=False)
    if x.shape.rank == 4:
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
