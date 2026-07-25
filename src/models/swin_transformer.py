"""
Swin Transformer backbone hook.

This uses Keras SwinTransformerV2 from keras-cv-attention-models.
The backbone is built as a feature extractor and the classification head
is added in this repository so stage-1 training still works when the
backbone is frozen.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.utils.config import Config, DEFAULT_CONFIG


class SwinTorchPreprocess(layers.Layer):
    """Normalise inputs with ImageNet-style mean/std (PyTorch convention).

    Replaces the former ``Lambda(_torch_preprocess)`` layer so that the
    layer can be serialised and deserialised by Keras 3.x without requiring
    ``safe_mode=False`` or pickling a Python lambda.
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
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Swin Transformer support requires the optional dependency 'keras-cv-attention-models'. "
            "Install it with `pip install keras-cv-attention-models` and then rerun training."
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

    x = inputs
    if cfg.use_augmentation:
        from src.preprocessing.augmentation import build_augmentation_pipeline

        x = build_augmentation_pipeline()(x)

    x = SwinTorchPreprocess(name="swin_torch_preprocess")(x)

    backbone = _build_swin_v2_backbone(input_shape)
    backbone.trainable = not cfg.freeze_backbone
    backbone._name = "swin_transformer_v2_tiny_ns_backbone"

    x = backbone(x)
    if x.shape.rank == 4:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="SwinTransformerV2Tiny_ns_MHI_Classifier")
    model.backbone = backbone
    return model, backbone

