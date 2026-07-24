"""
Shared helpers for MHI backbone builders.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.preprocessing.augmentation import build_augmentation_pipeline
from src.utils.config import Config, DEFAULT_CONFIG


def build_application_classifier(
    cfg: Config,
    application_builder,
    preprocess_input,
    model_name: str,
    backbone_name: str,
) -> tuple[Model, tf.keras.Model]:
    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    x = inputs
    if cfg.use_augmentation:
        x = build_augmentation_pipeline()(x)

    x = layers.Rescaling(255.0)(x)
    x = preprocess_input(x)

    base_model = application_builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = not cfg.freeze_backbone
    base_model._name = backbone_name

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=model_name)
    model.backbone = base_model
    return model, base_model
