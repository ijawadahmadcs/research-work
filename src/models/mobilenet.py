"""
mobilenet.py
============
MobileNetV2 backbone (the modern, widely-available successor to the
MobileNetV1 used in the base paper; depthwise-separable-convolution family
either way), ImageNet-pretrained, adapted for single-MHI-image
classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.preprocessing.augmentation import build_augmentation_pipeline
from src.utils.config import Config, DEFAULT_CONFIG


def build_mobilenet_model(cfg: Config = DEFAULT_CONFIG) -> Model:
    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    x = inputs
    if cfg.use_augmentation:
        x = build_augmentation_pipeline()(x)

    x = layers.Rescaling(255.0)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape,
    )
    base_model.trainable = not cfg.freeze_backbone
    base_model._name = "mobilenetv2_backbone"

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="MobileNetV2_MHI_Classifier")
    return model, base_model
