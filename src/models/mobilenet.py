"""
mobilenet.py
============
MobileNetV2 backbone (the modern, widely-available successor to the
MobileNetV1 used in the base paper; depthwise-separable-convolution family
either way), ImageNet-pretrained, adapted for single-MHI-image
classification.
"""

import tensorflow as tf
from tensorflow.keras import Model

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_mobilenet_model(cfg: Config = DEFAULT_CONFIG) -> Model:
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.MobileNetV2,
        preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input,
        model_name="MobileNetV2_MHI_Classifier",
        backbone_name="mobilenetv2_backbone",
    )
