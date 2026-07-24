"""
vgg16.py
========
VGG16 backbone, ImageNet-pretrained, adapted for single-MHI-image
classification (no GRU / temporal stage -- see model_factory.py docstring
for why that stage is removed).
"""

import tensorflow as tf
from tensorflow.keras import Model

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_vgg16_model(cfg: Config = DEFAULT_CONFIG) -> Model:
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.VGG16,
        preprocess_input=tf.keras.applications.vgg16.preprocess_input,
        model_name="VGG16_MHI_Classifier",
        backbone_name="vgg16_backbone",
    )
