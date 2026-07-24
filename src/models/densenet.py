"""
densenet.py
===========
DenseNet121 backbone, ImageNet-pretrained, adapted for single-MHI-image
classification.
"""

import tensorflow as tf

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_densenet_model(cfg: Config = DEFAULT_CONFIG) -> Model:
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.DenseNet121,
        preprocess_input=tf.keras.applications.densenet.preprocess_input,
        model_name="DenseNet121_MHI_Classifier",
        backbone_name="densenet121_backbone",
    )
