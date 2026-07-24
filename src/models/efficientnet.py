"""
EfficientNetB0 backbone, ImageNet-pretrained, adapted for single-MHI-image
classification.
"""

import tensorflow as tf

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_efficientnet_model(cfg: Config = DEFAULT_CONFIG):
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.EfficientNetB0,
        preprocess_input=tf.keras.applications.efficientnet.preprocess_input,
        model_name="EfficientNetB0_MHI_Classifier",
        backbone_name="efficientnetb0_backbone",
    )
