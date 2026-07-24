"""
NASNetMobile backbone, ImageNet-pretrained, adapted for single-MHI-image
classification.
"""

import tensorflow as tf

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_nasnet_model(cfg: Config = DEFAULT_CONFIG):
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.NASNetMobile,
        preprocess_input=tf.keras.applications.nasnet.preprocess_input,
        model_name="NASNetMobile_MHI_Classifier",
        backbone_name="nasnetmobile_backbone",
    )
