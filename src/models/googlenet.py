"""
GoogLeNet-style backbone. TensorFlow Keras does not expose InceptionV1,
so this uses InceptionV3 as the closest built-in ImageNet-pretrained
approximation.
"""

import tensorflow as tf

from src.models._shared import build_application_classifier
from src.utils.config import Config, DEFAULT_CONFIG


def build_googlenet_model(cfg: Config = DEFAULT_CONFIG):
    return build_application_classifier(
        cfg=cfg,
        application_builder=tf.keras.applications.InceptionV3,
        preprocess_input=tf.keras.applications.inception_v3.preprocess_input,
        model_name="InceptionV3_MHI_Classifier",
        backbone_name="inceptionv3_backbone",
    )
