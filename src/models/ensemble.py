"""
Model-fusion helpers.
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.models.densenet import build_densenet_model
from src.models.efficientnet import build_efficientnet_model
from src.models.googlenet import build_googlenet_model
from src.models.mobilenet import build_mobilenet_model
from src.models.nasnet import build_nasnet_model
from src.models.swin_transformer import build_swin_transformer_model
from src.models.vgg16 import build_vgg16_model
from src.utils.config import Config, DEFAULT_CONFIG


_MEMBER_BUILDERS = {
    "vgg16": build_vgg16_model,
    "mobilenet": build_mobilenet_model,
    "densenet": build_densenet_model,
    "efficientnet": build_efficientnet_model,
    "nasnet": build_nasnet_model,
    "googlenet": build_googlenet_model,
    "inceptionv3": build_googlenet_model,
    "swin": build_swin_transformer_model,
}


def build_ensemble_model(cfg: Config = DEFAULT_CONFIG) -> Tuple[Model, List[tf.keras.Model]]:
    member_names = cfg.ensemble_backbones or ["mobilenet", "vgg16", "densenet"]
    if len(member_names) < 2:
        raise ValueError("Ensemble mode requires at least two member backbones.")

    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    member_models = []
    member_outputs = []
    for member_name in member_names:
        key = member_name.lower()
        if key not in _MEMBER_BUILDERS:
            raise ValueError(
                f"Unknown ensemble member '{member_name}'. Choose from {sorted(_MEMBER_BUILDERS)}."
            )
        member_model, _ = _MEMBER_BUILDERS[key](cfg)
        member_models.append(member_model)
        member_outputs.append(member_model(inputs))

    outputs = layers.Average(name="ensemble_average")(member_outputs)
    model = Model(inputs, outputs, name="Ensemble_MHI_Classifier")
    model.member_backbones = [getattr(member, "backbone", member) for member in member_models]
    model.member_models = member_models
    return model, model.member_backbones
