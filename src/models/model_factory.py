"""
model_factory.py
================
Single entry point for building a model by backbone name, and a helper
for stage-2 fine-tuning (unfreezing the top N layers of the backbone).

Why there is no GRU here
------------------------
In the base paper, every video contributes a SEQUENCE of silhouette
frames. A CNN extracts a feature vector per frame, and a GRU then models
how those per-frame feature vectors evolve over time -- i.e. the GRU's
whole job is to learn the temporal dynamics *across frames of one video*.

Our pipeline collapses each video into a single MHI image before it ever
reaches the CNN. The temporal information the GRU used to learn
(recency, order, and direction of motion across frames) has already been
baked into the MHI itself by the recursive intensity-decay formula in
mhi_generator.py -- brightness IS the encoded "how recent was this
motion" signal. Mathematically: the GRU operated on a sequence
{f_1, f_2, ..., f_T} of per-frame CNN features and learned a summary
state h_T = GRU(f_1, ..., f_T). The MHI instead pre-computes a single
frame M = MHI(I_1, ..., I_T) via the fixed (non-learned) recursive rule
in mhi_generator.py, so the CNN now only ever sees ONE input, M, per
video: there is no sequence dimension left for a GRU to operate over.
Feeding a single time-step into a GRU is mathematically a no-op (a GRU
with T=1 degenerates to a single gated linear transform of one vector,
strictly less expressive than the FC + softmax head already used after
the CNN) -- so removing it is not just simpler, it avoids adding useless
parameters that could only overfit a small dataset.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model

from src.models.densenet import build_densenet_model
from src.models.mobilenet import build_mobilenet_model
from src.models.vgg16 import build_vgg16_model
from src.utils.config import Config, DEFAULT_CONFIG

_BUILDERS = {
    "vgg16": build_vgg16_model,
    "mobilenet": build_mobilenet_model,
    "densenet": build_densenet_model,
}


def build_model(cfg: Config = DEFAULT_CONFIG) -> Tuple[Model, tf.keras.Model]:
    """Returns (full_model, backbone_submodel)."""
    backbone_key = cfg.backbone.lower()
    if backbone_key not in _BUILDERS:
        raise ValueError(
            f"Unknown backbone '{cfg.backbone}'. Choose one of {list(_BUILDERS)}."
        )
    return _BUILDERS[backbone_key](cfg)


def unfreeze_for_fine_tuning(backbone: tf.keras.Model, fine_tune_at_layer: int = -30) -> None:
    """
    Unfreeze only the last `abs(fine_tune_at_layer)` layers of the backbone
    for stage-2 fine-tuning, keeping earlier (more generic, ImageNet-level)
    layers frozen. This is the standard "partial fine-tuning" recipe:
    early conv layers encode generic edge/texture detectors that transfer
    well regardless of domain, while later layers encode more
    task-specific features that benefit from adapting to MHI statistics
    (which look nothing like natural ImageNet photos).
    """
    backbone.trainable = True
    n_layers = len(backbone.layers)
    freeze_until = n_layers + fine_tune_at_layer if fine_tune_at_layer < 0 else fine_tune_at_layer
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until
        # Keep BatchNorm layers frozen even when "unfrozen" by index -- updating
        # BN running statistics on a small dataset destabilizes training badly.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
