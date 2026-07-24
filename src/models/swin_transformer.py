"""
Swin Transformer backbone hook.

This repository currently has no Swin-capable dependency installed, so
the builder raises a clear error telling the user what to install.
"""

from src.utils.config import Config, DEFAULT_CONFIG


def build_swin_transformer_model(cfg: Config = DEFAULT_CONFIG):
    raise ImportError(
        "Swin Transformer support is not available in the current environment. "
        "Install a Swin-capable library such as keras-cv-attention-models or timm/transformers, "
        "then wire the builder in src/models/swin_transformer.py."
    )
