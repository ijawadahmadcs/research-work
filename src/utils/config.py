"""
config.py
=========
Single source of truth for paths, class labels, and hyperparameters.
Every other module imports from here instead of hard-coding values,
so the whole pipeline stays reproducible and easy to re-run with a
different backbone / dataset location.
"""

import os
from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
MHI_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "mhi_images")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# Multi-class (3-way) setup — matches the base paper's first experiment.
CLASS_NAMES: List[str] = ["Healthy", "KOA", "PD"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Folder names as they exist under data/raw_videos and data/mhi_images.
# NM = "normal" / healthy in the original Kour et al. dataset naming.
RAW_CLASS_DIRS = {"Healthy": "NM", "KOA": "KOA", "PD": "PD"}


@dataclass
class Config:
    # ---- Image / MHI parameters --------------------------------------- #
    image_size: int = 224
    mhi_duration: int = 30            # tau: number of most-recent frames that stay "hot" in the MHI
    binary_threshold: int = 30        # threshold on frame differencing before updating the MHI
    mhi_decay: float = 1.0            # decay step per frame (linear ramp), see mhi_generator.py

    # ---- Splitting ------------------------------------------------------ #
    test_size: float = 0.15
    val_size: float = 0.15            # taken out of the remaining train split
    random_seed: int = 42
    subject_wise_split: bool = True   # CRITICAL for avoiding data leakage (see preprocessing.py)

    # ---- Model ----------------------------------------------------------- #
    backbone: str = "mobilenet"       # one of: "vgg16", "mobilenet", "densenet"
    num_classes: int = 3
    dropout_rate: float = 0.4
    dense_units: int = 128
    freeze_backbone: bool = True      # freeze_backbone=True -> feature extraction; then fine_tune_at unfreezes some

    # ---- Fine-tuning ------------------------------------------------------ #
    fine_tune: bool = True
    fine_tune_at_layer: int = -30     # unfreeze the last N layers of the backbone during stage 2

    # ---- Training ---------------------------------------------------------- #
    batch_size: int = 16
    epochs_stage1: int = 30           # frozen-backbone warmup (train head only)
    epochs_stage2: int = 50           # fine-tuning stage
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5
    early_stopping_patience: int = 10
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5

    # ---- Augmentation ------------------------------------------------------- #
    use_augmentation: bool = True

    # ---- Misc ----------------------------------------------------------------- #
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))


DEFAULT_CONFIG = Config()


def ensure_dirs(cfg: Config = DEFAULT_CONFIG) -> None:
    """Create every directory the pipeline writes to, if missing."""
    for d in [
        RAW_VIDEOS_DIR, MHI_IMAGES_DIR, PROCESSED_DIR, METADATA_DIR,
        MODELS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, RESULTS_DIR,
        os.path.join(OUTPUTS_DIR, "training_curves"),
        os.path.join(OUTPUTS_DIR, "confusion_matrices"),
        os.path.join(OUTPUTS_DIR, "roc_curves"),
        os.path.join(OUTPUTS_DIR, "predictions"),
        os.path.join(OUTPUTS_DIR, "logs"),
    ]:
        os.makedirs(d, exist_ok=True)
