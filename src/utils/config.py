import os
from dataclasses import dataclass, field
from typing import List, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
MHI_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "mhi_images")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLASS_NAMES: List[str] = ["NM", "KOA", "PD"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BACKBONE_CHOICES: List[str] = [
    "vgg16",
    "mobilenet",
    "densenet",
    "efficientnet",
    "nasnet",
    "googlenet",
    "inceptionv3",
    "swin",
    "ensemble",
]

RAW_CLASS_DIRS = {"NM": "NM", "KOA": "KOA", "PD": "PD"}


@dataclass
class Config:
    image_size: int = 224
    mhi_duration: int = 30            # tau: number of most-recent frames that stay "hot" in the MHI
    binary_threshold: Union[int, str] = "auto"  # "auto" = per-frame Otsu threshold (adapts to each
                                                 # video's own lighting/contrast); set an int to force
                                                 # a fixed threshold instead
    mhi_min_threshold: int = 12       # floor under an "auto" threshold, guards against noise-only
                                       # frame pairs producing "TV static" MHIs -- see mhi_generator.py
    mhi_morph_open_kernel: int = 3    # morphological-opening kernel size to erase isolated noise
                                       # pixels from the motion mask; 0 disables this cleanup step
    mhi_decay: float = 1.0            # decay step per frame (linear ramp), see mhi_generator.py

    test_size: float = 0.15
    val_size: float = 0.15            # taken out of the remaining train split
    random_seed: int = 42
    subject_wise_split: bool = True   # CRITICAL for avoiding data leakage (see preprocessing.py)

    backbone: str = "mobilenet"       # see BACKBONE_CHOICES
    ensemble_backbones: List[str] = field(default_factory=lambda: ["mobilenet", "vgg16", "densenet"])
    num_classes: int = 3
    dropout_rate: float = 0.4
    dense_units: int = 128
    freeze_backbone: bool = True      # freeze_backbone=True -> feature extraction; then fine_tune_at unfreezes some

    fine_tune: bool = True
    fine_tune_at_layer: int = -30     # unfreeze the last N layers of the backbone during stage 2

    batch_size: int = 16
    epochs_stage1: int = 30           # frozen-backbone warmup (train head only)
    epochs_stage2: int = 50           # fine-tuning stage
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5
    early_stopping_patience: int = 10
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5

    use_augmentation: bool = True

    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))


DEFAULT_CONFIG = Config()


def ensure_dirs(cfg: Config = DEFAULT_CONFIG) -> None:
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
