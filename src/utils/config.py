import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

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

# Per-model fine-tuning layer counts (how many layers from the end to unfreeze).
# Larger models need more layers unfrozen to adapt to MHI statistics.
BACKBONE_FINE_TUNE_LAYERS: Dict[str, int] = {
    "vgg16": 8,          # unfreeze block4_pool onwards (both conv blocks 4 & 5)
    "mobilenet": 30,     # unfreeze from block_14 depthwise blocks
    "densenet": 50,      # unfreeze last dense block + transition
    "efficientnet": 40,  # unfreeze last 3 MBConv blocks
    "nasnet": 40,        # unfreeze last 4 normal cells
    "googlenet": 30,     # inceptionv3: unfreeze last mixed layers
    "inceptionv3": 30,
    "swin": 40,          # unfreeze last 2 Swin stages
    "ensemble": 40,      # fallback — each member uses its own default
}


@dataclass
class Config:
    # ── MHI generation ──────────────────────────────────────────────────────
    image_size: int = 224
    mhi_duration: int = 30            # tau: frames that stay "hot"
    binary_threshold: Union[int, str] = "auto"  # "auto" = per-frame Otsu
    mhi_min_threshold: int = 12       # floor for auto-threshold
    mhi_morph_open_kernel: int = 3    # morphological-open kernel; 0 = disabled
    mhi_decay: float = 1.0            # decay step per frame
    mhi_colorize: bool = False        # if True, save 3-channel jet-colorized MHI

    # ── Dataset splitting ────────────────────────────────────────────────────
    test_size: float = 0.15
    val_size: float = 0.15            # fraction of (train+val) that becomes val
    random_seed: int = 42
    subject_wise_split: bool = True   # CRITICAL: prevents within-subject leakage

    # ── Model ────────────────────────────────────────────────────────────────
    backbone: str = "mobilenet"
    ensemble_backbones: List[str] = field(
        default_factory=lambda: ["mobilenet", "efficientnet", "densenet"]
    )
    num_classes: int = 3
    dropout_rate: float = 0.30        # was 0.40; reduced to lower regularisation
    dropout_rate2: float = 0.20       # second dropout in the two-layer head
    dense_units: int = 256            # was 128; wider head for richer features
    dense_units2: int = 128           # second FC layer
    freeze_backbone: bool = True

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    fine_tune: bool = True
    fine_tune_at_layer: int = -40     # default; overridden per-backbone in model_factory
    use_per_backbone_fine_tune_layers: bool = True  # use BACKBONE_FINE_TUNE_LAYERS dict

    # ── Training schedule ────────────────────────────────────────────────────
    batch_size: int = 16
    epochs_stage1: int = 40           # head-only warmup (more epochs with cosine decay)
    epochs_stage2: int = 60           # fine-tuning stage
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 5e-5  # was 1e-5; slightly higher for more fine-tune progress

    # AdamW
    weight_decay: float = 1e-4        # L2 weight decay (AdamW)

    # Cosine decay + warmup (stage 1)
    warmup_epochs: int = 5            # linear warmup before cosine decay

    # ReduceLROnPlateau (used in stage 2 as a safety net)
    early_stopping_patience: int = 15  # was 10; give more room with cosine schedule
    lr_reduce_patience: int = 7
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-8

    # ── Loss ─────────────────────────────────────────────────────────────────
    label_smoothing: float = 0.08     # smoothed cross-entropy for calibration
    use_class_weights: bool = True    # inverse-frequency weighting to help PD

    # ── Augmentation ─────────────────────────────────────────────────────────
    use_augmentation: bool = True
    augmentation_in_model: bool = False  # MUST be False: augmentation runs in tf.data

    # ── Precision ────────────────────────────────────────────────────────────
    use_mixed_precision: bool = False  # set True if GPU supports float16

    # ── Class names ──────────────────────────────────────────────────────────
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
        os.path.join(OUTPUTS_DIR, "tensorboard"),
        os.path.join(RESULTS_DIR, "ensemble"),
    ]:
        os.makedirs(d, exist_ok=True)
