"""
video_loader.py
================
Two responsibilities:
  1. Low-level video frame iteration (used by mhi_generator / notebooks for
     visual inspection of a video's frames).
  2. Building tf.data.Dataset objects from the already-generated
     processed/{train,validation,test}/<class>/ MHI image folders, for the
     training stage.

Key design changes vs. the original
-------------------------------------
- **Augmentation is applied in the tf.data pipeline** (training split only),
  NOT inside the Keras model graph.  This is the correct approach because
  Keras layer `training` flags inside a Sequential embedded in the model
  graph do not always toggle correctly during `.fit()` validation steps,
  causing augmentation to corrupt validation metrics.
- **No Rescaling(255.0) round-trip.**  Images are loaded as float32 in
  [0, 1].  Each backbone's `preprocess_input` expects a specific input
  range; we feed the [0, 1] image directly to the backbone wrapper which
  applies the correct `preprocess_input` there.  (Note: `_shared.py` also
  had a `Rescaling(255.0)` → `preprocess_input` pattern; that has been
  removed and `preprocess_input` now receives [0, 1] inputs directly —
  which matches what ImageNet-pretrained Keras applications expect when
  called with their documented `preprocess_input` functions.)
"""

import os
from typing import Iterator, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.augmentation import build_augmentation_pipeline, apply_augmentation
from src.utils.config import Config, DEFAULT_CONFIG, PROCESSED_DIR, CLASS_NAMES


def iter_video_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file, one at a time (memory-light)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _preprocess_image(path: str, label: int, image_size: int, num_classes: int):
    """
    Load a PNG MHI, resize, convert to 3-channel float32 in [0, 1].

    The backbone's preprocess_input is applied separately in the model
    (via the _shared.py builder), so we do NOT apply it here to keep
    the augmentation step in a clean [0, 1] space.
    """
    image_bytes = tf.io.read_file(path)
    # Decode as RGB directly if the file has 3 channels (colorized MHI),
    # or as grayscale and expand if it is single-channel.
    image = tf.io.decode_png(image_bytes, channels=3)  # always 3-ch output
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    label_one_hot = tf.one_hot(label, depth=num_classes)
    return image, label_one_hot


def build_dataset(
    split: str,
    cfg: Config = DEFAULT_CONFIG,
    processed_dir: str = PROCESSED_DIR,
    shuffle: bool = None,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a tf.data.Dataset for a given split ('train', 'validation', 'test')
    from data/processed/<split>/<class>/*.png.

    Augmentation is applied **only** to the training split and runs in the
    tf.data pipeline (not in the model graph).

    Returns (dataset, num_samples).
    """
    split_dir = os.path.join(processed_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"{split_dir} not found. Run src/preprocessing/preprocessing.py first."
        )

    paths, labels = [], []
    for class_name in cfg.class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".png"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(CLASS_NAMES.index(class_name))

    if not paths:
        raise RuntimeError(f"No MHI images found under {split_dir}.")

    if shuffle is None:
        shuffle = split == "train"

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=cfg.random_seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, l: _preprocess_image(p, l, cfg.image_size, cfg.num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # ── Augmentation: TRAINING ONLY, OUTSIDE the model graph ──────────────
    if cfg.use_augmentation and split == "train":
        aug_model = build_augmentation_pipeline()

        def _augment(image, label):
            return apply_augmentation(image, label, aug_model)

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    # ──────────────────────────────────────────────────────────────────────

    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)
