"""
video_loader.py
================
Two responsibilities:
  1. Low-level video frame iteration (used by mhi_generator / notebooks
     for visual inspection of a video's frames).
  2. Building tf.data.Dataset objects from the already-generated
     processed/{train,validation,test}/<class>/ MHI image folders, for
     the training stage. This is the "raw video -> MHI" boundary: by the
     time training runs, there are no videos left in the loop, only
     static MHI images, which is exactly the point of the MHI
     replacement for Mask R-CNN + per-frame silhouettes.
"""

import os
from typing import Iterator, Tuple

import cv2
import numpy as np
import tensorflow as tf

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
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_png(image_bytes, channels=1)  # MHI is stored as grayscale
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.grayscale_to_rgb(image)  # backbones expect 3 channels (ImageNet weights)
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
        ds = ds.shuffle(buffer_size=len(paths), seed=cfg.random_seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _preprocess_image(p, l, cfg.image_size, cfg.num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)
