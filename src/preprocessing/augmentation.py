"""
augmentation.py
================
Augmentation for MHI images, applied ONLY to the training split.

Design notes / what NOT to do
------------------------------
An MHI already encodes direction and recency of motion as pixel
brightness. Some common image augmentations would corrupt that meaning
and should be avoided or used cautiously:

  - Horizontal flip: generally SAFE here, because the KOA-PD-NM dataset
    already contains both left-to-right and right-to-left walks of the
    same subjects, so a flipped MHI still looks like a physically
    plausible gait pattern. Kept on by default; disable via
    `allow_horizontal_flip=False` if your data doesn't have that property.
  - Vertical flip: NOT used -- a gait pattern flipped upside-down is not
    a realistic sample and would just teach the network noise.
  - Large rotations: NOT used beyond a few degrees -- gait videos are
    recorded with a roughly fixed, level camera, so large rotations
    create unrealistic scenes.
  - Aggressive brightness/contrast jitter: mild only. MHI brightness
    encodes recency-of-motion information, not scene illumination, so we
    keep these small (+-10%) rather than the wide ranges typical for
    natural-image classification.
  - Additive Gaussian noise: included, mirroring the augmentation used in
    the base paper on their silhouette sequences, adapted here to single
    MHI images.

Only the TRAINING split is augmented. Validation and test data must
reflect the true underlying distribution the model will face at
inference time -- augmenting them would give an optimistic and invalid
estimate of generalization performance.
"""

import tensorflow as tf


def build_augmentation_pipeline(
    allow_horizontal_flip: bool = True,
    rotation_factor: float = 0.02,     # ~7 degrees max
    zoom_factor: float = 0.05,
    contrast_factor: float = 0.10,
    gaussian_noise_stddev: float = 0.02,  # relative to [0,1]-scaled pixel values
) -> tf.keras.Sequential:
    layers = []
    if allow_horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    layers.append(tf.keras.layers.RandomRotation(rotation_factor, fill_mode="constant"))
    layers.append(tf.keras.layers.RandomZoom(zoom_factor, fill_mode="constant"))
    layers.append(tf.keras.layers.RandomContrast(contrast_factor))
    layers.append(tf.keras.layers.GaussianNoise(gaussian_noise_stddev))
    return tf.keras.Sequential(layers, name="mhi_augmentation")
