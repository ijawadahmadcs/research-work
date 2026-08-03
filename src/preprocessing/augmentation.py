"""
augmentation.py
================
Augmentation for MHI images, applied **only** to the training split and
**only** inside the tf.data pipeline (not inside the Keras model graph).

Design notes / what NOT to do
------------------------------
An MHI already encodes direction and recency of motion as pixel brightness.
Some common image augmentations would corrupt that meaning and should be
avoided or used cautiously:

  - Horizontal flip: SAFE here.  The KOA-PD-NM dataset already contains
    both left-to-right and right-to-left walks of the same subjects, so a
    flipped MHI still looks like a physically plausible gait pattern.
  - Vertical flip: NOT used — a gait pattern flipped upside-down is not a
    realistic sample.
  - Large rotations: NOT used beyond a few degrees — gait videos are
    recorded with a roughly fixed, level camera.
  - Aggressive brightness/contrast jitter: mild only.  MHI brightness
    encodes recency-of-motion, not scene illumination, so we keep these
    small rather than the wide ranges typical for natural-image tasks.
  - Translation: small shifts are safe (simulates small camera offsets).
  - Additive Gaussian noise: included, mirrors augmentation used in the
    base paper adapted here to single MHI images.
  - Coarse dropout (random erasing): included at a mild level; forces
    the network not to rely on any single spatial region.

Only the TRAINING split is augmented via tf.data.Dataset.map().
Validation and test data must reflect the true underlying distribution
the model will face at inference time.
"""

import tensorflow as tf


def build_augmentation_pipeline(
    allow_horizontal_flip: bool = True,
    rotation_factor: float = 0.03,        # ≈11° max (was 0.02/7°)
    zoom_factor: float = 0.08,            # ±8% zoom (was 0.05)
    translation_factor: float = 0.05,     # ±5% shift in H and W
    contrast_factor: float = 0.12,        # ±12% contrast
    brightness_factor: float = 0.08,      # ±8% brightness
    gaussian_noise_stddev: float = 0.025, # relative to [0,1] pixels
) -> tf.keras.Sequential:
    """
    Returns a Keras Sequential augmentation model.

    Usage
    -----
    This model should be called **only** in the tf.data training pipeline:

        aug = build_augmentation_pipeline()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    Never call it during validation or test inference.
    """
    layers = []

    if allow_horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))

    layers.append(tf.keras.layers.RandomRotation(
        rotation_factor, fill_mode="reflect"
    ))
    layers.append(tf.keras.layers.RandomZoom(
        zoom_factor, fill_mode="reflect"
    ))
    layers.append(tf.keras.layers.RandomTranslation(
        height_factor=translation_factor,
        width_factor=translation_factor,
        fill_mode="reflect",
    ))
    layers.append(tf.keras.layers.RandomContrast(contrast_factor))
    layers.append(tf.keras.layers.RandomBrightness(
        factor=brightness_factor, value_range=(0.0, 1.0)
    ))
    layers.append(tf.keras.layers.GaussianNoise(gaussian_noise_stddev))

    return tf.keras.Sequential(layers, name="mhi_augmentation")


def apply_augmentation(
    image: tf.Tensor,
    label: tf.Tensor,
    aug_model: tf.keras.Sequential,
) -> tuple:
    """
    Apply augmentation to a single (image, label) pair inside tf.data.map().

    Parameters
    ----------
    image     : float32 tensor in [0, 1], shape (H, W, C)
    label     : one-hot label tensor
    aug_model : the Sequential model returned by build_augmentation_pipeline()

    Returns
    -------
    (augmented_image, label)
    """
    # Expand batch dim required by Keras layers, then squeeze back.
    image = tf.expand_dims(image, 0)
    image = aug_model(image, training=True)
    image = tf.squeeze(image, 0)
    # Clip to valid range in case noise pushed values slightly out of [0, 1].
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
