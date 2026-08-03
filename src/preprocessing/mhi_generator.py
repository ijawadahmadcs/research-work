"""
mhi_generator.py
================
Generates Motion History Images (MHIs) from video files using the
Bobick & Davis (2001) formulation with Otsu adaptive thresholding.

Optionally produces a **3-channel jet-colorised** MHI where pixel hue
encodes the recency of motion (bright-hot = recent, dark-cool = old).
Colorised MHIs have been shown to improve CNN discrimination because the
network receives richer gradient information from three channels rather
than a single grayscale intensity map.
"""

import os
from typing import Optional, Union

import cv2
import numpy as np


def compute_mhi(
    video_path: str,
    tau: int = 30,
    threshold: Union[int, str] = "auto",
    decay: float = 1.0,
    resize_to: Optional[int] = None,
    min_threshold: int = 12,
    morph_open_kernel_size: int = 3,
) -> np.ndarray:
    """
    Compute a single-channel (grayscale) MHI array from a video file.

    Parameters
    ----------
    video_path            : path to the source video.
    tau                   : recency window in frames — pixels with motion
                            within the last tau frames remain "hot".
    threshold             : ``"auto"`` = per-frame Otsu threshold (adapts to
                            each video's own lighting/contrast); supply an int
                            to force a fixed threshold.
    decay                 : linear decay per frame (default 1.0 gives a linear
                            ramp from 0 at oldest motion to tau at newest).
    resize_to             : if given, frames are resized to (resize_to × resize_to)
                            before computing the MHI.
    min_threshold         : floor for the auto-Otsu threshold; guards against
                            noise-only frame pairs producing TV-static MHIs.
    morph_open_kernel_size: kernel size for morphological opening to erase
                            isolated noise pixels; 0 disables the step.

    Returns
    -------
    uint8 array of shape (H, W) with values in [0, 255].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError(f"Video appears to have no frames: {video_path}")

    prev_gray = _preprocess_frame(prev_frame, resize_to)
    mhi = np.zeros_like(prev_gray, dtype=np.float32)

    use_auto_threshold = isinstance(threshold, str) and threshold.lower() == "auto"

    morph_kernel = None
    if morph_open_kernel_size and morph_open_kernel_size > 0:
        morph_kernel = np.ones(
            (morph_open_kernel_size, morph_open_kernel_size), np.uint8
        )

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        curr_gray = _preprocess_frame(frame, resize_to)

        # Frame differencing → binary motion mask
        diff = cv2.absdiff(curr_gray, prev_gray)

        if use_auto_threshold:
            otsu_value, _ = cv2.threshold(
                diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            effective_threshold = max(otsu_value, min_threshold)
            _, motion_mask_img = cv2.threshold(
                diff, effective_threshold, 255, cv2.THRESH_BINARY
            )
        else:
            _, motion_mask_img = cv2.threshold(
                diff, threshold, 255, cv2.THRESH_BINARY
            )

        if morph_kernel is not None:
            motion_mask_img = cv2.morphologyEx(
                motion_mask_img, cv2.MORPH_OPEN, morph_kernel
            )

        motion_mask = motion_mask_img.astype(bool)

        # Bobick & Davis recursive MHI update
        mhi[motion_mask] = tau
        mhi[~motion_mask] = np.maximum(mhi[~motion_mask] - decay, 0)

        prev_gray = curr_gray

    cap.release()

    if frame_count < 2:
        raise ValueError(
            f"Video has fewer than 2 usable frames, cannot compute motion: {video_path}"
        )

    return _normalize_to_uint8(mhi, tau)


def colorize_mhi(mhi_gray: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel grayscale MHI to a 3-channel jet-colorised image.

    The jet colormap maps:
      - 0 (no recent motion)  →  dark blue
      - 128 (medium recency)  →  green
      - 255 (very recent)     →  bright red

    This encodes temporal recency information as hue, providing CNNs with
    three independent gradient channels instead of one, which improves
    feature extraction — especially for subtle gait differences.

    Parameters
    ----------
    mhi_gray : uint8 array (H, W) with values in [0, 255].

    Returns
    -------
    uint8 array (H, W, 3) in BGR colour order (OpenCV convention).
    """
    return cv2.applyColorMap(mhi_gray, cv2.COLORMAP_JET)


def _preprocess_frame(frame: np.ndarray, resize_to: Optional[int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize_to is not None:
        gray = cv2.resize(gray, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    # Mild Gaussian blur suppresses camera-sensor noise registering as spurious motion.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _normalize_to_uint8(mhi: np.ndarray, tau: int) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    mhi_scaled = (mhi / float(tau)) * 255.0
    return np.clip(mhi_scaled, 0, 255).astype(np.uint8)


def save_mhi(mhi: np.ndarray, output_path: str) -> None:
    """Save a grayscale or colour MHI array to disk as a PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mhi)


def generate_mhi_for_video(
    video_path: str,
    output_path: str,
    tau: int = 30,
    threshold: Union[int, str] = "auto",
    decay: float = 1.0,
    resize_to: int = 224,
    min_threshold: int = 12,
    morph_open_kernel_size: int = 3,
    colorize: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper: compute MHI, optionally colorize, save, and return.

    Parameters
    ----------
    colorize : if True, save a 3-channel jet-colorised MHI (the saved file
               will have 3 BGR channels); otherwise save a 1-channel grayscale
               MHI.  Both variants are valid inputs for the CNN pipeline
               (the data loader converts grayscale to RGB on the fly if needed).

    Returns
    -------
    The MHI array that was saved (grayscale uint8 if colorize=False,
    BGR uint8 if colorize=True).
    """
    mhi_gray = compute_mhi(
        video_path,
        tau=tau,
        threshold=threshold,
        decay=decay,
        resize_to=resize_to,
        min_threshold=min_threshold,
        morph_open_kernel_size=morph_open_kernel_size,
    )

    if colorize:
        mhi_out = colorize_mhi(mhi_gray)
    else:
        mhi_out = mhi_gray

    save_mhi(mhi_out, output_path)
    return mhi_out
