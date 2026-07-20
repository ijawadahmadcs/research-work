"""
mhi_generator.py
=================
Converts one gait video into a single Motion History Image (MHI).

This module replaces the Mask R-CNN silhouette-segmentation stage of the
base paper. Instead of segmenting a person in every frame, we compute a
2D scalar function that encodes *where and how recently* motion occurred
across the whole video, condensing the full walking sequence into one image.

Mathematical formulation (Bobick & Davis, 2001 -- the standard MHI definition
referenced implicitly whenever a paper says "motion history image"):

    Let D(x, y, t) be a binary motion mask at pixel (x, y), time t:
        D(x, y, t) = 1  if |I(x, y, t) - I(x, y, t-1)| >= threshold
        D(x, y, t) = 0  otherwise

    The MHI H_tau is updated recursively:
        H_tau(x, y, t) = tau            if D(x, y, t) = 1
        H_tau(x, y, t) = max(0, H_tau(x, y, t-1) - decay)   otherwise

    where tau is the "temporal window" (mhi_duration) -- the number of the
    most recent frames a pixel's motion is remembered for -- and decay is the
    per-frame decrement (usually 1, giving a linear ramp from tau down to 0).

Interpreting the result: pixels that moved very recently are bright
(close to tau); pixels that moved a while ago are dimmer; pixels that
never moved are 0 (black). After normalizing to [0, 255] and resizing,
this single grayscale image is a compact spatiotemporal signature of the
entire gait cycle -- brightness gradients trace the direction and recency
of limb motion, which is exactly the information a CNN needs to tell a
shuffling PD gait from an antalgic KOA limp from a normal stride.
"""

import os
from typing import Optional

import cv2
import numpy as np


def compute_mhi(
    video_path: str,
    tau: int = 30,
    threshold: int = 30,
    decay: float = 1.0,
    resize_to: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a single Motion History Image for an entire video.

    Parameters
    ----------
    video_path : path to the input video file (.mov, .mp4, ...)
    tau        : temporal duration -- max "brightness" value in the MHI
    threshold  : absolute pixel-intensity difference above which motion is
                 registered between consecutive frames
    decay      : linear decay applied to the MHI at each frame where no
                 motion is detected at a pixel
    resize_to  : if given, frames are resized to (resize_to, resize_to)
                 BEFORE motion detection (keeps compute cost constant
                 regardless of the source video's resolution)

    Returns
    -------
    mhi : np.ndarray, dtype=uint8, shape (H, W), values in [0, 255]
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

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        curr_gray = _preprocess_frame(frame, resize_to)

        # Frame differencing -> binary motion mask
        diff = cv2.absdiff(curr_gray, prev_gray)
        motion_mask = diff >= threshold

        # Recursive MHI update (Bobick & Davis formulation)
        mhi[motion_mask] = tau
        mhi[~motion_mask] = np.maximum(mhi[~motion_mask] - decay, 0)

        prev_gray = curr_gray

    cap.release()

    if frame_count < 2:
        raise ValueError(
            f"Video has fewer than 2 usable frames, cannot compute motion: {video_path}"
        )

    # Normalize to [0, 255] for storage / CNN input.
    mhi_normalized = _normalize_to_uint8(mhi, tau)
    return mhi_normalized


def _preprocess_frame(frame: np.ndarray, resize_to: Optional[int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize_to is not None:
        gray = cv2.resize(gray, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    # Mild blur reduces camera-sensor noise from registering as spurious "motion".
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _normalize_to_uint8(mhi: np.ndarray, tau: int) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    mhi_scaled = (mhi / float(tau)) * 255.0
    return np.clip(mhi_scaled, 0, 255).astype(np.uint8)


def save_mhi(mhi: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mhi)


def generate_mhi_for_video(
    video_path: str,
    output_path: str,
    tau: int = 30,
    threshold: int = 30,
    decay: float = 1.0,
    resize_to: int = 224,
) -> np.ndarray:
    """Convenience wrapper: compute + save in one call. Returns the MHI array."""
    mhi = compute_mhi(video_path, tau=tau, threshold=threshold, decay=decay, resize_to=resize_to)
    save_mhi(mhi, output_path)
    return mhi
