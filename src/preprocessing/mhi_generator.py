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
    applied_thresholds = []  # track what was actually used per frame, useful for debugging/logging

    morph_kernel = None
    if morph_open_kernel_size and morph_open_kernel_size > 0:
        morph_kernel = np.ones((morph_open_kernel_size, morph_open_kernel_size), np.uint8)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        curr_gray = _preprocess_frame(frame, resize_to)

        # Frame differencing -> binary motion mask
        diff = cv2.absdiff(curr_gray, prev_gray)

        if use_auto_threshold:
            otsu_value, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            effective_threshold = max(otsu_value, min_threshold)
            _, motion_mask_img = cv2.threshold(diff, effective_threshold, 255, cv2.THRESH_BINARY)
            applied_thresholds.append(effective_threshold)
        else:
            _, motion_mask_img = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        if morph_kernel is not None:
            motion_mask_img = cv2.morphologyEx(motion_mask_img, cv2.MORPH_OPEN, morph_kernel)

        motion_mask = motion_mask_img.astype(bool)

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
    threshold: Union[int, str] = "auto",
    decay: float = 1.0,
    resize_to: int = 224,
    min_threshold: int = 12,
    morph_open_kernel_size: int = 3,
) -> np.ndarray:
    """Convenience wrapper: compute + save in one call. Returns the MHI array."""
    mhi = compute_mhi(
        video_path, tau=tau, threshold=threshold, decay=decay, resize_to=resize_to,
        min_threshold=min_threshold, morph_open_kernel_size=morph_open_kernel_size,
    )
    save_mhi(mhi, output_path)
    return mhi
