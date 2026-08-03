"""
preprocessing.py
================
Orchestrates the full raw-video -> MHI -> split pipeline:

    1. Walk data/raw_videos/{KOA,PD,NM} and generate one MHI per video.
    2. Build a metadata table (video, subject_id, class, severity).
    3. Split SUBJECT-WISE (not video-wise!) into train/val/test.
    4. Copy each MHI image into data/processed/{train,val,test}/<class>/.

Why subject-wise splitting matters
-----------------------------------
Each subject contributes two videos (walking left-to-right and
right-to-left). If a video-wise random split is used, it is entirely
possible for one direction of a subject's walk to land in the training
set and the other direction of the *same subject* to land in the test
set. The model can then partly "recognize the person" rather than learn
gait pathology in general, inflating test accuracy in a way that will not
generalize -- a classic and easy-to-miss data leakage bug in gait /
biometrics research. Splitting by subject_id first, then assigning every
video belonging to that subject to a single split, closes that hole.
"""

import os
import shutil
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.mhi_generator import generate_mhi_for_video
from src.utils.config import (
    Config, DEFAULT_CONFIG, RAW_VIDEOS_DIR, MHI_IMAGES_DIR, PROCESSED_DIR,
    METADATA_DIR, RAW_CLASS_DIRS,
)
from src.utils.helpers import list_videos, extract_subject_id, parse_class_and_severity, safe_makedirs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_metadata_table(raw_videos_dir: str = RAW_VIDEOS_DIR) -> pd.DataFrame:
    """Scan raw_videos/ and build one row per video with class/subject/severity."""
    rows = []
    for class_name, folder_name in RAW_CLASS_DIRS.items():
        class_dir = os.path.join(raw_videos_dir, folder_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Expected class directory not found, skipping: {class_dir}")
            continue
        for video_path in list_videos(class_dir):
            _, severity = parse_class_and_severity(video_path)
            subject_id = extract_subject_id(video_path)
            rows.append({
                "video_path": video_path,
                "class": class_name,
                "severity": severity,
                "subject_id": subject_id,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No videos found under {raw_videos_dir}. Populate data/raw_videos/"
            f"{{KOA,PD,NM}} with the KOA-PD-NM dataset before running preprocessing."
        )
    logger.info(f"Found {len(df)} videos across {df['subject_id'].nunique()} subjects.")
    return df


def generate_all_mhis(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Generate and save an MHI for every video listed in df; adds an 'mhi_path' column."""
    mhi_paths = []
    for i, row in df.iterrows():
        out_name = os.path.splitext(os.path.basename(row["video_path"]))[0] + ".png"
        out_path = os.path.join(MHI_IMAGES_DIR, row["class"], out_name)
        try:
            generate_mhi_for_video(
                row["video_path"], out_path,
                tau=cfg.mhi_duration, threshold=cfg.binary_threshold,
                decay=cfg.mhi_decay, resize_to=cfg.image_size,
                min_threshold=cfg.mhi_min_threshold,
                morph_open_kernel_size=cfg.mhi_morph_open_kernel,
                colorize=getattr(cfg, "mhi_colorize", False),
            )
            mhi_paths.append(out_path)
        except (IOError, ValueError) as e:
            logger.warning(f"Skipping unreadable/degenerate video {row['video_path']}: {e}")
            mhi_paths.append(None)

        if (i + 1) % 20 == 0:
            logger.info(f"  MHI progress: {i + 1}/{len(df)}")

    df = df.copy()
    df["mhi_path"] = mhi_paths
    n_failed = df["mhi_path"].isna().sum()
    if n_failed:
        logger.warning(f"{n_failed} videos failed MHI generation and will be excluded.")
    return df.dropna(subset=["mhi_path"]).reset_index(drop=True)


def subject_wise_split(
    df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """
    Assign a 'split' column (train/validation/test) by splitting unique
    subject_ids stratified on their (majority) class, then propagating
    that assignment to every video belonging to each subject.
    """
    subjects = df.groupby("subject_id")["class"].agg(lambda s: s.value_counts().idxmax())
    subject_ids = subjects.index.to_numpy()
    subject_classes = subjects.to_numpy()

    train_subj, test_subj = train_test_split(
        subject_ids, test_size=cfg.test_size, random_state=cfg.random_seed,
        stratify=subject_classes,
    )
    # Re-derive classes for the remaining train subjects to stratify the val split too.
    train_classes = subjects.loc[train_subj].to_numpy()
    relative_val_size = cfg.val_size / (1.0 - cfg.test_size)
    train_subj, val_subj = train_test_split(
        train_subj, test_size=relative_val_size, random_state=cfg.random_seed,
        stratify=train_classes,
    )

    split_map: Dict[str, str] = {}
    split_map.update({s: "train" for s in train_subj})
    split_map.update({s: "validation" for s in val_subj})
    split_map.update({s: "test" for s in test_subj})

    df = df.copy()
    df["split"] = df["subject_id"].map(split_map)

    logger.info(
        "Subject-wise split -> train: %d subjects / %d videos | "
        "validation: %d subjects / %d videos | test: %d subjects / %d videos",
        len(train_subj), (df["split"] == "train").sum(),
        len(val_subj), (df["split"] == "validation").sum(),
        len(test_subj), (df["split"] == "test").sum(),
    )
    return df


def materialize_processed_dataset(df: pd.DataFrame, processed_dir: str = PROCESSED_DIR) -> None:
    """Copy each MHI PNG into processed/<split>/<class>/ so Keras'
    image_dataset_from_directory (or ImageDataGenerator.flow_from_directory)
    can consume it directly."""
    for _, row in df.iterrows():
        dest_dir = os.path.join(processed_dir, row["split"], row["class"])
        safe_makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, os.path.basename(row["mhi_path"]))
        shutil.copy2(row["mhi_path"], dest_path)


def run_full_preprocessing(cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """End-to-end: raw videos -> MHIs -> subject-wise split -> processed/ folders + metadata CSVs."""
    df = build_metadata_table()
    df = generate_all_mhis(df, cfg)
    df = subject_wise_split(df, cfg)
    materialize_processed_dataset(df)

    safe_makedirs(METADATA_DIR)
    df.to_csv(os.path.join(METADATA_DIR, "labels.csv"), index=False)
    df[["subject_id", "class"]].drop_duplicates().to_csv(
        os.path.join(METADATA_DIR, "subjects.csv"), index=False
    )
    df.groupby(["class", "severity", "split"]).size().reset_index(name="count").to_csv(
        os.path.join(METADATA_DIR, "dataset_statistics.csv"), index=False
    )
    df[["subject_id", "class", "split"]].drop_duplicates().to_csv(
        os.path.join(METADATA_DIR, "train_val_test_split.csv"), index=False
    )
    logger.info("Preprocessing complete. Metadata written to %s", METADATA_DIR)
    return df


if __name__ == "__main__":
    run_full_preprocessing()
