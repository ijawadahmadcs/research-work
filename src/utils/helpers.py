"""
helpers.py
==========
Small, generic utility functions used across multiple modules
(mostly filesystem / naming helpers for the video -> MHI -> split pipeline).
"""

import glob
import os
import re
from typing import List, Tuple


VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv")


def list_videos(directory: str) -> List[str]:
    """Recursively list every video file under `directory`."""
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(directory, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


def extract_subject_id(filepath: str) -> str:
    """
    Best-effort extraction of a subject identifier from a filename, e.g.:
        'KOA_early_S014_LR.mov' -> 'S014'
        'PD_moderate_subject07_RL.mov' -> 'subject07'

    IMPORTANT: this heuristic is dataset-naming-dependent. If your copy of
    the KOA-PD-NM dataset uses a different naming convention, adjust the
    regex below or (safer) maintain an explicit subjects.csv mapping
    filename -> subject_id -> class -> severity, and have
    preprocessing.py read that CSV instead of parsing filenames.
    Getting this right is what prevents the *same person's* left- and
    right-walking videos from ending up in both train and test.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"(subject[_\-]?\d+|S\d+|P\d+)", base, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: treat the whole stem minus a trailing direction tag as the subject id.
    return re.sub(r"([_\-](LR|RL|L|R))$", "", base, flags=re.IGNORECASE)


def parse_class_and_severity(filepath: str) -> Tuple[str, str]:
    """
    Infer (class_name, severity) from a raw video's path, assuming the
    KOA/data_dir/KOA_EL|KOA_MD|KOA_SV convention shown in the base paper
    (Early / Moderate / Severe). NM videos have no severity.
    """
    parts = [p.upper() for p in filepath.replace("\\", "/").split("/")]
    if "KOA" in parts:
        cls = "KOA"
    elif "PD" in parts:
        cls = "PD"
    else:
        cls = "NM"

    severity = "NA"
    for tag, name in (("_EL", "Early"), ("_MD", "Moderate"), ("_SV", "Severe")):
        if any(tag in p for p in parts):
            severity = name
            break
    return cls, severity


def safe_makedirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
