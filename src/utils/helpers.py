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
    Extract a **person-level** subject identifier so that BOTH walking-direction
    videos of the same participant (e.g. ``001_NM_01.MOV`` and
    ``001_NM_02.MOV``) map to the **same** subject ID and therefore land in the
    same train/val/test split.

    This is the single most important function for preventing data leakage: if
    each video is treated as its own "subject", the two walks of the same
    person can end up on opposite sides of the split, inflating test accuracy
    because the model can partly memorise the person's body shape rather than
    their gait pathology.

    Strategy (applied in order):
    1. Match the conventional ``NNN_CLASS[_severity]`` prefix pattern used by
       this dataset (e.g. ``001_NM``, ``005_KOA_EL``, ``003_PD_MD``).
       The trailing ``_01``/``_02`` walk-direction tag is stripped so both
       walks share the same key.
    2. Fall back to any ``S\\d+`` / ``P\\d+`` / ``subject\\d+`` token for
       datasets that use explicit subject tags.
    3. Last resort: strip a trailing ``_LR|_RL|_L|_R|_01|_02`` direction tag
       from the full stem and use the remainder.

    Adjust the regexes below if your copy of the dataset uses a different
    naming convention, or (safer) maintain an explicit subjects.csv that maps
    filename -> subject_id, and have preprocessing.py read that CSV instead.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]

    # -------------------------------------------------------------------
    # Strategy 1: NNN_CLASS[_SEVERITY][_01|_02] pattern
    # Matches: 001_NM_01, 001_NM_02, 001_KOA_01_EL, 003_PD_02_MD …
    # Captures:  001_NM,               001_KOA,          003_PD
    # (the trailing _01/_02 walk-direction index is intentionally excluded)
    # -------------------------------------------------------------------
    m = re.match(
        r"^(\d+_(?:NM|KOA|PD))(?:_\d{2})?(?:_[A-Z]{2,})*$",
        base,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # -------------------------------------------------------------------
    # Strategy 2: explicit subject token (S014, P07, subject_03 …)
    # -------------------------------------------------------------------
    m = re.search(r"(subject[_\-]?\d+|S\d+|P\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # -------------------------------------------------------------------
    # Strategy 3: strip trailing walk-direction / take tag and use the rest
    # -------------------------------------------------------------------
    stem = re.sub(r"([_\-](?:LR|RL|L|R|\d{2}))$", "", base, flags=re.IGNORECASE)
    return stem.upper()


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
