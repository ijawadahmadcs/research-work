"""
ensemble.py
===========
Late-fusion ensemble utilities for the MHI gait classifier.

Design — Why late fusion?
--------------------------
The original implementation used Keras ``Average()`` to average the
softmax outputs of concurrently trained sub-models — i.e. a jointly
trained ensemble.  This has two problems:

1. **Suboptimal individual models**: each sub-model gets less effective
   learning-rate scheduling because the joint loss gradient is averaged,
   diluting the benefit of the two-stage (head → fine-tune) schedule that
   each backbone needs independently.
2. **No flexibility**: the uniform average is not tuned; a single
   under-performing model drags down the whole ensemble.

Late-fusion (post-hoc ensemble) avoids both issues:
- Each model is trained independently with its own optimal schedule.
- Ensemble weights are tuned on the **validation set** (not the test set).
- Multiple fusion strategies can be compared without re-training.

Functions
---------
soft_vote()           : Uniform-average soft voting.
weighted_soft_vote()  : Weighted-average soft voting (weights sum to 1).
load_all_predictions(): Load all individual-model prediction CSVs.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core fusion functions
# ─────────────────────────────────────────────────────────────────────────────

def soft_vote(
    proba_list: List[np.ndarray],
) -> np.ndarray:
    """
    Uniform soft voting: average class probabilities across all models.

    Parameters
    ----------
    proba_list : list of (N, C) float32 arrays, one per model.

    Returns
    -------
    (N, C) array of averaged probabilities.
    """
    return np.mean(np.stack(proba_list, axis=0), axis=0)


def weighted_soft_vote(
    proba_list: List[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """
    Weighted soft voting: weighted average of class probabilities.

    Parameters
    ----------
    proba_list : list of (N, C) float32 arrays, one per model.
    weights    : sequence of non-negative floats (need not sum to 1;
                 they are normalised internally).

    Returns
    -------
    (N, C) array of weighted-averaged probabilities.
    """
    if len(proba_list) != len(weights):
        raise ValueError(
            f"len(proba_list)={len(proba_list)} != len(weights)={len(weights)}"
        )
    w = np.array(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("Weights must sum to a positive value.")
    w = w / w.sum()

    stacked = np.stack(proba_list, axis=0)  # (M, N, C)
    return np.einsum("m,mnc->nc", w, stacked)


def majority_vote(
    pred_list: List[np.ndarray],
) -> np.ndarray:
    """
    Hard majority voting: each model casts one vote for its argmax class.

    Parameters
    ----------
    pred_list : list of (N,) int arrays of predicted class indices.

    Returns
    -------
    (N,) int array of majority-voted class indices.
    """
    # Shape: (M, N)
    votes = np.stack(pred_list, axis=0)
    n_samples = votes.shape[1]
    result = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        counts = np.bincount(votes[:, i], minlength=votes.max() + 1)
        result[i] = np.argmax(counts)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_predictions(
    predictions_dir: str,
    class_names: List[str],
    backbones: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Load all ``<backbone>_test_predictions.csv`` files from *predictions_dir*.

    Parameters
    ----------
    predictions_dir : directory containing prediction CSVs.
    class_names     : ordered list of class names (e.g. ["NM", "KOA", "PD"]).
    backbones       : if given, only load predictions for those backbones;
                      otherwise discover all CSVs automatically.

    Returns
    -------
    dict mapping backbone_name -> {"y_true": ndarray, "y_pred": ndarray,
                                   "proba": ndarray(N,C)}
    """
    import os
    import pandas as pd

    result = {}
    if backbones is None:
        import glob
        csvs = glob.glob(os.path.join(predictions_dir, "*_test_predictions.csv"))
        backbones = [
            os.path.basename(f).replace("_test_predictions.csv", "")
            for f in csvs
        ]

    for backbone in backbones:
        csv_path = os.path.join(
            predictions_dir, f"{backbone}_test_predictions.csv"
        )
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        y_true = np.array([class_names.index(lbl) for lbl in df["true_label"]])
        y_pred = np.array([class_names.index(lbl) for lbl in df["predicted_label"]])
        proba = df[[f"prob_{c}" for c in class_names]].to_numpy(dtype=np.float32)

        result[backbone] = {"y_true": y_true, "y_pred": y_pred, "proba": proba}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Keras model builder (kept for backwards compatibility / joint-train option)
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble_model(cfg):
    """
    Build a jointly-trained Keras ensemble model (average of member outputs).

    NOTE: This is kept for backwards compatibility.  For research-quality
    evaluation, prefer the late-fusion functions above (soft_vote,
    weighted_soft_vote) which operate on independently trained model outputs.
    """
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    from src.models.densenet import build_densenet_model
    from src.models.efficientnet import build_efficientnet_model
    from src.models.googlenet import build_googlenet_model
    from src.models.mobilenet import build_mobilenet_model
    from src.models.nasnet import build_nasnet_model
    from src.models.swin_transformer import build_swin_transformer_model
    from src.models.vgg16 import build_vgg16_model

    _MEMBER_BUILDERS = {
        "vgg16": build_vgg16_model,
        "mobilenet": build_mobilenet_model,
        "densenet": build_densenet_model,
        "efficientnet": build_efficientnet_model,
        "nasnet": build_nasnet_model,
        "googlenet": build_googlenet_model,
        "inceptionv3": build_googlenet_model,
        "swin": build_swin_transformer_model,
    }

    member_names = cfg.ensemble_backbones or ["mobilenet", "vgg16", "densenet"]
    if len(member_names) < 2:
        raise ValueError("Ensemble mode requires at least two member backbones.")

    input_shape = (cfg.image_size, cfg.image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="mhi_input")

    member_models, member_outputs = [], []
    for member_name in member_names:
        key = member_name.lower()
        if key not in _MEMBER_BUILDERS:
            raise ValueError(
                f"Unknown ensemble member '{member_name}'. "
                f"Choose from {sorted(_MEMBER_BUILDERS)}."
            )
        member_model, _ = _MEMBER_BUILDERS[key](cfg)
        member_models.append(member_model)
        member_outputs.append(member_model(inputs))

    outputs = layers.Average(name="ensemble_average")(member_outputs)
    model = Model(inputs, outputs, name="Ensemble_MHI_Classifier")
    model.member_backbones = [
        getattr(m, "backbone", m) for m in member_models
    ]
    model.member_models = member_models
    return model, model.member_backbones
