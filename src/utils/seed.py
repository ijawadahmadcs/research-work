"""
seed.py
=======
Global determinism helper. Reproducibility is a first-class requirement
for a publication-quality pipeline, so every entry point calls
set_global_seed() before touching numpy / tensorflow / random.

Note: full bit-for-bit determinism on GPU is not guaranteed by TF even
with these settings (cuDNN algorithms can be non-deterministic), but this
gets you as close as practically possible and removes all *avoidable*
sources of randomness (split, weight init, augmentation).
"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Opt-in to deterministic ops where TF supports them.
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except ImportError:
        pass
