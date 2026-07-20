"""
logger.py
=========
Thin wrapper around the standard logging module so every script in the
pipeline logs consistently to both console and a run-specific file under
outputs/logs/. Keeping this centralized means training runs are always
auditable after the fact -- important for a research pipeline where you
need to explain exactly what happened in run #7 six months later.
"""

import logging
import os
import sys
from datetime import datetime

from src.utils.config import OUTPUTS_DIR


def get_logger(name: str, run_tag: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. imported twice) -- don't duplicate handlers.
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    log_dir = os.path.join(OUTPUTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_tag}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
