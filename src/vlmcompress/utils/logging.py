from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[str] = None, name: str = "vlmcompress") -> logging.Logger:
    """Create a console+file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
