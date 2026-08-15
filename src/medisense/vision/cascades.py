"""
Haar cascade lookup.

`cv2.data` only exists on the pip `opencv-python` wheels; distro-packaged
OpenCV builds ship the XML files under /usr/share and expose no `cv2.data`
attribute at all. Touching it unguarded turns a missing convenience module
into a crash at startup, so resolution falls back to the usual install roots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger("medisense.vision.cascades")

_FALLBACK_DIRS = (
    "/usr/share/opencv4/haarcascades",
    "/usr/share/opencv/haarcascades",
    "/usr/local/share/opencv4/haarcascades",
    "/usr/share/OpenCV/haarcascades",
)


def cascade_path(filename: str) -> Optional[str]:
    """Absolute path to a bundled cascade XML, or None if it cannot be found."""
    data = getattr(cv2, "data", None)
    roots = []
    if data is not None and getattr(data, "haarcascades", None):
        roots.append(data.haarcascades)
    roots.extend(_FALLBACK_DIRS)

    for root in roots:
        candidate = Path(root) / filename
        if candidate.is_file():
            return str(candidate)
    return None


def load_cascade(filename: str) -> Optional[cv2.CascadeClassifier]:
    """Load a cascade, or return None if it is missing or unreadable."""
    path = cascade_path(filename)
    if path is None:
        logger.warning("Haar cascade %s not found in any known OpenCV data dir.", filename)
        return None
    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        logger.warning("Haar cascade %s found at %s but failed to load.", filename, path)
        return None
    return classifier
