"""Bed edge proximity check based on torso center position."""
from __future__ import annotations

import numpy as np

from medisense.config import Thresholds


def check_boundary(landmarks, cfg: Thresholds) -> tuple[str, float, float]:
    """
    Returns (status, center_x, center_y) where center_x/y are normalized
    [0, 1] frame coordinates of the torso center (mean of shoulders + hips).
    """
    try:
        idxs = [11, 12, 23, 24]
        pts = [
            landmarks[i]
            for i in idxs
            if i < len(landmarks) and getattr(landmarks[i], "visibility", 1.0) > 0.2
        ]
        if len(pts) < 2:
            return "CENTER", 0.5, 0.5
        cx = float(np.mean([p.x for p in pts]))
        cy = float(np.mean([p.y for p in pts]))
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
    except (IndexError, AttributeError, TypeError, ValueError):
        return "CENTER", 0.5, 0.5

    if cx < cfg.bed_margin:
        return "LEFT EDGE", cx, cy
    if cx > 1.0 - cfg.bed_margin:
        return "RIGHT EDGE", cx, cy
    return "CENTER", cx, cy
