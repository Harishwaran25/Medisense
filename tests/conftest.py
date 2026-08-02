"""Fake landmark helpers so detector logic can be tested without a camera
or MediaPipe/YOLO actually running."""
import pytest
from medisense.config import Thresholds


class FakeLandmark:
    def __init__(self, x=0.5, y=0.5, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(overrides: dict = None) -> list:
    """33 MediaPipe-compatible pose landmarks, all defaulted to center, with any
    indices in `overrides` (idx -> (x, y[, visibility])) customized."""
    overrides = overrides or {}
    lms = [FakeLandmark() for _ in range(33)]
    for idx, vals in overrides.items():
        x, y = vals[0], vals[1]
        vis = vals[2] if len(vals) > 2 else 1.0
        lms[idx] = FakeLandmark(x=x, y=y, visibility=vis)
    return lms


def lying_patient(cx=0.5, cy=0.45, spread=0.18) -> list:
    """Typical side-camera lying torso (shoulders ≈ hips in y)."""
    return make_landmarks({
        0: (cx, cy - 0.05),
        2: (cx - 0.02, cy - 0.06),
        5: (cx + 0.02, cy - 0.06),
        7: (cx - 0.04, cy - 0.05),
        8: (cx + 0.04, cy - 0.05),
        11: (cx - spread / 2, cy),
        12: (cx + spread / 2, cy),
        13: (cx - spread / 2 - 0.05, cy + 0.02),
        14: (cx + spread / 2 + 0.05, cy + 0.02),
        15: (cx - spread / 2 - 0.08, cy + 0.04),
        16: (cx + spread / 2 + 0.08, cy + 0.04),
        23: (cx - spread / 2, cy + 0.02),
        24: (cx + spread / 2, cy + 0.02),
    })


@pytest.fixture
def cfg():
    return Thresholds(
        fall_calibration_frames=5,
        posture_calibration_frames=5,
        smooth_window=3,
    )
