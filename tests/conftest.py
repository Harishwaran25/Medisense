"""Fake landmark helpers so detector logic can be tested without a camera
or MediaPipe/YOLO actually running."""
import sys
from pathlib import Path

import pytest

# `pytest tests/` should work straight from a clone. The pyproject sets
# `pythonpath`, but that option needs pytest >= 7, so make the layout
# discoverable here too rather than failing on collection.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from medisense.config import Thresholds  # noqa: E402


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


class FakeClock:
    """Manually advanced clock, so duration logic is tested without sleeping."""

    kind = "fake"

    def __init__(self, start: float = 1000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float = 1.0) -> float:
        self.t += seconds
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cfg():
    return Thresholds(
        fall_calibration_frames=5,
        posture_calibration_frames=5,
        smooth_window=3,
    )
