"""
Fall detection for a *bedridden* patient.

Lying flat is the expected resting state — NOT a fall.

A fall / off-bed event is inferred from deviation vs a short lying
baseline:
  - torso center drops significantly toward the floor (higher image-y)
  - OR large sudden torso displacement (slide / tumble)
  - OR person leaves the calibrated bed horizontal band

Requires a brief calibration window while the patient is lying normally.
"""
from __future__ import annotations

import numpy as np

from medisense.smoothing import Smoother, LabelSmoother
from medisense.config import Thresholds


class FallDetector:
    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self.ratio_smooth = Smoother(cfg.smooth_window)
        self.lbl_smooth = LabelSmoother(10, initial="SAFE")
        self.cal: list[tuple[float, float]] = []
        self.ready = False
        self.base_cy: float | None = None
        self.base_cx: float | None = None
        self.prev_c: tuple[float, float] | None = None

    def _torso_center(self, landmarks) -> tuple[float, float] | None:
        try:
            idxs = [11, 12, 23, 24]
            if any(landmarks[i].visibility < 0.3 for i in idxs):
                return None
            cx = float(np.mean([landmarks[i].x for i in idxs]))
            cy = float(np.mean([landmarks[i].y for i in idxs]))
            return cx, cy
        except (IndexError, AttributeError, TypeError):
            return None

    def update(self, landmarks) -> tuple[bool, float]:
        """
        Returns:
            (is_fallen, diagnostic_ratio)
            diagnostic_ratio = |shoulder_y - hip_y| (kept for UI continuity).
        """
        try:
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            raw_ratio = abs(shoulder_y - hip_y)
            ratio = self.ratio_smooth.update(raw_ratio)
        except (IndexError, AttributeError, TypeError):
            return False, 1.0

        center = self._torso_center(landmarks)
        if center is None:
            return False, ratio

        cx, cy = center

        if not self.ready:
            self.cal.append((cx, cy))
            self.prev_c = (cx, cy)
            if len(self.cal) >= self.cfg.fall_calibration_frames:
                self.base_cx = float(np.mean([c[0] for c in self.cal]))
                self.base_cy = float(np.mean([c[1] for c in self.cal]))
                self.ready = True
            return False, ratio

        drop = cy - (self.base_cy or cy)
        lateral = abs(cx - (self.base_cx or cx))
        speed = 0.0
        if self.prev_c is not None:
            speed = float(
                np.sqrt((cx - self.prev_c[0]) ** 2 + (cy - self.prev_c[1]) ** 2)
            )
        self.prev_c = (cx, cy)

        # Off-bed / fall heuristics for a lying patient.
        floor_drop = drop >= self.cfg.fall_drop_threshold
        ejected = lateral >= self.cfg.fall_lateral_threshold and drop >= (
            self.cfg.fall_drop_threshold * 0.5
        )
        tumble = speed >= self.cfg.fall_speed_threshold and drop >= (
            self.cfg.fall_drop_threshold * 0.4
        )

        raw_fallen = floor_drop or ejected or tumble
        label = "FALLEN" if raw_fallen else "SAFE"
        return self.lbl_smooth.update(label) == "FALLEN", ratio
