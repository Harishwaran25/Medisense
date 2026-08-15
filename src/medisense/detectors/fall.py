"""
Fall detection for a *bedridden* patient.

Lying flat is the expected resting state — NOT a fall.

A fall / off-bed event is inferred from deviation vs a short lying
baseline:
  - torso center drops significantly toward the floor (higher image-y)
  - OR large sudden torso displacement (slide / tumble)
  - OR person leaves the calibrated bed horizontal band

The baseline is the whole detector, so it is maintained rather than frozen:

  - A calibration window the patient moved through is rejected instead of
    averaged. Baselining on a transition (or on an empty bed) biases every
    later measurement for the rest of the session.
  - Once accepted, the baseline drifts toward the patient's current position,
    so raising the bed head or a nurse repositioning the patient does not
    permanently offset every reading.
  - Drift requires the patient to have been *parked* — near-zero spread over a
    window of frames — not merely moving slowly. A gradual slide off the bed
    has a tiny per-frame speed but a large windowed displacement, so gating on
    instantaneous speed alone lets the baseline chase the patient down and the
    fall is never detected at all. Drift also stops once the deviation passes
    halfway to the alert threshold.
"""
from __future__ import annotations

from collections import deque

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
        self.recent: deque[tuple[float, float]] = deque(
            maxlen=max(2, cfg.fall_drift_stable_frames)
        )

    def recalibrate(self) -> None:
        """Re-arm the baseline, e.g. after the bed or camera is moved."""
        self.cal.clear()
        self.recent.clear()
        self.ready = False
        self.base_cx = None
        self.base_cy = None
        self.prev_c = None
        self.lbl_smooth = LabelSmoother(10, initial="SAFE")

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
            self._calibrate(cx, cy)
            self.prev_c = (cx, cy)
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
        fallen = self.lbl_smooth.update(label) == "FALLEN"

        self.recent.append((cx, cy))
        if not fallen:
            self._drift_baseline(cx, cy, drop, lateral)
        return fallen, ratio

    def _calibrate(self, cx: float, cy: float) -> None:
        self.cal.append((cx, cy))
        if len(self.cal) < self.cfg.fall_calibration_frames:
            return

        xs = [c[0] for c in self.cal]
        ys = [c[1] for c in self.cal]
        if max(float(np.std(xs)), float(np.std(ys))) > self.cfg.fall_calibration_max_std:
            # Patient moved through the window: drop the older half and retry
            # rather than averaging a transition into the baseline.
            self.cal = self.cal[len(self.cal) // 2:]
            return

        self.base_cx = float(np.mean(xs))
        self.base_cy = float(np.mean(ys))
        self.ready = True

    def _drift_baseline(self, cx: float, cy: float, drop: float, lateral: float) -> None:
        """Absorb a new resting position, but only once the patient is parked."""
        if self.base_cx is None or self.base_cy is None:
            return
        if abs(drop) >= self.cfg.fall_drop_threshold * 0.5:
            return
        if lateral >= self.cfg.fall_lateral_threshold * 0.5:
            return
        if not self._parked():
            return

        a = self.cfg.fall_baseline_drift
        self.base_cx = (1.0 - a) * self.base_cx + a * cx
        self.base_cy = (1.0 - a) * self.base_cy + a * cy

    def _parked(self) -> bool:
        """True when recent positions barely move — stationary, not sliding."""
        if len(self.recent) < self.recent.maxlen:
            return False
        xs = [c[0] for c in self.recent]
        ys = [c[1] for c in self.recent]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return spread <= self.cfg.fall_drift_max_spread
