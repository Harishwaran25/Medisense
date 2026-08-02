"""
Posture change detection (sitting up / rolling over).

Calibrates a baseline nose-height and shoulder-tilt over the first
N frames, then flags deviation from that baseline. Because it's
baseline-relative, the patient must stay in their initial resting
position during calibration for accurate readings.
"""
import numpy as np

from medisense.smoothing import Smoother, LabelSmoother
from medisense.config import Thresholds


class PostureDetector:
    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self.cal = []
        self.ready = False
        self.base_ny = None
        self.base_t = None
        self.lbl_smooth = LabelSmoother(12, initial="LYING")
        self.nc_smooth = Smoother(8)
        self.tc_smooth = Smoother(8)

    def _tilt(self, landmarks) -> float:
        dy = abs(landmarks[11].y - landmarks[12].y)
        dx = abs(landmarks[11].x - landmarks[12].x)
        return dy / dx if dx > 0.01 else 0.0

    def update(self, landmarks) -> tuple[str, float]:
        ny = landmarks[0].y
        t = self._tilt(landmarks)

        if not self.ready:
            self.cal.append((ny, t))
            if len(self.cal) >= self.cfg.posture_calibration_frames:
                self.base_ny = float(np.mean([f[0] for f in self.cal]))
                self.base_t = float(np.mean([f[1] for f in self.cal]))
                self.ready = True
            return "CALIBRATING", 0.0

        nc = self.nc_smooth.update(self.base_ny - ny)
        tc = self.tc_smooth.update(abs(t - self.base_t))

        if nc > self.cfg.sitting_up_threshold:
            raw = "SITTING UP"
        elif tc > self.cfg.rolling_threshold:
            raw = "ROLLING"
        else:
            raw = "LYING"

        return self.lbl_smooth.update(raw), tc
