"""
Agitation detection.

Tracks average frame-to-frame movement of key body landmarks over a
rolling time window. Sustained high movement suggests restlessness or
distress rather than normal, brief repositioning.
"""
from collections import deque
import numpy as np

from medisense.clock import WallClock
from medisense.smoothing import Smoother, LabelSmoother
from medisense.config import Thresholds, BODY_POINTS
from medisense.vision.landmarks import snapshot_landmarks


class AgitationDetector:
    def __init__(self, cfg: Thresholds, clock=None):
        self.cfg = cfg
        self.clock = clock or WallClock()
        self.history = deque()
        self.prev_lms = None
        self.val_smooth = Smoother(10)
        self.lbl_smooth = LabelSmoother(10, initial="CALM")

    def update(self, landmarks) -> tuple[float, str]:
        now = self.clock.now()
        if self.prev_lms is not None:
            diffs = [
                np.sqrt(
                    (landmarks[i].x - self.prev_lms[i].x) ** 2
                    + (landmarks[i].y - self.prev_lms[i].y) ** 2
                )
                for i in BODY_POINTS
                if landmarks[i].visibility > 0.3 and self.prev_lms[i].visibility > 0.3
            ]
            if diffs:
                self.history.append((now, float(np.mean(diffs))))
        # CRITICAL: copy coords — MediaPipe reuses landmark objects across frames.
        self.prev_lms = snapshot_landmarks(landmarks)

        while self.history and now - self.history[0][0] > self.cfg.agitation_window_sec:
            self.history.popleft()

        if len(self.history) < 5:
            return 0.0, "READING"

        raw_avg = float(np.mean([m for _, m in self.history]))
        avg = self.val_smooth.update(raw_avg)

        if avg >= self.cfg.agitation_high:
            raw_lbl = "HIGH"
        elif avg >= self.cfg.agitation_mild:
            raw_lbl = "MILD"
        else:
            raw_lbl = "CALM"

        return avg, self.lbl_smooth.update(raw_lbl)
