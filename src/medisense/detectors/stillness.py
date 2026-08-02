"""
Stillness / unconsciousness detection.

FIX (from v1): the previous version required `face NOT visible` as an
extra external condition before firing UNCONSCIOUS, on the assumption
that "face visible = patient is conscious." That assumption is false —
an unconscious patient lying face-up toward the camera still has a
visible face. That gate has been removed here.

Instead, this detector is self-contained: it fires UNCONSCIOUS only
when BOTH the body landmarks AND the face position have been still for
the full duration. Tracking face position (not just visibility) still
gives us the benefit the old code wanted — a patient who is awake and
glancing around resets the timer — without the false assumption that
a visible-but-motionless face means consciousness.
"""
from collections import deque
import time
import numpy as np

from medisense.smoothing import Smoother, LabelSmoother
from medisense.config import Thresholds, BODY_POINTS
from medisense.vision.landmarks import snapshot_landmarks


class StillnessDetector:
    def __init__(self, cfg: Thresholds, warmup_frames: int = 60):
        self.cfg = cfg
        self.since = None
        self.prev = None
        self.skip = warmup_frames
        self.lbl_smooth = LabelSmoother(8, initial="ACTIVE")
        self.mv_smooth = Smoother(6)
        self.face_buf = deque(maxlen=6)

    def update(self, landmarks, face_cx=None, face_cy=None) -> tuple[float, str]:
        now = time.time()
        if self.skip > 0:
            self.skip -= 1
            self.prev = snapshot_landmarks(landmarks)
            return 0.0, "READING"

        # Low landmark confidence = body off-screen; can't judge stillness.
        body_visible = all(landmarks[i].visibility > 0.4 for i in [11, 12, 23, 24])
        if not body_visible:
            self.since = None
            self.prev = snapshot_landmarks(landmarks)
            return 0.0, "READING"

        face_moving = False
        if face_cx is not None and face_cy is not None:
            self.face_buf.append((face_cx, face_cy))
            if len(self.face_buf) >= 4:
                xs = [p[0] for p in self.face_buf]
                ys = [p[1] for p in self.face_buf]
                face_spread = np.std(xs) + np.std(ys)
                face_moving = face_spread > self.cfg.face_move_threshold

        if self.prev is not None:
            diffs = [
                np.sqrt(
                    (landmarks[i].x - self.prev[i].x) ** 2
                    + (landmarks[i].y - self.prev[i].y) ** 2
                )
                for i in BODY_POINTS
                if landmarks[i].visibility > 0.3 and self.prev[i].visibility > 0.3
            ]
            mv = self.mv_smooth.update(float(np.mean(diffs)) if diffs else 1.0)

            if mv < self.cfg.movement_threshold and not face_moving:
                if self.since is None:
                    self.since = now
                dur = now - self.since
            else:
                self.since = None
                dur = 0.0
        else:
            dur = 0.0

        self.prev = snapshot_landmarks(landmarks)

        if dur >= self.cfg.unconscious_sec:
            raw = "UNCONSCIOUS"
        elif dur >= self.cfg.unconscious_sec * 0.5:
            raw = "VERY STILL"
        elif dur > 2.0:
            raw = "STILL"
        else:
            raw = "ACTIVE"

        return dur, self.lbl_smooth.update(raw)
