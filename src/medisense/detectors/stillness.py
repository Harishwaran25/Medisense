"""
Stillness interpretation.

History of this detector, because the reasoning matters more than the code:

v1 required `face NOT visible` before firing UNCONSCIOUS, on the assumption
that a visible face means a conscious patient. That is false — an unconscious
patient lying face-up still has a visible face — so the gate was removed and
face *position* tracking took its place, since a patient who is awake and
glancing around moves their head.

v2 then fired CRITICAL "UNCONSCIOUS" after 20 seconds of stillness. That is
worse than the bug it replaced. A sleeping patient is motionless for hours, so
overnight monitoring — the entire point of the system — produced a critical
alert roughly every cooldown period, all night, every night. Any alarm that
cries wolf that reliably gets muted, and a muted monitor detects nothing.

Stillness cannot distinguish sleep from unconsciousness, because they look
identical to a camera watching for movement. Respiration can. So stillness now
reports only how long the patient has been still, and escalation is gated on
the respiration signal:

    still + breathing detected        -> ASLEEP            (normal)
    still + no breathing detected     -> NO RESPIRATION    (critical)
    still + respiration unmeasurable  -> PROLONGED STILL   (warning, and only
                                         after a long delay)

The third case is the honest one. When the chest is occluded, the room is too
dark, or the stream has frozen, the system says it cannot tell instead of
either escalating or staying silent — a nurse can act on "I can't see whether
this patient is breathing" but not on a false critical.
"""
from collections import deque
import numpy as np

from medisense.clock import WallClock
from medisense.smoothing import Smoother, LabelSmoother
from medisense.config import Thresholds, BODY_POINTS
from medisense.detectors import breathing as resp
from medisense.vision.landmarks import snapshot_landmarks

READING = "READING"
ACTIVE = "ACTIVE"
STILL = "STILL"
ASLEEP = "ASLEEP"
PROLONGED_STILL = "PROLONGED STILL"
NO_RESPIRATION = "NO RESPIRATION"


class StillnessDetector:
    def __init__(self, cfg: Thresholds, warmup_frames: int = 60, clock=None):
        self.cfg = cfg
        self.clock = clock or WallClock()
        self.since = None
        self.no_resp_since = None
        self.prev = None
        self.skip = warmup_frames
        self.lbl_smooth = LabelSmoother(8, initial=ACTIVE)
        self.mv_smooth = Smoother(6)
        self.face_buf = deque(maxlen=6)

    def update(self, landmarks, face_cx=None, face_cy=None, breathing_state=None) -> tuple[float, str]:
        """
        Returns (stillness_duration_sec, state).

        `breathing_state` is a `detectors.breathing` state. Passing None means
        respiration is not being measured at all, which is treated the same as
        unmeasurable — never as absent.
        """
        now = self.clock.now()
        if self.skip > 0:
            self.skip -= 1
            self.prev = snapshot_landmarks(landmarks)
            return 0.0, READING

        # Low landmark confidence = body off-screen; can't judge stillness.
        body_visible = all(landmarks[i].visibility > 0.4 for i in [11, 12, 23, 24])
        if not body_visible:
            self._reset_timers()
            self.prev = snapshot_landmarks(landmarks)
            return 0.0, READING

        face_moving = False
        if face_cx is not None and face_cy is not None:
            self.face_buf.append((face_cx, face_cy))
            if len(self.face_buf) >= 4:
                xs = [p[0] for p in self.face_buf]
                ys = [p[1] for p in self.face_buf]
                face_spread = np.std(xs) + np.std(ys)
                face_moving = face_spread > self.cfg.face_move_threshold

        dur = 0.0
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
                dur = max(0.0, now - self.since)
            else:
                self._reset_timers()

        self.prev = snapshot_landmarks(landmarks)
        return dur, self.lbl_smooth.update(self._classify(dur, breathing_state, now))

    def _reset_timers(self) -> None:
        self.since = None
        self.no_resp_since = None

    def _classify(self, dur: float, breathing_state, now: float) -> str:
        if dur < self.cfg.still_sec:
            self.no_resp_since = None
            return ACTIVE

        if breathing_state == resp.NO_BREATHING:
            if self.no_resp_since is None:
                self.no_resp_since = now
            if now - self.no_resp_since >= self.cfg.no_respiration_sec:
                return NO_RESPIRATION
            # Absence is suspected but not yet confirmed for long enough.
            return STILL

        self.no_resp_since = None

        if breathing_state == resp.BREATHING:
            return ASLEEP if dur >= self.cfg.asleep_sec else STILL

        # Respiration unmeasurable: say so, and only after a long wait.
        if dur >= self.cfg.prolonged_stillness_sec:
            return PROLONGED_STILL
        return STILL
