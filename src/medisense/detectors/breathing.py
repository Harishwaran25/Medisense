"""
Respiration estimate from chest-region intensity.

Stillness on its own cannot separate sleep from unconsciousness — a sleeping
patient is motionless for hours, which is exactly why a stillness timer alone
produces continuous false criticals overnight. Breathing is the signal that
distinguishes them: the chest keeps moving when the patient does not.

Method: track mean intensity of a chest ROI over a rolling window, detrend it,
and look for a dominant frequency inside the plausible respiration band. Small
periodic depth/shading changes from chest rise and fall show up there.

This is deliberately conservative about its own competence. When the ROI is
missing, the room is too dark, the frames are duplicates (frozen stream), or
the patient is moving enough to swamp the signal, it reports UNVERIFIED rather
than guessing. "No breathing detected" and "cannot tell" must never collapse
into the same answer in a system that escalates on the former.

Not a certified respiration monitor. Camera-based respiration is sensitive to
lighting, bedding, camera angle, and distance, and is unvalidated here.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from medisense.config import Thresholds

logger = logging.getLogger("medisense.detectors.breathing")

READING = "READING"
BREATHING = "BREATHING"
NO_BREATHING = "NO BREATHING"
UNVERIFIED = "UNVERIFIED"

_MIN_ROI_PX = 24
_MIN_SAMPLES = 24


@dataclass
class BreathingResult:
    state: str = READING
    bpm: float = 0.0
    confidence: float = 0.0
    reason: str = "warming up"


class BreathingDetector:
    """
    Feed one frame per tick via `update`. Results become meaningful once the
    rolling window covers `cfg.respiration_window_sec` of source time.
    """

    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self.samples: deque[tuple[float, float]] = deque()
        self.result = BreathingResult()
        self._prev_roi: Optional[np.ndarray] = None
        self._duplicates: deque[bool] = deque(maxlen=120)

    def reset(self) -> None:
        self.samples.clear()
        self._prev_roi = None
        self._duplicates.clear()
        self.result = BreathingResult()

    def update(self, frame_bgr, landmarks, now: float) -> BreathingResult:
        roi = self._chest_roi(frame_bgr, landmarks)
        if roi is None:
            self._decay(now, "chest region not visible")
            return self.result

        duplicate = (
            self._prev_roi is not None
            and self._prev_roi.shape == roi.shape
            and np.array_equal(self._prev_roi, roi)
        )
        self._duplicates.append(duplicate)
        self._prev_roi = roi

        # A live sensor never produces bit-identical frames, even pointed at a
        # motionless chest — there is always read noise. Near-total pixel
        # identity therefore means the source is stuck, which is the one case
        # that must not be reported as absent breathing. Measured as a fraction
        # of the window rather than a consecutive run, because a genuine
        # low-amplitude signal quantised to 8 bits repeats values in bursts
        # around its peaks.
        if len(self._duplicates) >= self.cfg.respiration_duplicate_frames:
            duplicate_fraction = sum(self._duplicates) / len(self._duplicates)
            if duplicate_fraction >= self.cfg.respiration_duplicate_fraction:
                self._decay(now, "frame source appears frozen")
                return self.result

        gray = np.asarray(roi, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        if float(gray.mean()) < self.cfg.respiration_min_luma:
            self._decay(now, "chest region too dark to measure")
            return self.result

        self.samples.append((now, float(gray.mean())))
        window = self.cfg.respiration_window_sec
        while self.samples and now - self.samples[0][0] > window:
            self.samples.popleft()

        span = self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.0
        if len(self.samples) < _MIN_SAMPLES or span < window * 0.8:
            self.result = BreathingResult(READING, 0.0, 0.0, "collecting respiration window")
            return self.result

        self.result = self._analyse(span)
        return self.result

    def _decay(self, now: float, reason: str) -> None:
        """Drop stale samples and report that respiration cannot be measured."""
        window = self.cfg.respiration_window_sec
        while self.samples and now - self.samples[0][0] > window:
            self.samples.popleft()
        self.result = BreathingResult(UNVERIFIED, 0.0, 0.0, reason)

    def _analyse(self, span: float) -> BreathingResult:
        times = np.array([t for t, _ in self.samples], dtype=np.float64)
        values = np.array([v for _, v in self.samples], dtype=np.float64)

        # Resample onto a uniform grid; frame arrival is never perfectly even.
        n = len(values)
        uniform_t = np.linspace(times[0], times[-1], n)
        values = np.interp(uniform_t, times, values)

        # Remove lighting drift so the transform sees oscillation, not slope.
        values = values - np.polyval(np.polyfit(uniform_t, values, 1), uniform_t)

        amplitude = float(np.std(values))
        if amplitude < self.cfg.respiration_min_amplitude:
            return BreathingResult(
                NO_BREATHING, 0.0, 0.0, f"no chest oscillation (amplitude {amplitude:.4f})"
            )
        if amplitude > self.cfg.respiration_max_amplitude:
            return BreathingResult(
                UNVERIFIED, 0.0, 0.0, f"signal swamped by motion (amplitude {amplitude:.3f})"
            )

        fs = (n - 1) / span
        spectrum = np.abs(np.fft.rfft(values * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        low = self.cfg.respiration_min_bpm / 60.0
        high = self.cfg.respiration_max_bpm / 60.0
        band = (freqs >= low) & (freqs <= high)
        # Compare against everything physiologically plausible, excluding DC.
        reference = (freqs > 0.02) & (freqs <= max(high * 3.0, 2.0))

        if not band.any() or not reference.any():
            return BreathingResult(
                UNVERIFIED, 0.0, 0.0, f"window too short to resolve {self.cfg.respiration_min_bpm:.0f} bpm"
            )

        band_power = float(np.sum(spectrum[band] ** 2))
        total_power = float(np.sum(spectrum[reference] ** 2))
        confidence = band_power / total_power if total_power > 0 else 0.0
        peak_freq = float(freqs[band][int(np.argmax(spectrum[band]))])
        bpm = peak_freq * 60.0

        if confidence < self.cfg.respiration_conf_threshold:
            return BreathingResult(
                NO_BREATHING, bpm, confidence,
                f"no dominant respiration frequency (confidence {confidence:.2f})",
            )
        return BreathingResult(BREATHING, bpm, confidence, f"{bpm:.0f} bpm")

    def _chest_roi(self, frame_bgr, landmarks) -> Optional[np.ndarray]:
        """
        Box centred on the chest, sized from the patient's own dimensions.

        Deliberately orientation-independent. A bedridden patient filmed from
        the side or overhead often lies with their torso axis running across
        the frame rather than down it, so anything defined as "shoulders down
        to mid-torso" collapses to a few pixels exactly when it is needed.
        The box is placed along the shoulder-to-hip vector instead.
        """
        if frame_bgr is None or landmarks is None:
            return None
        try:
            h, w = frame_bgr.shape[:2]
        except Exception:
            return None
        if h < _MIN_ROI_PX or w < _MIN_ROI_PX:
            return None

        try:
            pts = [landmarks[i] for i in (11, 12, 23, 24)]
        except (IndexError, TypeError):
            return None
        if any(float(getattr(p, "visibility", 0.0) or 0.0) < 0.4 for p in pts):
            return None

        sx = (float(pts[0].x) + float(pts[1].x)) / 2.0
        sy = (float(pts[0].y) + float(pts[1].y)) / 2.0
        hx = (float(pts[2].x) + float(pts[3].x)) / 2.0
        hy = (float(pts[2].y) + float(pts[3].y)) / 2.0

        torso = float(np.hypot(hx - sx, hy - sy))
        shoulder_span = float(np.hypot(float(pts[0].x) - float(pts[1].x),
                                       float(pts[0].y) - float(pts[1].y)))
        scale = max(torso, shoulder_span)
        if scale < 0.05:
            # Patient too small in frame to isolate a chest region.
            return None

        # A third of the way from shoulders to hips is roughly mid-sternum.
        cx = sx + 0.35 * (hx - sx)
        cy = sy + 0.35 * (hy - sy)
        half = 0.45 * scale / 2.0

        px0, px1 = int(max(0, (cx - half) * w)), int(min(w, (cx + half) * w))
        py0, py1 = int(max(0, (cy - half) * h)), int(min(h, (cy + half) * h))
        if px1 - px0 < _MIN_ROI_PX or py1 - py0 < _MIN_ROI_PX:
            return None

        roi = frame_bgr[py0:py1, px0:px1]
        if roi is None or roi.size == 0:
            return None
        return roi.copy()
