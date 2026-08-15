"""
Respiration detector tests.

The distinction these lock down is not "does it find a sine wave" but the one
that matters clinically: absence of breathing and inability to measure
breathing must never produce the same answer, because only one of them is
allowed to raise a critical alert.
"""
import numpy as np

from medisense.config import Thresholds
from medisense.detectors.breathing import (
    BREATHING,
    NO_BREATHING,
    READING,
    UNVERIFIED,
    BreathingDetector,
)
from conftest import make_landmarks

FPS = 15.0
BASE_LEVEL = 90
# Chest-region mean intensity moves by well under a gray level when someone
# breathes; tens of levels means the patient is moving, not breathing.
BREATH_AMPLITUDE = 2.0


def torso_landmarks(cx=0.5, cy=0.5, span=0.22, torso=0.30):
    """Landmarks with a chest region big enough to sample."""
    return make_landmarks({
        11: (cx - span / 2, cy),
        12: (cx + span / 2, cy),
        23: (cx - span / 2, cy + torso),
        24: (cx + span / 2, cy + torso),
    })


def frame_with_chest(level: float, rng=None, base=BASE_LEVEL, size=(240, 320)) -> np.ndarray:
    """
    Synthetic frame at a given chest brightness.

    Includes sensor noise by default, because a real camera always has some
    and its absence is what the frozen-source check keys on.
    """
    frame = np.full((size[0], size[1], 3), base + level, dtype=np.float64)
    if rng is not None:
        frame += rng.normal(0.0, 1.5, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def run(det, cfg, signal_fn, seconds=None, lms=None, base=BASE_LEVEL, noisy=True):
    """Feed frames for a span of source time, returning the final result."""
    seconds = seconds or cfg.respiration_window_sec + 1.0
    lms = lms if lms is not None else torso_landmarks()
    rng = np.random.default_rng(1234) if noisy else None
    result = None
    for i in range(int(seconds * FPS)):
        t = i / FPS
        result = det.update(frame_with_chest(signal_fn(t), rng, base), lms, t)
    return result


def test_periodic_chest_signal_reads_as_breathing():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    # 15 breaths/min = 0.25 Hz, comfortably inside the plausible band.
    result = run(det, cfg, lambda t: BREATH_AMPLITUDE * np.sin(2 * np.pi * 0.25 * t))
    assert result.state == BREATHING
    assert 10 <= result.bpm <= 22, result.bpm
    assert result.confidence >= cfg.respiration_conf_threshold


def test_motionless_chest_on_a_live_camera_reads_as_no_breathing():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    result = run(det, cfg, lambda t: 0.0)
    assert result.state == NO_BREATHING


def test_frozen_frames_are_unverified_not_absent_breathing():
    """A stuck stream is the dangerous case: it looks exactly like apnoea."""
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    lms = torso_landmarks()
    frozen = frame_with_chest(0.0)
    result = None
    for i in range(int(cfg.respiration_window_sec * FPS) + 30):
        result = det.update(frozen, lms, i / FPS)
    assert result.state == UNVERIFIED
    assert "frozen" in result.reason


def test_dark_room_is_unverified():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    result = run(det, cfg, lambda t: 0.0, base=4)
    assert result.state == UNVERIFIED
    assert "dark" in result.reason


def test_missing_chest_landmarks_is_unverified():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    hidden = make_landmarks({i: (0.5, 0.5, 0.0) for i in (11, 12, 23, 24)})
    result = det.update(frame_with_chest(0.0), hidden, 0.0)
    assert result.state == UNVERIFIED
    assert "chest" in result.reason


def test_patient_too_far_away_is_unverified():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    tiny = torso_landmarks(span=0.01, torso=0.01)
    result = det.update(frame_with_chest(0.0), tiny, 0.0)
    assert result.state == UNVERIFIED


def test_partial_window_reports_reading_not_a_verdict():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    result = run(det, cfg, lambda t: BREATH_AMPLITUDE * np.sin(2 * np.pi * 0.25 * t), seconds=3.0)
    assert result.state == READING


def test_horizontal_patient_is_measurable():
    """Torso axis across the frame, as filmed from overhead or the side."""
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    sideways = make_landmarks({
        11: (0.35, 0.45),
        12: (0.35, 0.62),
        23: (0.68, 0.45),
        24: (0.68, 0.62),
    })
    result = run(det, cfg, lambda t: BREATH_AMPLITUDE * np.sin(2 * np.pi * 0.25 * t), lms=sideways)
    assert result.state == BREATHING


def test_gross_motion_is_unverified_not_breathing():
    cfg = Thresholds()
    det = BreathingDetector(cfg)
    rng = np.random.default_rng(0)
    result = run(det, cfg, lambda t: float(rng.integers(-120, 120)))
    assert result.state == UNVERIFIED
