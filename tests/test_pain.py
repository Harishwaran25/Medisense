"""
Expression channel tests.

The point being locked down: the OpenCV contrast/edge heuristic keeps the
channel alive when the model is unavailable, but it must never be able to
raise a critical alert or be mistaken for the model in the audit trail.
"""
import numpy as np

from medisense.config import Thresholds
from medisense.detectors.pain import PainDetector, cv_distress_score
from conftest import FakeClock


def high_contrast_face(size=96) -> np.ndarray:
    """Maximal contrast and edge density — the heuristic's strongest input."""
    face = np.zeros((size, size, 3), dtype=np.uint8)
    face[::2] = 255
    return face


def make_detector(cfg=None, clock=None):
    det = PainDetector(cfg or Thresholds(), clock=clock or FakeClock())
    det._worker_loop = lambda: None      # keep model loading out of the test
    return det


def test_cv_distress_score_is_bounded():
    label, score = cv_distress_score(high_contrast_face())
    assert 0.0 <= score <= 1.0
    assert label in ("tense", "calm")


def test_heuristic_fallback_never_reaches_critical():
    cfg = Thresholds(pain_hold_seconds=0.0)
    clock = FakeClock()
    det = make_detector(cfg, clock)
    try:
        det._use_cv = True
        det._pipe = None
        det.mode = "cv_fallback"
        for _ in range(20):
            clock.advance(1.0)
            det._run_inference(high_contrast_face())
        _, score, state = det.get()
        assert score > 0
        assert state in ("NORMAL", "WARNING")
    finally:
        det.close()


def test_neural_path_can_reach_critical_when_sustained():
    cfg = Thresholds(pain_hold_seconds=5.0, critical_pain_threshold=0.7)
    clock = FakeClock()
    det = make_detector(cfg, clock)
    try:
        det._use_cv = False
        det._pipe = object()
        det._neural_score = lambda img: ("sad", 0.95, 0.9)
        for _ in range(10):
            clock.advance(1.0)
            det._run_inference(high_contrast_face())
        assert det.get()[2] == "CRITICAL"
    finally:
        det.close()


def test_neural_path_needs_the_hold_period():
    cfg = Thresholds(pain_hold_seconds=6.0, critical_pain_threshold=0.7)
    clock = FakeClock()
    det = make_detector(cfg, clock)
    try:
        det._use_cv = False
        det._pipe = object()
        det._neural_score = lambda img: ("sad", 0.95, 0.9)
        for _ in range(3):
            clock.advance(1.0)
            det._run_inference(high_contrast_face())
        assert det.get()[2] == "WARNING"
    finally:
        det.close()


def test_mode_is_reported_so_callers_can_tell_the_paths_apart():
    det = make_detector()
    try:
        det.mode = "cv_fallback"
        assert det.get_mode() == "cv_fallback"
        det.mode = "neural"
        assert det.get_mode() == "neural"
    finally:
        det.close()


def test_undersized_face_crops_are_ignored():
    det = make_detector()
    try:
        det._ready = True
        det.update(np.zeros((8, 8, 3), dtype=np.uint8))
        assert det._jobs.empty()
    finally:
        det.close()


def test_landmark_derived_crops_are_never_scored():
    """
    A synthetic crop is a square guessed around the nose. Scoring it invents
    expressions from pillows and bedding.
    """
    det = make_detector()
    try:
        det._ready = True
        det.update(high_contrast_face(), is_detection=False)
        assert det._jobs.empty()

        det.update(high_contrast_face(), is_detection=True)
        assert not det._jobs.empty()
    finally:
        det.close()


def test_blurred_crops_are_rejected():
    det = make_detector()
    try:
        det._ready = True
        flat = np.full((96, 96, 3), 120, dtype=np.uint8)
        det.update(flat, is_detection=True)
        assert det._jobs.empty()
    finally:
        det.close()


def test_low_confidence_prediction_reports_uncertain_not_a_label():
    cfg = Thresholds(emotion_min_confidence=0.5)
    det = make_detector(cfg)
    try:
        det._use_cv = False
        det._pipe = object()
        # A flat seven-way split: no emotion is actually indicated.
        det._neural_score = lambda img: ("sad", 0.62, 0.18)
        det._run_inference(high_contrast_face())
        emotion, _, state = det.get()
        assert emotion == "uncertain"
        assert state == "NORMAL"
    finally:
        det.close()


def test_label_and_score_describe_the_same_prediction():
    """The panel used to be able to read "NEUTRAL 61%"."""
    cfg = Thresholds()
    det = make_detector(cfg)
    try:
        det._use_cv = False
        det._pipe = lambda img: [
            {"label": "neutral", "score": 0.30},
            {"label": "sad", "score": 0.26},
            {"label": "fear", "score": 0.24},
            {"label": "disgust", "score": 0.20},
        ]
        top, distress, confidence = det._neural_score(high_contrast_face())
        assert top == "neutral"
        assert confidence == 0.30
        # Below the confidence gate, so this never becomes a displayed reading.
        assert confidence < cfg.emotion_min_confidence
        assert 0.0 <= distress <= 1.0
    finally:
        det.close()


def test_state_goes_stale_instead_of_reporting_a_calm_patient():
    cfg = Thresholds(pain_stale_sec=5.0)
    clock = FakeClock()
    det = make_detector(cfg, clock)
    try:
        det._use_cv = False
        det._pipe = object()
        det._neural_score = lambda img: ("happy", 0.05, 0.9)
        det._run_inference(high_contrast_face())
        assert det.get()[2] == "NORMAL"

        clock.advance(30.0)
        emotion, score, state = det.get()
        assert state == "NOT ASSESSED"
        assert emotion == "no face"
        assert score == 0.0
    finally:
        det.close()
