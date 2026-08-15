"""Production-oriented vision / expression safety tests (no camera required)."""
import numpy as np

from medisense.config import Thresholds
from medisense.detectors.pain import cv_distress_score
from medisense.health import run_healthcheck
from medisense.vision.face import FaceFinder
from medisense.vision.landmarks import face_center, landmarks_from_coco
from medisense.vision.pose import PoseEstimator, PoseResult
from conftest import lying_patient


def test_config_validate_ok():
    cfg = Thresholds()
    assert cfg.validate() == []


def test_config_validate_bad_backend():
    cfg = Thresholds(pose_backend="banana")
    assert any("pose_backend" in p for p in cfg.validate())


def test_face_finder_handles_empty_and_tiny():
    ff = FaceFinder()
    assert ff.find(None)[0] is False
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert ff.find(tiny)[0] is False


def test_face_finder_synthetic_from_landmarks():
    ff = FaceFinder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a bright blob where the nose is so crop is non-empty.
    lms = lying_patient(cx=0.5, cy=0.4)
    nx, ny = int(lms[0].x * 640), int(lms[0].y * 480)
    cv_x0, cv_y0 = max(0, nx - 40), max(0, ny - 40)
    frame[cv_y0:ny + 40, cv_x0:nx + 40] = 200
    result = ff.find(frame, landmarks=lms)
    # May or may not detect Haar; synthetic path should still yield a crop.
    assert result.found is True
    assert result.box is not None
    assert result.crop is not None and result.crop.size > 0


def test_landmark_crops_are_flagged_as_not_a_detection():
    """
    The expression model must be able to tell a detected face from a square
    guessed around the nose.
    """
    ff = FaceFinder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lms = lying_patient(cx=0.5, cy=0.4)
    result = ff.find(frame, landmarks=lms)
    assert result.source == "landmark"
    assert result.is_detection is False


def test_no_face_result_is_still_index_addressable():
    """Older call sites unpack the first element."""
    ff = FaceFinder()
    assert ff.find(None)[0] is False


def test_side_lying_face_is_rotated_upright():
    """Expression models are trained on upright faces."""
    ff = FaceFinder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[180:300, 260:380] = 220
    lms = lying_patient(cx=0.5, cy=0.5)
    # Eyes stacked vertically: head rolled onto its side.
    lms[2] = type(lms[2])(0.50, 0.42, 1.0)
    lms[5] = type(lms[5])(0.50, 0.58, 1.0)
    result = ff.find(frame, landmarks=lms)
    assert result.found is True
    assert abs(result.roll_deg) > 45


def test_cv_distress_score_bounds():
    face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    label, score = cv_distress_score(face)
    assert 0.0 <= score <= 1.0
    assert label in ("tense", "calm")


def test_face_center_requires_visibility():
    lms = lying_patient()
    for i in (0, 2, 5, 7, 8):
        lms[i] = type(lms[i])(lms[i].x, lms[i].y, 0.0)
    assert face_center(lms) is None


def test_pose_estimator_rejects_bad_frames(cfg):
    # Force mediapipe-only if yolo missing — still must not crash.
    cfg.pose_backend = "auto"
    est = PoseEstimator(cfg)
    try:
        r = est.process(None)
        assert isinstance(r, PoseResult)
        assert r.landmarks is None
        bad = np.zeros((100, 100), dtype=np.uint8)  # 2D
        r2 = est.process(bad)
        assert r2.landmarks is None
    finally:
        est.close()


def test_coco_zero_keypoints_skipped():
    xy = [(0, 0)] * 17
    conf = [0.9] * 17
    lms = landmarks_from_coco(xy, conf, 100, 100)
    assert all(lm.visibility == 0.0 for lm in lms)


def test_healthcheck_structure():
    cfg = Thresholds(pose_backend="auto")
    # warm_pose=False keeps this light if models aren't downloaded
    report = run_healthcheck(cfg, warm_pose=False)
    assert isinstance(report.checks, dict)
    assert "opencv" in report.checks
