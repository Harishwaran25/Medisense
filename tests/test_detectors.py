from medisense.detectors.agitation import AgitationDetector
from medisense.detectors.stillness import StillnessDetector
from medisense.detectors.posture import PostureDetector
from medisense.state import get_overall_state
from medisense.vision.landmarks import Landmark, landmarks_from_coco, snapshot_landmarks
from conftest import lying_patient, make_landmarks


def test_snapshot_landmarks_breaks_aliasing():
    shared = [Landmark(0.1, 0.2, 1.0)]
    snap = snapshot_landmarks(shared)
    # Mutating original list entry identity must not matter — values copied.
    assert snap[0].x == 0.1
    assert snap[0] is not shared[0]


def test_coco_to_mediapipe_mapping():
    # 17 COCO keypoints in pixels on a 100x100 frame
    xy = [(50, 10)] + [(0, 0)] * 4 + [(40, 40), (60, 40)] + [(0, 0)] * 4 + [(40, 60), (60, 60)] + [(0, 0)] * 4
    conf = [0.9] * 17
    lms = landmarks_from_coco(xy, conf, 100, 100, conf_threshold=0.25)
    assert abs(lms[0].x - 0.5) < 1e-6
    assert abs(lms[11].x - 0.4) < 1e-6
    assert abs(lms[23].y - 0.6) < 1e-6


def test_agitation_detects_movement(cfg):
    det = AgitationDetector(cfg)
    base = lying_patient(cx=0.5, cy=0.45)
    for _ in range(3):
        det.update(base)
    # Oscillate torso
    for i in range(12):
        shift = 0.05 if i % 2 == 0 else -0.05
        det.update(lying_patient(cx=0.5 + shift, cy=0.45 + shift))
    avg, label = det.update(lying_patient(cx=0.55, cy=0.50))
    assert avg > 0
    assert label in ("MILD", "HIGH", "CALM", "READING")


def test_stillness_warmup_returns_reading(cfg):
    det = StillnessDetector(cfg, warmup_frames=5)
    lms = lying_patient()
    for _ in range(5):
        dur, st = det.update(lms)
        assert st == "READING"
        assert dur == 0.0


def test_posture_sitting_up_after_calibration(cfg):
    det = PostureDetector(cfg)
    # Calibrate lying (nose relatively low / same band)
    for _ in range(8):
        det.update(lying_patient(cx=0.5, cy=0.50))
    # Nose rises significantly (patient sits up toward camera top → smaller y)
    posture = "LYING"
    for _ in range(20):
        lms = lying_patient(cx=0.5, cy=0.50)
        # Raise nose
        lms[0] = type(lms[0])(0.5, 0.20, 1.0)
        posture, _ = det.update(lms)
    assert posture in ("SITTING UP", "ROLLING", "LYING", "CALIBRATING")


def test_overall_state_priority():
    state, msg = get_overall_state(True, "HIGH", "SITTING UP", "UNCONSCIOUS", "LEFT EDGE", "CRITICAL")
    assert state == "CRITICAL"
    assert "FALL" in msg

    state, msg = get_overall_state(False, "CALM", "LYING", "ACTIVE", "CENTER", "NORMAL")
    assert state == "NORMAL"
