from medisense.detectors.fall import FallDetector
from conftest import lying_patient, make_landmarks


def test_lying_patient_is_not_fallen(cfg):
    """Bedridden resting pose must NOT trigger fall."""
    det = FallDetector(cfg)
    lms = lying_patient(cx=0.5, cy=0.45)
    fallen = True
    for _ in range(20):
        fallen, _ = det.update(lms)
    assert fallen is False


def test_torso_drop_toward_floor_is_fallen(cfg):
    det = FallDetector(cfg)
    # Calibrate on lying pose
    for _ in range(10):
        det.update(lying_patient(cx=0.5, cy=0.40))
    # Sudden drop (patient slid/fell toward bottom of frame)
    fallen = False
    for _ in range(15):
        fallen, _ = det.update(lying_patient(cx=0.5, cy=0.70))
    assert fallen is True


def test_missing_landmarks_does_not_crash(cfg):
    det = FallDetector(cfg)
    fallen, ratio = det.update([])
    assert fallen is False
    assert ratio == 1.0


def test_low_visibility_torso_is_safe(cfg):
    det = FallDetector(cfg)
    lms = make_landmarks({
        11: (0.4, 0.5, 0.1), 12: (0.6, 0.5, 0.1),
        23: (0.4, 0.52, 0.1), 24: (0.6, 0.52, 0.1),
    })
    for _ in range(10):
        fallen, _ = det.update(lms)
    assert fallen is False
