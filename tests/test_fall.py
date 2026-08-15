from medisense.config import Thresholds
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


def test_calibration_rejects_a_window_the_patient_moved_through():
    """Baselining a transition biases every later reading for the session."""
    cfg = Thresholds(fall_calibration_frames=10, fall_calibration_max_std=0.02)
    det = FallDetector(cfg)
    for i in range(10):
        det.update(lying_patient(cx=0.5, cy=0.30 + i * 0.04))
    assert det.ready is False

    for _ in range(10):
        det.update(lying_patient(cx=0.5, cy=0.66))
    assert det.ready is True
    assert abs(det.base_cy - 0.67) < 0.02


def test_baseline_absorbs_a_new_resting_position():
    """Repositioning must not permanently offset every later reading."""
    cfg = Thresholds(
        fall_calibration_frames=10, fall_baseline_drift=0.1, fall_drift_stable_frames=10
    )
    det = FallDetector(cfg)
    for _ in range(10):
        det.update(lying_patient(cx=0.5, cy=0.40))
    start = det.base_cy

    # Nurse shifts the patient, who then settles at the new position.
    fallen = False
    for _ in range(120):
        fallen, _ = det.update(lying_patient(cx=0.5, cy=0.46))

    assert fallen is False
    assert det.base_cy > start + 0.03


def test_drift_does_not_absorb_a_slow_slide_off_the_bed():
    """
    A gradual slide has a tiny per-frame speed but keeps moving. If the
    baseline chases it, the fall is never detected at all.
    """
    cfg = Thresholds(fall_calibration_frames=10, fall_baseline_drift=0.1)
    det = FallDetector(cfg)
    for _ in range(10):
        det.update(lying_patient(cx=0.5, cy=0.35))

    fallen = False
    cy = 0.35
    for _ in range(400):
        cy = min(0.85, cy + 0.001)
        fallen, _ = det.update(lying_patient(cx=0.5, cy=cy))
        if fallen:
            break
    assert fallen is True


def test_recalibrate_rearms_the_baseline(cfg):
    det = FallDetector(cfg)
    for _ in range(10):
        det.update(lying_patient(cx=0.5, cy=0.40))
    assert det.ready is True

    det.recalibrate()
    assert det.ready is False
    assert det.base_cy is None

    for _ in range(10):
        fallen, _ = det.update(lying_patient(cx=0.5, cy=0.72))
    assert det.ready is True
    assert fallen is False
