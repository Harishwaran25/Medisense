from medisense.detectors.boundary import check_boundary
from conftest import make_landmarks


def test_center_position_is_safe(cfg):
    lms = make_landmarks({11: (0.5, 0.5), 12: (0.5, 0.5), 23: (0.5, 0.5), 24: (0.5, 0.5)})
    status, cx, cy = check_boundary(lms, cfg)
    assert status == "CENTER"


def test_left_edge_is_flagged(cfg):
    lms = make_landmarks({11: (0.02, 0.5), 12: (0.02, 0.5), 23: (0.02, 0.5), 24: (0.02, 0.5)})
    status, cx, cy = check_boundary(lms, cfg)
    assert status == "LEFT EDGE"


def test_right_edge_is_flagged(cfg):
    lms = make_landmarks({11: (0.98, 0.5), 12: (0.98, 0.5), 23: (0.98, 0.5), 24: (0.98, 0.5)})
    status, cx, cy = check_boundary(lms, cfg)
    assert status == "RIGHT EDGE"
