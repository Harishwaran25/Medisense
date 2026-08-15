import pytest

from medisense.config import Thresholds
from medisense.vision.capture import CAMERA, FILE, STREAM, open_capture, resolve_source


def test_digit_source_is_camera_index():
    assert resolve_source("0") == (0, CAMERA)
    assert resolve_source("2") == (2, CAMERA)
    assert resolve_source(1) == (1, CAMERA)


def test_blank_source_defaults_to_first_camera():
    assert resolve_source("") == (0, CAMERA)
    assert resolve_source("   ") == (0, CAMERA)


def test_url_source_is_stream():
    assert resolve_source("rtsp://cam.local/stream") == ("rtsp://cam.local/stream", STREAM)
    assert resolve_source("http://cam.local/feed.mjpg")[1] == STREAM


def test_path_source_is_file():
    assert resolve_source("clips/night_roll.mp4") == ("clips/night_roll.mp4", FILE)


def test_missing_file_fails_clearly_instead_of_empty_reads():
    with pytest.raises(FileNotFoundError):
        open_capture("clips/does_not_exist.mp4")


def test_config_accepts_a_file_path_as_source():
    """Validation must not reject a non-numeric source."""
    cfg = Thresholds(cam_source="clips/sleep.mp4")
    assert cfg.validate() == []
    assert resolve_source(cfg.cam_source)[1] == FILE
