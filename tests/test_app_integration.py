"""
End-to-end run of the main loop against a generated video file.

Exercises the wiring rather than the models: capture, video clock, detectors,
state fusion, alert confirmation and clean shutdown. Pose is stubbed so the
test needs no downloaded weights, and MediaPipe is deliberately not required —
a YOLO-only install must be able to start, which a module-level MediaPipe
import in the renderer previously prevented.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from medisense import app as app_module
from medisense.config import (
    EVENT_NO_RESPIRATION,
    EVENT_STILLNESS_UNVERIFIED,
    Thresholds,
)
from medisense.health import HealthReport
from medisense.vision.landmarks import Landmark, NUM_LANDMARKS
from medisense.vision.pose import PoseResult

FPS = 6.0
# Long enough to clear the respiration window, the stillness delay and both
# alert confirmation stages.
SECONDS = 80
SIZE = (240, 320)

# Chest oscillation has to survive video compression, so it is larger than a
# real one while staying under the gross-motion ceiling.
CHEST_AMPLITUDE = 5.0


def sleeping_landmarks():
    lms = [Landmark(0.5, 0.5, 0.9) for _ in range(NUM_LANDMARKS)]
    lms[11] = Landmark(0.39, 0.40, 0.95)
    lms[12] = Landmark(0.61, 0.40, 0.95)
    lms[23] = Landmark(0.39, 0.70, 0.95)
    lms[24] = Landmark(0.61, 0.70, 0.95)
    return lms


class StubPose:
    """Emits a motionless, fully visible patient every frame."""

    backend_name = "stub"
    ready = True

    def __init__(self, cfg):
        self.cfg = cfg
        self._lms = sleeping_landmarks()

    def process(self, frame):
        return PoseResult(
            landmarks=self._lms,
            backend="stub",
            person_bbox=(60, 60, 260, 200),
            confidence=0.9,
            raw_landmarks=self._lms,
        )

    def close(self):
        return None


def write_clip(path, breathing: bool):
    """A still 'patient' whose chest region may or may not oscillate."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (SIZE[1], SIZE[0]))
    assert writer.isOpened(), "MJPG writer unavailable"
    rng = np.random.default_rng(4)
    try:
        for i in range(int(FPS * SECONDS)):
            t = i / FPS
            frame = np.full((SIZE[0], SIZE[1], 3), 130.0)
            frame += rng.normal(0.0, 1.2, frame.shape)
            if breathing:
                chest = CHEST_AMPLITUDE * np.sin(2 * np.pi * 0.25 * t)
                frame[int(0.42 * SIZE[0]):int(0.62 * SIZE[0]),
                      int(0.42 * SIZE[1]):int(0.60 * SIZE[1])] += chest
            writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.release()


@pytest.fixture(scope="session")
def breathing_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clips") / "sleeping.avi"
    write_clip(path, breathing=True)
    return path


@pytest.fixture(scope="session")
def apnoea_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clips") / "apnoea.avi"
    write_clip(path, breathing=False)
    return path


@pytest.fixture
def stubbed_app(monkeypatch, tmp_path):
    """Run the loop in a temp cwd with pose and model loading stubbed out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "PoseEstimator", StubPose)
    monkeypatch.setattr(
        app_module, "run_healthcheck",
        lambda cfg, warm_pose=True: HealthReport(ok=True, pose_backend="stub"),
    )
    # Keep the expression model off the network; the heuristic path is enough.
    monkeypatch.setattr(
        "medisense.detectors.pain.PainDetector._load_model",
        lambda self: setattr(self, "_ready", True) or setattr(self, "_use_cv", True),
    )
    monkeypatch.setenv("MEDISENSE_HEADLESS", "1")
    monkeypatch.setenv("MEDISENSE_POSE_BACKEND", "auto")
    return tmp_path


def run_against(clip, monkeypatch):
    monkeypatch.setenv("MEDISENSE_CAM_SOURCE", str(clip))
    fired = []
    original = app_module.AlertManager._fire_alert

    def record(self, severity, event, detail, duration):
        fired.append(event)
        return None

    monkeypatch.setattr(app_module.AlertManager, "_fire_alert", record)
    try:
        code = app_module.main()
    finally:
        monkeypatch.setattr(app_module.AlertManager, "_fire_alert", original)
    return code, fired


def test_sleeping_patient_produces_no_alerts_over_a_long_clip(
    stubbed_app, monkeypatch, breathing_clip
):
    """The overnight false-alarm case, end to end."""
    code, fired = run_against(breathing_clip, monkeypatch)
    assert code == 0
    assert fired == [], f"a breathing, sleeping patient should not alert, got {fired}"


def test_motionless_patient_without_chest_movement_does_alert(
    stubbed_app, monkeypatch, apnoea_clip
):
    code, fired = run_against(apnoea_clip, monkeypatch)
    assert code == 0
    assert EVENT_NO_RESPIRATION in fired


def test_events_are_written_to_the_audit_database(stubbed_app, monkeypatch, apnoea_clip):
    from medisense.reporting.db import EventLogger

    monkeypatch.setenv("MEDISENSE_CAM_SOURCE", str(apnoea_clip))
    assert app_module.main() == 0

    events = EventLogger(db_path=str(stubbed_app / "medisense_events.db")).get_events(hours=1.0)
    assert events
    assert {e["event_type"] for e in events} & {EVENT_NO_RESPIRATION, EVENT_STILLNESS_UNVERIFIED}
    assert all(e["pose_backend"] == "stub" for e in events)


def test_missing_source_file_reports_failure(stubbed_app, monkeypatch):
    monkeypatch.setenv("MEDISENSE_CAM_SOURCE", str(stubbed_app / "nope.avi"))
    assert app_module.main() == 1


def test_bad_config_is_rejected_before_opening_the_source(stubbed_app, monkeypatch):
    monkeypatch.setenv("MEDISENSE_POSE_BACKEND", "banana")
    assert Thresholds().validate()
    assert app_module.main() == 2


def test_overlay_path_runs_with_real_detector_output(stubbed_app, monkeypatch, apnoea_clip):
    """
    The snapshot handed to the overlay is only built when not headless, so
    without this the field wiring is never executed by any test. Windowing is
    stubbed; the drawing itself is real.
    """
    monkeypatch.setenv("MEDISENSE_HEADLESS", "0")
    monkeypatch.setenv("MEDISENSE_CAM_SOURCE", str(apnoea_clip))
    monkeypatch.setattr(app_module.AlertManager, "_fire_alert", lambda *a, **k: None)

    drawn = []
    monkeypatch.setattr(app_module.cv2, "imshow", lambda name, frame: drawn.append(frame.shape))
    monkeypatch.setattr(app_module.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(app_module.cv2, "destroyAllWindows", lambda: None)

    assert app_module.main() == 0
    assert drawn, "no frames were rendered"
