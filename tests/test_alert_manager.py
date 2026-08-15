import time

from medisense.alerting.alert_manager import AlertManager
from medisense.config import Thresholds
from medisense.reporting.db import EventLogger
from conftest import FakeClock


def make_manager(tmp_db=None, **overrides):
    cfg = Thresholds(**overrides)
    clock = FakeClock()
    alert = AlertManager(cfg, clock=clock)
    if tmp_db is not None:
        alert._event_db = EventLogger(db_path=str(tmp_db))
    return alert, clock


def test_alert_stages_and_resolve():
    cfg = Thresholds(stage1_sec=0.2, stage2_sec=0.4, alert_cooldown_sec=60.0)
    alert = AlertManager(cfg)
    try:
        alert.trigger("CRITICAL", "TEST EVENT", "unit test")
        time.sleep(0.55)
        # Allow monitor loop to advance stages / fire
        time.sleep(0.6)
        log = alert.get_log()
        assert any(e["event"] == "TEST EVENT" for e in log)
        alert.resolve("TEST EVENT")
        with alert._lock:
            assert "TEST EVENT" not in alert._active
    finally:
        alert.close()


def test_stages_advance_on_the_injected_clock():
    """Replaying a file must confirm over video seconds, not wall seconds."""
    alert, clock = make_manager(stage1_sec=3.0, stage2_sec=8.0)
    try:
        alert.trigger("CRITICAL", "FALL DETECTED", "fell")
        alert.tick()
        with alert._lock:
            assert alert._active["FALL DETECTED"]["stage"] == 0

        clock.advance(3.5)
        alert.tick()
        with alert._lock:
            assert alert._active["FALL DETECTED"]["stage"] == 1

        clock.advance(5.0)
        alert.tick()
        for _ in range(50):
            if alert.get_log():
                break
            time.sleep(0.02)
        assert any(e["event"] == "FALL DETECTED" for e in alert.get_log())
    finally:
        alert.close()


def test_single_frame_dropout_does_not_restart_confirmation():
    """
    An intermittently occluded but genuine event must still reach stage 2.
    Deleting the pending event on any dropout made that impossible.
    """
    alert, clock = make_manager(stage1_sec=3.0, stage2_sec=8.0)
    try:
        alert.trigger("CRITICAL", "FALL DETECTED", "fell")
        for _ in range(20):
            clock.advance(0.4)
            alert.trigger("CRITICAL", "FALL DETECTED", "fell")
            clock.advance(0.04)          # one frame where pose is lost
            alert.resolve("FALL DETECTED")
            alert.tick()

        with alert._lock:
            assert "FALL DETECTED" in alert._active
            assert alert._active["FALL DETECTED"]["stage"] >= 1
    finally:
        alert.close()


def test_sustained_absence_still_clears_the_event():
    alert, clock = make_manager(stage1_sec=3.0, stage2_sec=8.0)
    try:
        alert.trigger("WARNING", "POSTURE CHANGE", "sitting up")
        clock.advance(2.0)
        for _ in range(10):
            clock.advance(1.0)
            alert.resolve("POSTURE CHANGE")
        with alert._lock:
            assert "POSTURE CHANGE" not in alert._active
    finally:
        alert.close()


def test_trigger_refreshes_detail_without_restarting_the_clock():
    alert, clock = make_manager(stage1_sec=3.0, stage2_sec=8.0)
    try:
        alert.trigger("CRITICAL", "NO RESPIRATION", "still 20s")
        with alert._lock:
            since = alert._active["NO RESPIRATION"]["since"]

        clock.advance(4.0)
        alert.trigger("CRITICAL", "NO RESPIRATION", "still 24s")
        with alert._lock:
            info = alert._active["NO RESPIRATION"]
            assert info["since"] == since
            assert info["detail"] == "still 24s"
    finally:
        alert.close()


def test_fired_events_record_pipeline_context(tmp_path):
    """
    Without this the stored distress score is always 0.0 and the shift report
    averages nothing.
    """
    db_path = tmp_path / "events.db"
    alert, clock = make_manager(tmp_db=db_path, stage1_sec=1.0, stage2_sec=2.0)
    try:
        alert.set_context(
            pose_backend="yolo", distress_score=0.62, expression_mode="cv_fallback"
        )
        alert.trigger("CRITICAL", "HIGH AGITATION", "restless")
        clock.advance(1.5)
        alert.tick()
        clock.advance(1.5)
        alert.tick()

        events = []
        for _ in range(100):
            events = EventLogger(db_path=str(db_path)).get_events(hours=1.0)
            if events:
                break
            time.sleep(0.02)

        assert events, "expected the fired alert to reach the database"
        row = events[-1]
        assert row["pose_backend"] == "yolo"
        assert abs(row["distress_score"] - 0.62) < 1e-6
        assert row["expression_mode"] == "cv_fallback"
    finally:
        alert.close()
