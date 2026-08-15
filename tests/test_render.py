"""
Overlay tests.

Drawing code had no tests, which is how it ended up rendering two skeletons on
top of each other. These do not check that it looks good — they check that it
draws without raising for every state combination, that it does not paint over
the whole frame, and that uncertain readings are labelled rather than shown as
numbers.
"""
import numpy as np

from medisense.config import Thresholds
from medisense.detectors import breathing as resp
from medisense.detectors import pain as pain_states
from medisense.detectors import stillness as still
from medisense.ui.model import MonitorSnapshot
from medisense.ui.render import (
    _expression_lines,
    _movement_text,
    _respiration_text,
    ascii_safe,
    format_duration,
    recent_alerts,
    render,
)
from conftest import lying_patient


def blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_renders_a_normal_frame_without_raising():
    frame = blank()
    render(frame, MonitorSnapshot(landmarks=lying_patient()), Thresholds())
    assert frame.any(), "overlay drew nothing"


def test_renders_every_severity():
    for state, message in (
        ("NORMAL", "Patient stable"),
        ("WARNING", "STILL — RESPIRATION UNVERIFIED"),
        ("CRITICAL", "NO RESPIRATION DETECTED"),
    ):
        frame = blank()
        render(frame, MonitorSnapshot(state=state, message=message), Thresholds())
        assert frame.any()


def test_renders_every_stillness_and_respiration_state():
    cfg = Thresholds()
    for stillness in (
        still.READING, still.ACTIVE, still.STILL,
        still.ASLEEP, still.PROLONGED_STILL, still.NO_RESPIRATION,
    ):
        for respiration in (
            resp.READING, resp.BREATHING, resp.NO_BREATHING, resp.UNVERIFIED, "NOT MEASURED",
        ):
            frame = blank()
            render(
                frame,
                MonitorSnapshot(
                    stillness_state=stillness,
                    stillness_sec=93.0,
                    respiration_state=respiration,
                    respiration_bpm=15.0,
                    respiration_reason="chest region too dark to measure",
                ),
                cfg,
            )
            assert frame.any()


def test_survives_a_tiny_frame():
    """Small frames must not produce negative-size draw calls."""
    frame = blank(h=120, w=200)
    render(frame, MonitorSnapshot(state="CRITICAL", message="FALL DETECTED"), Thresholds())


def test_does_not_paint_over_the_whole_frame():
    """The patient has to remain visible; the old centre banner covered them."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    render(
        frame,
        MonitorSnapshot(state="CRITICAL", message="FALL DETECTED", fallen=True),
        Thresholds(),
    )
    video_area = frame[:, 300:]
    untouched = np.count_nonzero(np.all(video_area == 200, axis=2))
    assert untouched > 0.5 * video_area.shape[0] * video_area.shape[1]


def test_alerts_are_optional_and_bounded():
    frame = blank()
    alerts = [
        {"time": "19:04:0%d" % i, "event": f"EVENT {i}", "severity": "CRITICAL", "detail": ""}
        for i in range(6)
    ]
    render(frame, MonitorSnapshot(alerts=alerts), Thresholds())
    assert frame.any()


def test_newest_alert_is_shown_first():
    """AlertManager.get_log() is append-ordered, so the tail is the newest."""
    log = [
        {"time": stamp, "event": "X", "severity": "CRITICAL", "detail": ""}
        for stamp in ("19:04:01", "19:04:09", "19:05:30")
    ]
    assert [e["time"] for e in recent_alerts(log)] == ["19:05:30", "19:04:09"]
    assert recent_alerts([]) == []


def test_respiration_text_matches_state():
    assert _respiration_text(
        MonitorSnapshot(respiration_state=resp.BREATHING, respiration_bpm=16.4)
    )[0] == "16 BPM"
    assert _respiration_text(MonitorSnapshot(respiration_state=resp.NO_BREATHING))[0] == "NOT DETECTED"
    assert _respiration_text(MonitorSnapshot(respiration_state=resp.UNVERIFIED))[0] == "UNVERIFIED"


def test_unassessed_expression_is_labelled_not_scored():
    text, _, note = _expression_lines(
        MonitorSnapshot(pain_state=pain_states.NOT_ASSESSED, pain_score=0.0)
    )
    assert text == "NOT ASSESSED"
    assert "%" not in text
    assert note


def test_uncertain_expression_shows_no_number():
    text, _, _ = _expression_lines(
        MonitorSnapshot(emotion=pain_states.UNCERTAIN, pain_score=0.61)
    )
    assert text == "UNCERTAIN"
    assert "%" not in text


def test_heuristic_expression_is_labelled_as_such():
    _, _, note = _expression_lines(
        MonitorSnapshot(emotion="tense", pain_score=0.55, expression_mode="cv_fallback")
    )
    assert "heuristic" in note


def test_non_ascii_message_characters_are_substituted():
    """
    Hershey fonts draw anything outside ASCII as "???", and the state messages
    are full of em-dashes and middle dots.
    """
    assert ascii_safe("Asleep \u2014 breathing detected") == "Asleep - breathing detected"
    assert ascii_safe("ASLEEP \u00b7 10m") == "ASLEEP - 10m"
    assert "?" not in ascii_safe("16 BPM \u00b1 2")


def test_state_messages_survive_the_overlay_unmangled():
    from medisense.state import get_overall_state

    for kwargs in (
        {"fallen": True},
        {"still_st": still.NO_RESPIRATION},
        {"still_st": still.ASLEEP},
        {"still_st": still.PROLONGED_STILL},
        {"boundary_st": "LEFT EDGE"},
        {"posture": "SITTING UP"},
        {"agi_st": "HIGH"},
        {"pain_state": "CRITICAL"},
    ):
        args = dict(
            fallen=False, still_st=still.ACTIVE, agi_st="CALM",
            boundary_st="CENTER", posture="LYING", pain_state="NORMAL",
        )
        args.update(kwargs)
        _state, message = get_overall_state(**args)
        assert "?" not in ascii_safe(message), message
        frame = blank()
        render(frame, MonitorSnapshot(state=_state, message=message), Thresholds())


def test_movement_row_does_not_repeat_the_verdict():
    text, _ = _movement_text(
        MonitorSnapshot(stillness_state=still.NO_RESPIRATION, stillness_sec=64)
    )
    assert "RESPIRATION" not in text
    assert "STILL" in text


def test_duration_formatting():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(93) == "1m 33s"
    assert format_duration(3725) == "1h 02m"
