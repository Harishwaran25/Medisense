"""
Tests for SQLite event logging and the shift handover report.

These deliberately use the `EVENT_*` constants rather than literal strings.
The earlier versions asserted on names like "FALLEN" and "AGITATION" that the
alert manager never actually wrote, so they passed while the report's
recommendations section could not fire for any real event.
"""
from medisense.config import (
    EVENT_AGITATION,
    EVENT_FALL,
    EVENT_NO_RESPIRATION,
    EVENT_STILLNESS_UNVERIFIED,
)
from medisense.reporting.db import EventLogger
from medisense.reporting.llm_summary import compute_shift_metrics, generate_shift_summary


def test_sqlite_event_logger_writes_and_reads(tmp_path):
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(EVENT_FALL, "CRITICAL", "Patient fell off bed", "yolo", 0.85, 4.2)
    logger.log_event(EVENT_AGITATION, "WARNING", "Thrashing legs", "yolo", 0.40, 2.1)

    events = logger.get_events(hours=1.0)
    assert len(events) == 2
    assert events[0]["event_type"] == EVENT_FALL
    assert events[0]["severity"] == "CRITICAL"
    assert events[1]["event_type"] == EVENT_AGITATION


def test_expression_mode_is_persisted(tmp_path):
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(
        EVENT_AGITATION, "WARNING", "restless", "yolo", 0.5, 2.0,
        expression_mode="cv_fallback",
    )
    assert logger.get_events(hours=1.0)[0]["expression_mode"] == "cv_fallback"


def test_existing_database_without_the_mode_column_is_migrated(tmp_path):
    import sqlite3

    db_path = str(tmp_path / "old.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                detail TEXT,
                pose_backend TEXT,
                distress_score REAL,
                duration_sec REAL
            )
            """
        )
        conn.commit()

    logger = EventLogger(db_path=db_path)
    logger.log_event(EVENT_FALL, "CRITICAL", "fell", "yolo", 0.1, 1.0)
    assert logger.get_events(hours=1.0)[0]["event_type"] == EVENT_FALL


def test_compute_shift_metrics():
    sample_events = [
        {"event_type": EVENT_FALL, "severity": "CRITICAL", "distress_score": 0.8, "duration_sec": 5.0},
        {"event_type": EVENT_AGITATION, "severity": "WARNING", "distress_score": 0.4, "duration_sec": 3.0},
        {"event_type": EVENT_AGITATION, "severity": "WARNING", "distress_score": 0.6, "duration_sec": 2.0},
    ]

    metrics = compute_shift_metrics(sample_events)
    assert metrics["total_events"] == 3
    assert metrics["counts_by_type"][EVENT_FALL] == 1
    assert metrics["counts_by_type"][EVENT_AGITATION] == 2
    assert metrics["avg_distress"] == 0.6
    assert metrics["distress_samples"] == 3
    assert metrics["total_alert_duration_sec"] == 10.0


def test_generate_shift_summary_fallback(tmp_path):
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(EVENT_AGITATION, "WARNING", "Patient restless", "yolo", 0.50, 6.0)

    report = generate_shift_summary(db_path=db_path, shift_hours=8.0)
    assert "MEDISENSE PATIENT SHIFT HANDOVER REPORT" in report
    assert "Total Safety Alerts Triggered: 1" in report
    assert f"{EVENT_AGITATION}: 1 event(s)" in report


def test_recommendations_fire_for_the_names_the_app_actually_writes(tmp_path):
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(EVENT_FALL, "CRITICAL", "fell", "yolo", 0.0, 9.0)
    logger.log_event(EVENT_NO_RESPIRATION, "CRITICAL", "no chest movement", "yolo", 0.0, 22.0)
    logger.log_event(EVENT_STILLNESS_UNVERIFIED, "WARNING", "chest occluded", "yolo", 0.0, 130.0)

    report = generate_shift_summary(db_path=db_path, shift_hours=8.0)
    assert "Perform physical & neurological check" in report
    assert "Confirm airway and breathing" in report
    assert "MONITORING GAP" in report


def test_distress_average_is_omitted_when_nothing_measured_it(tmp_path):
    """Reporting an average of 0.0 from zero samples is worse than silence."""
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(EVENT_FALL, "CRITICAL", "fell", "yolo", 0.0, 9.0)

    report = generate_shift_summary(db_path=db_path, shift_hours=8.0)
    assert "not recorded this shift" in report
    assert "0.0 / 1.0" not in report


def test_heuristic_derived_scores_are_labelled_in_the_report(tmp_path):
    db_path = str(tmp_path / "events.db")
    logger = EventLogger(db_path=db_path)
    logger.log_event(
        "PAIN DETECTED", "CRITICAL", "tense", "yolo", 0.72, 7.0,
        expression_mode="cv_fallback",
    )

    report = generate_shift_summary(db_path=db_path, shift_hours=8.0)
    assert "heuristic fallback, not the model" in report
