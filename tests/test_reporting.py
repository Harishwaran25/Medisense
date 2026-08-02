"""Tests for SQLite Event Logging and LLM Shift Handover Summary Generator."""
import os
import tempfile
import time
from medisense.reporting.db import EventLogger
from medisense.reporting.llm_summary import compute_shift_metrics, generate_shift_summary


def test_sqlite_event_logger_writes_and_reads():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        logger = EventLogger(db_path=db_path)
        logger.log_event("FALLEN", "CRITICAL", "Patient fell off bed", "yolo", 0.85, 4.2)
        logger.log_event("AGITATION", "WARNING", "Thrashing legs", "yolo", 0.40, 2.1)

        events = logger.get_events(hours=1.0)
        assert len(events) == 2
        assert events[0]["event_type"] == "FALLEN"
        assert events[0]["severity"] == "CRITICAL"
        assert events[1]["event_type"] == "AGITATION"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_compute_shift_metrics():
    sample_events = [
        {"event_type": "FALLEN", "severity": "CRITICAL", "distress_score": 0.8, "duration_sec": 5.0},
        {"event_type": "AGITATION", "severity": "WARNING", "distress_score": 0.4, "duration_sec": 3.0},
        {"event_type": "AGITATION", "severity": "WARNING", "distress_score": 0.6, "duration_sec": 2.0},
    ]

    metrics = compute_shift_metrics(sample_events)
    assert metrics["total_events"] == 3
    assert metrics["counts_by_type"]["FALLEN"] == 1
    assert metrics["counts_by_type"]["AGITATION"] == 2
    assert metrics["avg_distress"] == 0.6
    assert metrics["total_alert_duration_sec"] == 10.0


def test_generate_shift_summary_fallback():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        logger = EventLogger(db_path=db_path)
        logger.log_event("AGITATION", "WARNING", "Patient restless", "yolo", 0.50, 6.0)

        report = generate_shift_summary(db_path=db_path, shift_hours=8.0)
        assert "MEDISENSE PATIENT SHIFT HANDOVER REPORT" in report
        assert "Total Safety Alerts Triggered: 1" in report
        assert "AGITATION: 1 event(s)" in report
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
