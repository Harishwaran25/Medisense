"""
SQLite Event Logger for clinical audit trails and shift summary reports.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("medisense.reporting.db")


@dataclass
class AlertEventRecord:
    event_type: str
    severity: str
    detail: str
    pose_backend: str
    distress_score: float
    duration_sec: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventLogger:
    """Thread-safe SQLite event logger."""

    def __init__(self, db_path: str = "medisense_events.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detail TEXT,
                    pose_backend TEXT,
                    distress_score REAL,
                    duration_sec REAL
                )
            """)
            conn.commit()

    def log_event(
        self,
        event_type: str,
        severity: str,
        detail: str = "",
        pose_backend: str = "unknown",
        distress_score: float = 0.0,
        duration_sec: float = 0.0,
    ):
        ts = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO alert_events
                    (timestamp, event_type, severity, detail, pose_backend, distress_score, duration_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (ts, event_type, severity, detail, pose_backend, distress_score, duration_sec),
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to log alert event to SQLite: %s", e)

    def get_events(self, hours: float = 8.0) -> list[dict]:
        cutoff = time.time() - (hours * 3600)
        events = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp, event_type, severity, detail, pose_backend, distress_score, duration_sec
                    FROM alert_events
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """,
                    (cutoff,),
                )
                for row in cursor.fetchall():
                    events.append(dict(row))
        except Exception as e:
            logger.error("Failed to retrieve events from SQLite: %s", e)
        return events
