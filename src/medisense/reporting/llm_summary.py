"""
Automated LLM-powered Shift Handover Summary Report Generator.
Aggregates SQLite event metrics and uses LLM / structured synthesis to generate
clinical handover reports for medical staff.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from medisense.reporting.db import EventLogger

logger = logging.getLogger("medisense.reporting.llm_summary")


def compute_shift_metrics(events: list[dict]) -> dict:
    """Computes summary statistics from raw alert events."""
    total_events = len(events)
    counts_by_type = {}
    counts_by_severity = {}
    distress_scores = []
    total_duration = 0.0

    for ev in events:
        etype = ev["event_type"]
        sev = ev["severity"]
        counts_by_type[etype] = counts_by_type.get(etype, 0) + 1
        counts_by_severity[sev] = counts_by_severity.get(sev, 0) + 1

        if ev.get("distress_score", 0.0) > 0:
            distress_scores.append(ev["distress_score"])

        total_duration += ev.get("duration_sec", 0.0)

    avg_distress = (sum(distress_scores) / len(distress_scores)) if distress_scores else 0.0

    return {
        "total_events": total_events,
        "counts_by_type": counts_by_type,
        "counts_by_severity": counts_by_severity,
        "avg_distress": round(avg_distress, 2),
        "total_alert_duration_sec": round(total_duration, 1),
    }


def generate_fallback_summary(metrics: dict, shift_hours: float) -> str:
    """Generates a structured medical handoff report without requiring external APIs."""
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines = [
        "================================================================================",
        "                     MEDISENSE PATIENT SHIFT HANDOVER REPORT                   ",
        "================================================================================",
        f"Generated At: {now_str}",
        f"Shift Duration Analyzed: Last {shift_hours} Hours",
        "--------------------------------------------------------------------------------",
        "1. SHIFT EXECUTIVE SUMMARY",
        f"   - Total Safety Alerts Triggered: {metrics['total_events']}",
        f"   - Average Patient Distress Score: {metrics['avg_distress']} / 1.0",
        f"   - Cumulative Warning/Alert Duration: {metrics['total_alert_duration_sec']} seconds",
        "",
        "2. EVENT BREAKDOWN BY TYPE",
    ]

    if metrics["counts_by_type"]:
        for etype, count in metrics["counts_by_type"].items():
            lines.append(f"   - {etype}: {count} event(s)")
    else:
        lines.append("   - No safety alerts triggered during this shift window (Patient Stable).")

    lines.extend([
        "",
        "3. ALERT SEVERITY DISTRIBUTION",
    ])

    if metrics["counts_by_severity"]:
        for sev, count in metrics["counts_by_severity"].items():
            lines.append(f"   - {sev}: {count}")
    else:
        lines.append("   - Normal baseline throughout shift.")

    lines.extend([
        "",
        "4. CARE RECOMMENDATIONS & CLINICAL OBSERVATIONS",
    ])

    fall_count = metrics["counts_by_type"].get("FALLEN", 0)
    agitation_count = metrics["counts_by_type"].get("AGITATION", 0)
    stillness_count = metrics["counts_by_type"].get("STILLNESS", 0)

    if fall_count > 0:
        lines.append("   - [HIGH RISK]: Fall incident recorded. Perform physical & neurological check.")
    if agitation_count > 0:
        lines.append(f"   - [OBSERVATION]: {agitation_count} agitation episode(s). Review pain management / comfort.")
    if stillness_count > 0:
        lines.append(f"   - [OBSERVATION]: Extended stillness detected ({stillness_count} time(s)). Check patient responsiveness.")
    if metrics["total_events"] == 0:
        lines.append("   - Patient rested quietly without safety incidents.")

    lines.append("================================================================================")
    return "\n".join(lines)


def generate_shift_summary(
    db_path: str = "medisense_events.db",
    shift_hours: float = 8.0,
    api_key: Optional[str] = None,
) -> str:
    """
    Fetches events from DB and generates a comprehensive shift handover report.
    Uses Google Gemini API if GEMINI_API_KEY is available, otherwise uses structured synthesis.
    """
    logger.info("Generating shift handover summary for the last %.1f hours...", shift_hours)
    db = EventLogger(db_path)
    events = db.get_events(hours=shift_hours)
    metrics = compute_shift_metrics(events)

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
You are an expert AI clinical assistant for MediSense Patient Safety Monitor.
Synthesize the following shift event log into a professional, concise medical shift handover report for incoming nurses and doctors.

Shift Duration: {shift_hours} hours
Event Summary: {metrics}
Raw Events: {events[:50]}

Structure your response with:
1. Executive Summary
2. Clinical Highlights & Safety Alerts
3. Recommended Nursing Actions for Next Shift
Keep the tone professional, objective, and clear.
"""
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning("Gemini LLM call failed or unavailable (%s) — using fallback synthesis.", e)

    return generate_fallback_summary(metrics, shift_hours)


if __name__ == "__main__":
    import sys

    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    report = generate_shift_summary(shift_hours=hours)
    print(report)
