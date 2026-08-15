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

from medisense.config import (
    EVENT_AGITATION,
    EVENT_BOUNDARY,
    EVENT_FALL,
    EVENT_NO_RESPIRATION,
    EVENT_PAIN,
    EVENT_STILLNESS_UNVERIFIED,
)
from medisense.reporting.db import EventLogger

logger = logging.getLogger("medisense.reporting.llm_summary")


def compute_shift_metrics(events: list[dict]) -> dict:
    """Computes summary statistics from raw alert events."""
    total_events = len(events)
    counts_by_type = {}
    counts_by_severity = {}
    distress_scores = []
    modes = {}
    total_duration = 0.0

    for ev in events:
        etype = ev["event_type"]
        sev = ev["severity"]
        counts_by_type[etype] = counts_by_type.get(etype, 0) + 1
        counts_by_severity[sev] = counts_by_severity.get(sev, 0) + 1

        if ev.get("distress_score", 0.0) > 0:
            distress_scores.append(ev["distress_score"])

        mode = ev.get("expression_mode") or "unknown"
        modes[mode] = modes.get(mode, 0) + 1

        total_duration += ev.get("duration_sec", 0.0)

    avg_distress = (sum(distress_scores) / len(distress_scores)) if distress_scores else 0.0

    return {
        "total_events": total_events,
        "counts_by_type": counts_by_type,
        "counts_by_severity": counts_by_severity,
        "avg_distress": round(avg_distress, 2),
        "distress_samples": len(distress_scores),
        "expression_modes": modes,
        "total_alert_duration_sec": round(total_duration, 1),
    }


def _distress_line(metrics: dict) -> str:
    """
    Report the distress average only when scores were actually recorded, and
    name the source. An average of "0.0" is meaningless if nothing measured a
    score, and a heuristic-derived average must not read like a model output.
    """
    if not metrics["distress_samples"]:
        return "   - Average Patient Distress Score: not recorded this shift"
    modes = metrics.get("expression_modes", {})
    heuristic = modes.get("cv_fallback", 0)
    suffix = ""
    if heuristic:
        suffix = f" ({heuristic} of {metrics['total_events']} from the heuristic fallback, not the model)"
    return (
        f"   - Average Patient Distress Score: {metrics['avg_distress']} / 1.0"
        f" across {metrics['distress_samples']} alert(s){suffix}"
    )


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
        _distress_line(metrics),
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

    counts = metrics["counts_by_type"]
    if counts.get(EVENT_FALL, 0):
        lines.append("   - [HIGH RISK]: Fall incident recorded. Perform physical & neurological check.")
    if counts.get(EVENT_NO_RESPIRATION, 0):
        lines.append(
            f"   - [HIGH RISK]: Respiration not detected on {counts[EVENT_NO_RESPIRATION]} occasion(s). "
            "Confirm airway and breathing; verify camera view of the chest."
        )
    if counts.get(EVENT_AGITATION, 0):
        lines.append(
            f"   - [OBSERVATION]: {counts[EVENT_AGITATION]} agitation episode(s). "
            "Review pain management / comfort."
        )
    if counts.get(EVENT_STILLNESS_UNVERIFIED, 0):
        lines.append(
            f"   - [MONITORING GAP]: Respiration was unverifiable during "
            f"{counts[EVENT_STILLNESS_UNVERIFIED]} period(s) of prolonged stillness. "
            "Check camera framing, occlusion by bedding, and room lighting."
        )
    if counts.get(EVENT_BOUNDARY, 0):
        lines.append(
            f"   - [OBSERVATION]: Patient reached the bed edge {counts[EVENT_BOUNDARY]} time(s). "
            "Consider rail position and repositioning schedule."
        )
    if counts.get(EVENT_PAIN, 0):
        lines.append(
            f"   - [OBSERVATION]: Facial distress flagged {counts[EVENT_PAIN]} time(s). "
            "Distress proxy only — assess with a clinical pain scale."
        )
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


def main() -> int:
    """Console entry point: `medisense-report [hours] [db_path]`."""
    import sys

    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    db_path = sys.argv[2] if len(sys.argv) > 2 else "medisense_events.db"
    print(generate_shift_summary(db_path=db_path, shift_hours=hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
