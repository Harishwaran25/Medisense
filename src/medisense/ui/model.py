"""
What the overlay is allowed to know.

The renderer used to take twenty-odd positional arguments, which made it easy
to draw the wrong value in the wrong panel and impossible to add a field
without touching every call site. One snapshot per frame keeps drawing code
honest and lets the panel layout change without rewiring the main loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MonitorSnapshot:
    # Overall verdict
    state: str = "NORMAL"
    message: str = "Patient stable"

    # Vitals and states shown to staff
    respiration_state: str = "NOT MEASURED"
    respiration_bpm: float = 0.0
    respiration_reason: str = ""
    stillness_state: str = "READING"
    stillness_sec: float = 0.0
    posture: str = "READING"
    agitation_state: str = "READING"
    boundary_state: str = "CENTER"
    fallen: bool = False

    # Expression channel, with the provenance needed to read it correctly
    emotion: str = "neutral"
    pain_score: float = 0.0
    pain_state: str = "NORMAL"
    expression_mode: str = "unknown"
    face_source: Optional[str] = None

    # Overlay geometry
    landmarks: Optional[list] = None
    person_bbox: Optional[tuple[int, int, int, int]] = None
    face_box: Optional[tuple[int, int, int, int]] = None
    patient_cx: float = 0.5
    patient_cy: float = 0.5

    # Technical footer
    pose_backend: str = "auto"
    source_label: str = "camera"
    low_light: bool = False
    alerts: list[dict[str, Any]] = field(default_factory=list)
