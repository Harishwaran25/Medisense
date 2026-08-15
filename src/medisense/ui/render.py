"""
Clinical overlay.

Deliberately sparse. The previous panel showed eight blocks including a raw
`ratio 0.123` diagnostic, two progress bars, a four-entry alert history and a
full-width red banner across the middle of the video, and it drew two
different skeletons on top of each other because both the pose backend and the
renderer were drawing one. At a glance a nurse could not tell what mattered.

What earns space here is what changes a decision: the overall verdict, the
respiration reading, how long the patient has been still, and three state
chips. Developer diagnostics are in the log, not on the screen. Anything
uncertain is labelled as uncertain rather than shown as a number.

Pure drawing code — no detection logic lives here.
"""
from __future__ import annotations

import datetime
import logging

import cv2

from medisense.config import Thresholds
from medisense.detectors import breathing as resp
from medisense.detectors import pain as pain_states
from medisense.detectors import stillness as still
from medisense.ui.model import MonitorSnapshot

logger = logging.getLogger("medisense.ui.render")

# Drawing must not make MediaPipe a hard requirement. Importing it at module
# scope meant a YOLO-only install could not start the app at all, even though
# the pose stack itself treats MediaPipe as optional.
_FALLBACK_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (0, 11), (0, 12),
]

try:
    import mediapipe as mp

    _POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
except Exception as e:  # pragma: no cover - depends on install
    logger.info("MediaPipe unavailable for skeleton edges — using built-in set: %s", e)
    _POSE_CONNECTIONS = _FALLBACK_EDGES

# Palette (BGR). Muted, low-saturation, one accent.
C_BG = (26, 28, 34)
C_LINE = (48, 52, 62)
C_TEXT = (228, 231, 238)
C_DIM = (128, 136, 152)
C_OK = (120, 190, 130)
C_WARN = (70, 170, 235)
C_CRIT = (72, 72, 226)
C_ACCENT = (188, 172, 120)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 236
PAD = 14

SEVERITY_COLORS = {"NORMAL": C_OK, "WARNING": C_WARN, "CRITICAL": C_CRIT}

# OpenCV's Hershey fonts are ASCII-only and draw anything else as "???".
# Messages reach the overlay from several modules, so they are sanitised at
# draw time rather than by policing every string that produces one.
_SUBSTITUTIONS = {
    "\u00b7": "-", "\u2014": "-", "\u2013": "-", "\u2022": "-",
    "\u2026": "...", "\u00b0": " deg", "\u2265": ">=", "\u2264": "<=",
    "\u00b1": "+/-", "\u2192": "->", "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
}


def _severity_color(state: str):
    return SEVERITY_COLORS.get(state, C_DIM)


def ascii_safe(text) -> str:
    text = str(text)
    for bad, good in _SUBSTITUTIONS.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "replace").decode("ascii")


def _put(frame, text, origin, scale, color, thickness=1):
    cv2.putText(frame, ascii_safe(text), origin, FONT, scale, color, thickness, cv2.LINE_AA)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class _Column:
    """
    A top-down text cursor, so panel items cannot drift out of alignment.

    With `frame=None` it measures instead of drawing, which lets the panel be
    sized to its contents rather than stretched down the whole window.
    """

    def __init__(self, frame, x: int, y: int, width: int):
        self.frame = frame
        self.x = x
        self.y = y
        self.width = width

    @property
    def drawing(self) -> bool:
        return self.frame is not None

    def gap(self, px: int) -> None:
        self.y += px

    def label(self, text: str) -> None:
        if self.drawing:
            _put(self.frame, text.upper(), (self.x, self.y), 0.34, C_DIM)
        self.y += 15

    def value(self, text: str, color=None, scale: float = 0.55) -> None:
        if self.drawing:
            _put(self.frame, text, (self.x, self.y), scale, color or C_TEXT)
        self.y += int(20 * scale / 0.55)

    def note(self, text: str, color=None) -> None:
        if self.drawing:
            _put(self.frame, text, (self.x, self.y), 0.32, color or C_DIM)
        self.y += 13

    def rule(self) -> None:
        self.y += 4
        if self.drawing:
            cv2.line(self.frame, (self.x, self.y), (self.x + self.width, self.y), C_LINE, 1)
        self.y += 14

    def chips(self, items: list[tuple[str, str, tuple]], per_row: int = 2) -> None:
        """Compact state pills, two per row."""
        cw = (self.width - 8) // per_row
        for i, (label, value, color) in enumerate(items):
            col = i % per_row
            if col == 0 and i:
                self.y += 34
            if self.drawing:
                cx = self.x + col * (cw + 8)
                _put(self.frame, label.upper(), (cx, self.y), 0.29, C_DIM)
                _put(self.frame, value[:12], (cx, self.y + 15), 0.4, color)
        self.y += 34


def _panel_background(frame, x, y, w, h, alpha=0.9):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_LINE, 1)


def draw_skeleton(frame, landmarks, min_vis=0.4, color=C_ACCENT):
    """Single thin skeleton. Both the pose backend and the UI used to draw one."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = {}
    for i, lm in enumerate(landmarks):
        if float(getattr(lm, "visibility", 0.0) or 0.0) < min_vis:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        if 0 <= px < w and 0 <= py < h:
            pts[i] = (px, py)

    for a, b in _POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def _respiration_text(snap: MonitorSnapshot) -> tuple[str, tuple, str]:
    state = snap.respiration_state
    if state == resp.BREATHING:
        return f"{snap.respiration_bpm:.0f} BPM", C_OK, ""
    if state == resp.NO_BREATHING:
        return "NOT DETECTED", C_CRIT, ""
    if state == resp.UNVERIFIED:
        return "UNVERIFIED", C_WARN, snap.respiration_reason
    if state == resp.READING:
        return "MEASURING", C_DIM, ""
    return "—", C_DIM, ""


def _movement_text(snap: MonitorSnapshot) -> tuple[str, tuple]:
    """
    Describes movement only. The reason it matters belongs to the verdict —
    printing NO RESPIRATION here as well just said the same thing twice.
    """
    state = snap.stillness_state
    color = {
        still.NO_RESPIRATION: C_CRIT,
        still.PROLONGED_STILL: C_WARN,
        still.ASLEEP: C_OK,
        still.READING: C_DIM,
    }.get(state, C_OK)
    text = {
        still.READING: "measuring",
        still.ACTIVE: "MOVING",
        still.STILL: "STILL",
        still.ASLEEP: "ASLEEP",
        still.PROLONGED_STILL: "STILL",
        still.NO_RESPIRATION: "STILL",
    }.get(state, state)
    if snap.stillness_sec >= 1 and state != still.READING:
        return f"{text}   {format_duration(snap.stillness_sec)}", color
    return text, color


def recent_alerts(alerts: list, limit: int = 2) -> list:
    """Newest first. AlertManager.get_log() is in append order, so oldest first."""
    return list(reversed(alerts))[:limit]


def _expression_lines(snap: MonitorSnapshot) -> tuple[str, tuple, str]:
    """Label and number always describe the same prediction, or say so."""
    if snap.pain_state == pain_states.LOADING:
        return "loading model", C_DIM, ""
    if snap.pain_state == pain_states.NOT_ASSESSED:
        return "NOT ASSESSED", C_DIM, "no clear view of the face"
    if snap.emotion == pain_states.UNCERTAIN:
        return "UNCERTAIN", C_DIM, "model confidence below threshold"

    color = _severity_color(snap.pain_state) if snap.pain_state != "NORMAL" else C_OK
    text = f"{snap.emotion.upper()}  {snap.pain_score:.0%}"
    if snap.expression_mode == "cv_fallback":
        return text, C_WARN, "heuristic fallback — cannot escalate"
    return text, color, ""


def render(frame, snap: MonitorSnapshot, cfg: Thresholds):
    """Draw the overlay onto `frame` in place."""
    h, w = frame.shape[:2]
    accent = _severity_color(snap.state)

    draw_skeleton(frame, snap.landmarks)
    _draw_scene(frame, snap, cfg, accent)

    # Measure first so the panel is as tall as its contents. A sidebar
    # stretched to the window height is mostly empty space, and empty space
    # reads as missing information.
    measured = _Column(None, PAD + 14, PAD + 30, PANEL_W - 28)
    _fill_panel(measured, snap, accent)
    content_h = measured.y - PAD + 46  # footer block
    panel_h = max(150, min(content_h, h - 2 * PAD))

    _panel_background(frame, PAD, PAD, PANEL_W, panel_h)
    _fill_panel(_Column(frame, PAD + 14, PAD + 30, PANEL_W - 28), snap, accent)
    _draw_footer(frame, snap, panel_h)


def _fill_panel(col: _Column, snap: MonitorSnapshot, accent):
    frame = col.frame

    # Verdict
    if col.drawing:
        cv2.circle(frame, (col.x + 5, col.y - 5), 5, accent, -1, cv2.LINE_AA)
        _put(frame, snap.state, (col.x + 18, col.y), 0.62, accent)
    col.y += 18
    col.note(snap.message[:34], C_TEXT)
    col.rule()

    # Vitals
    col.label("respiration")
    value, color, reason = _respiration_text(snap)
    col.value(value, color)
    if reason:
        col.note(reason[:36])
    col.gap(8)

    col.label("movement")
    movement, mcolor = _movement_text(snap)
    col.value(movement, mcolor, scale=0.46)
    col.rule()

    # States that matter but do not need a number
    col.chips([
        ("posture", snap.posture, C_WARN if snap.posture in ("SITTING UP", "ROLLING") else C_TEXT),
        ("agitation", snap.agitation_state,
         C_CRIT if snap.agitation_state == "HIGH"
         else C_WARN if snap.agitation_state == "MILD" else C_TEXT),
        ("position", snap.boundary_state, C_CRIT if snap.boundary_state != "CENTER" else C_TEXT),
    ])
    col.rule()

    col.label("expression")
    etext, ecolor, enote = _expression_lines(snap)
    col.value(etext, ecolor, scale=0.44)
    if enote:
        col.note(enote[:38])

    if snap.alerts:
        col.rule()
        col.label("last alerts")
        for entry in recent_alerts(snap.alerts):
            color = _severity_color(entry.get("severity", ""))
            col.note(f"{entry.get('time', '')}  {str(entry.get('event', ''))[:22]}", color)


def _draw_footer(frame, snap: MonitorSnapshot, panel_h: int):
    y = PAD + panel_h - 14
    bits = [snap.pose_backend.upper(), snap.source_label]
    if snap.low_light:
        bits.append("night vision")
    _put(frame, " | ".join(bits)[:34], (PAD + 14, y - 15), 0.3, C_DIM)
    _put(frame, datetime.datetime.now().strftime("%H:%M:%S"), (PAD + 14, y), 0.42, C_TEXT)


def _draw_scene(frame, snap: MonitorSnapshot, cfg: Thresholds, accent):
    """Bed guides, patient marker, face box and the alert strip."""
    h, w = frame.shape[:2]
    scene_x = PAD + PANEL_W + 10

    # Bed edges: thin, dim, labelled once.
    lx = int(w * cfg.bed_margin)
    rx = int(w * (1.0 - cfg.bed_margin))
    for x in (lx, rx):
        if x > scene_x:
            cv2.line(frame, (x, PAD), (x, h - PAD), C_LINE, 1, cv2.LINE_AA)
    if rx > scene_x:
        _put(frame, "BED EDGE", (rx - 64, h - PAD - 8), 0.3, C_DIM)

    if snap.person_bbox is not None:
        x1, y1, x2, y2 = snap.person_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_LINE, 1, cv2.LINE_AA)

    if snap.face_box is not None:
        fx, fy, fw, fh = snap.face_box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), C_ACCENT, 1, cv2.LINE_AA)

    if 0.01 < snap.patient_cx < 0.99 and 0.01 < snap.patient_cy < 0.99:
        px, py = int(snap.patient_cx * w), int(snap.patient_cy * h)
        marker = C_CRIT if snap.boundary_state != "CENTER" else C_ACCENT
        cv2.circle(frame, (px, py), 7, marker, 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 1, marker, -1, cv2.LINE_AA)

    # A single strip, top of the video area — unmissable without covering the
    # patient the way the old full-width centre banner did.
    if snap.state != "NORMAL":
        y0 = PAD
        y1 = PAD + 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (scene_x, y0), (w - PAD, y1), accent, -1)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
        cv2.rectangle(frame, (scene_x, y0), (w - PAD, y1), accent, 1)
        cv2.rectangle(frame, (scene_x, y0), (scene_x + 4, y1), accent, -1)
        _put(frame, snap.message[:44], (scene_x + 14, y0 + 20), 0.5, C_TEXT)
