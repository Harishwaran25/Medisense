"""
MediSense configuration.

Secrets from environment / .env. Thresholds in one dataclass.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("medisense.config")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r — using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r — using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_source(name: str, default: str) -> str:
    """Frame source as text: camera index, file path, or stream URL."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _default_pose_weights() -> str:
    """
    Prefer weights bundled next to the package, otherwise use the bare asset
    name so Ultralytics downloads it into the working directory.

    The default used to be the bundled path unconditionally, which pointed at
    an empty directory inside the installed package and disagreed with the
    documented value. On a system-wide install that path is not writable, so
    the download had nowhere to go.
    """
    override = os.environ.get("MEDISENSE_YOLO_MODEL", "").strip()
    if override:
        return override
    bundled = Path(__file__).resolve().parent / "models" / "yolov8n-pose.pt"
    return str(bundled) if bundled.is_file() else "yolov8n-pose.pt"


@dataclass
class Secrets:
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("MEDISENSE_TELEGRAM_BOT_TOKEN", "").strip()
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("MEDISENSE_TELEGRAM_CHAT_ID", "").strip()
    )
    relative_phone_number: str = field(
        default_factory=lambda: os.environ.get("MEDISENSE_RELATIVE_PHONE", "").strip()
    )

    def missing(self) -> list:
        required = {
            "MEDISENSE_TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
            "MEDISENSE_TELEGRAM_CHAT_ID": self.telegram_chat_id,
        }
        return [name for name, val in required.items() if not val]


@dataclass
class Thresholds:
    # Webcam index ("0"), video file path, or stream URL. Files replay against
    # a video clock so second-based thresholds stay comparable to live runs.
    cam_source: str = field(default_factory=lambda: _env_source("MEDISENSE_CAM_SOURCE", "0"))
    loop_video: bool = field(default_factory=lambda: _env_bool("MEDISENSE_LOOP_VIDEO", False))
    headless: bool = field(default_factory=lambda: _env_bool("MEDISENSE_HEADLESS", False))

    pose_backend: str = field(
        default_factory=lambda: os.environ.get("MEDISENSE_POSE_BACKEND", "auto").strip().lower()
    )
    yolo_pose_model: str = field(default_factory=lambda: _default_pose_weights())
    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    yolo_kpt_conf: float = 0.25
    yolo_failover_errors: int = 8

    agitation_window_sec: float = 10.0
    agitation_high: float = 0.040
    agitation_mild: float = 0.015

    # Stillness thresholds. A sleeping patient is motionless for hours, so
    # stillness alone is never critical — see detectors/stillness.py.
    still_sec: float = 5.0
    asleep_sec: float = 45.0
    prolonged_stillness_sec: float = 120.0
    movement_threshold: float = 0.004
    face_move_threshold: float = 0.015

    # Respiration is what separates sleep from unconsciousness.
    respiration_window_sec: float = 20.0
    respiration_min_bpm: float = 8.0
    respiration_max_bpm: float = 34.0
    respiration_conf_threshold: float = 0.35
    respiration_min_amplitude: float = 0.05
    respiration_max_amplitude: float = 6.0
    respiration_min_luma: float = 18.0
    respiration_duplicate_frames: int = 15
    respiration_duplicate_fraction: float = 0.9
    no_respiration_sec: float = 20.0

    sitting_up_threshold: float = 0.15
    rolling_threshold: float = 0.13
    posture_calibration_frames: int = 30

    bed_margin: float = 0.12

    fall_calibration_frames: int = 45
    fall_drop_threshold: float = 0.18
    fall_lateral_threshold: float = 0.22
    fall_speed_threshold: float = 0.08
    fall_ratio_threshold: float = 0.07
    fall_horizontal_threshold: float = 0.15
    # Baseline maintenance: reject a calibration window the patient moved
    # through, then let the accepted baseline follow slow repositioning.
    fall_calibration_max_std: float = 0.04
    fall_baseline_drift: float = 0.02
    fall_drift_stable_frames: int = 45
    fall_drift_max_spread: float = 0.01

    emotion_every_n_frames: int = 30
    pain_score_threshold: float = 0.50
    critical_pain_threshold: float = 0.70
    pain_hold_seconds: float = 6.0
    # Expression quality gates. A classifier returns a confident-looking label
    # for any image, so anything that is not clearly a face must be rejected
    # before it is scored rather than after. Tunable because how readable a
    # face is depends entirely on camera placement and ward lighting.
    emotion_min_confidence: float = field(
        default_factory=lambda: _env_float("MEDISENSE_EMOTION_MIN_CONFIDENCE", 0.45)
    )
    emotion_min_face_px: int = field(
        default_factory=lambda: _env_int("MEDISENSE_EMOTION_MIN_FACE_PX", 64)
    )
    emotion_min_sharpness: float = field(
        default_factory=lambda: _env_float("MEDISENSE_EMOTION_MIN_SHARPNESS", 8.0)
    )
    pain_stale_sec: float = field(
        default_factory=lambda: _env_float("MEDISENSE_PAIN_STALE_SEC", 10.0)
    )

    smooth_window: int = 8

    stage1_sec: float = 3.0
    stage2_sec: float = 8.0
    alert_cooldown_sec: float = 30.0
    # Fraction of a detection gap that is deducted from confirmation progress.
    # 1.0 restores the old behaviour of restarting the clock on any dropout.
    alert_decay_factor: float = 2.0

    pose_miss_clear_frames: int = 15
    camera_fail_alert_frames: int = 30
    camera_fail_exit_frames: int = 150

    def validate(self) -> list[str]:
        """Return human-readable config problems (empty if OK)."""
        problems = []
        if self.pose_backend not in ("auto", "yolo", "mediapipe"):
            problems.append(f"pose_backend must be auto|yolo|mediapipe, got {self.pose_backend!r}")
        if not (0.0 < self.bed_margin < 0.45):
            problems.append("bed_margin must be between 0 and 0.45")
        if self.stage1_sec <= 0 or self.stage2_sec < self.stage1_sec:
            problems.append("need 0 < stage1_sec <= stage2_sec")
        if self.agitation_mild > self.agitation_high:
            problems.append("agitation_mild must be <= agitation_high")
        if self.pain_score_threshold > self.critical_pain_threshold:
            problems.append("pain_score_threshold must be <= critical_pain_threshold")
        if self.emotion_every_n_frames < 1:
            problems.append("emotion_every_n_frames must be >= 1")
        if self.yolo_failover_errors < 1:
            problems.append("yolo_failover_errors must be >= 1")
        if not (0 < self.still_sec <= self.asleep_sec <= self.prolonged_stillness_sec):
            problems.append(
                "need 0 < still_sec <= asleep_sec <= prolonged_stillness_sec"
            )
        if self.respiration_min_bpm >= self.respiration_max_bpm:
            problems.append("respiration_min_bpm must be < respiration_max_bpm")
        if self.respiration_window_sec < 120.0 / self.respiration_min_bpm:
            problems.append(
                "respiration_window_sec is too short to resolve respiration_min_bpm "
                f"(need >= {120.0 / self.respiration_min_bpm:.0f}s)"
            )
        if self.no_respiration_sec < self.respiration_window_sec:
            problems.append(
                "no_respiration_sec must be >= respiration_window_sec, otherwise "
                "escalation can precede a full measurement window"
            )
        if not (0.0 < self.fall_baseline_drift < 1.0):
            problems.append("fall_baseline_drift must be between 0 and 1")
        if self.alert_decay_factor <= 0:
            problems.append("alert_decay_factor must be > 0")
        if not (0.0 <= self.emotion_min_confidence < 1.0):
            problems.append("emotion_min_confidence must be in [0, 1)")
        if self.emotion_min_face_px < 24:
            problems.append("emotion_min_face_px must be >= 24 to be classifiable")
        if self.emotion_min_sharpness < 0:
            problems.append("emotion_min_sharpness must be >= 0")
        if self.pain_stale_sec <= 0:
            problems.append("pain_stale_sec must be > 0")
        return problems


BODY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24]

# Canonical alert event names. Anything that reads events back out of the
# database (shift reports, dashboards) must match on these constants rather
# than re-typing the strings: the shift report previously looked for "FALLEN"
# and "AGITATION" while the alert manager wrote "FALL DETECTED" and "HIGH
# AGITATION", so its recommendations section could never trigger.
EVENT_FALL = "FALL DETECTED"
EVENT_NO_RESPIRATION = "NO RESPIRATION"
EVENT_AGITATION = "HIGH AGITATION"
EVENT_BOUNDARY = "BED BOUNDARY"
EVENT_POSTURE = "POSTURE CHANGE"
EVENT_PAIN = "PAIN DETECTED"
EVENT_STILLNESS_UNVERIFIED = "STILLNESS UNVERIFIED"
EVENT_POSE_LOST = "POSE LOST"
EVENT_CAMERA_OFFLINE = "CAMERA OFFLINE"

PATIENT_EVENTS = (
    EVENT_FALL,
    EVENT_NO_RESPIRATION,
    EVENT_AGITATION,
    EVENT_BOUNDARY,
    EVENT_POSTURE,
    EVENT_PAIN,
    EVENT_STILLNESS_UNVERIFIED,
)

PAIN_WEIGHTS = {
    "sad": 0.7,
    "fear": 0.8,
    "disgust": 0.9,
    "angry": 0.6,
    "surprise": 0.3,
    "happy": 0.0,
    "neutral": 0.0,
}

secrets = Secrets()


def setup_logging(log_file: str = "medisense.log", level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
