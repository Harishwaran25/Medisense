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
    cam_source: int = field(default_factory=lambda: _env_int("MEDISENSE_CAM_SOURCE", 0))
    headless: bool = field(default_factory=lambda: _env_bool("MEDISENSE_HEADLESS", False))

    pose_backend: str = field(
        default_factory=lambda: os.environ.get("MEDISENSE_POSE_BACKEND", "auto").strip().lower()
    )
    yolo_pose_model: str = field(
        default_factory=lambda: os.environ.get(
            "MEDISENSE_YOLO_MODEL",
            str(Path(__file__).resolve().parent / "models" / "yolov8n-pose.pt"),
        ).strip()
    )
    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    yolo_kpt_conf: float = 0.25
    yolo_failover_errors: int = 8

    agitation_window_sec: float = 10.0
    agitation_high: float = 0.040
    agitation_mild: float = 0.015

    unconscious_sec: float = 20.0
    movement_threshold: float = 0.004
    face_move_threshold: float = 0.015

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

    emotion_every_n_frames: int = 30
    pain_score_threshold: float = 0.50
    critical_pain_threshold: float = 0.70
    pain_hold_seconds: float = 6.0

    smooth_window: int = 8

    stage1_sec: float = 3.0
    stage2_sec: float = 8.0
    alert_cooldown_sec: float = 30.0

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
        return problems


BODY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24]

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
