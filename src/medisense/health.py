"""
Startup / runtime health checks for production readiness.

Call `run_healthcheck()` before opening the camera so missing deps,
broken cascade files, or an unavailable pose backend fail fast with a
clear message instead of crashing mid-loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("medisense.health")


@dataclass
class HealthReport:
    ok: bool
    pose_backend: str = "none"
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_fatal(self):
        if not self.ok:
            msg = "; ".join(self.errors) or "MediSense health check failed"
            raise RuntimeError(msg)


def run_healthcheck(cfg, warm_pose: bool = True) -> HealthReport:
    """
    Verify critical dependencies. Pose backend is required.
    Emotion model is optional (falls back to CV heuristic).
    """
    report = HealthReport(ok=True)

    # OpenCV
    try:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            report.errors.append(f"Haar cascade failed to load: {cascade_path}")
            report.ok = False
            report.checks["opencv"] = False
        else:
            report.checks["opencv"] = True
    except Exception as e:
        report.errors.append(f"OpenCV unavailable: {e}")
        report.ok = False
        report.checks["opencv"] = False

    # NumPy smoke
    try:
        _ = np.zeros((8, 8, 3), dtype=np.uint8)
        report.checks["numpy"] = True
    except Exception as e:
        report.errors.append(f"NumPy unavailable: {e}")
        report.ok = False
        report.checks["numpy"] = False

    # YOLO optional probe
    yolo_ok = False
    try:
        from ultralytics import YOLO  # noqa: F401

        yolo_ok = True
        report.checks["ultralytics"] = True
    except Exception as e:
        report.checks["ultralytics"] = False
        report.warnings.append(f"Ultralytics/YOLO not importable: {e}")

    # MediaPipe optional probe
    mp_ok = False
    try:
        import mediapipe as mp  # noqa: F401

        mp_ok = True
        report.checks["mediapipe"] = True
    except Exception as e:
        report.checks["mediapipe"] = False
        report.warnings.append(f"MediaPipe not importable: {e}")

    prefer = (cfg.pose_backend or "auto").lower()
    if prefer == "yolo" and not yolo_ok:
        report.errors.append(
            "MEDISENSE_POSE_BACKEND=yolo but ultralytics is not installed. "
            "Run: pip install ultralytics"
        )
        report.ok = False
    elif prefer == "mediapipe" and not mp_ok:
        report.errors.append(
            "MEDISENSE_POSE_BACKEND=mediapipe but mediapipe is not installed."
        )
        report.ok = False
    elif prefer == "auto" and not yolo_ok and not mp_ok:
        report.errors.append(
            "No pose backend available. Install ultralytics and/or mediapipe."
        )
        report.ok = False

    if not report.ok:
        _log_report(report)
        return report

    if warm_pose:
        try:
            from medisense.vision.pose import PoseEstimator

            est = PoseEstimator(cfg)
            report.pose_backend = est.backend_name
            report.checks["pose_backend"] = est.backend_name != "none"
            if est.backend_name == "none":
                report.errors.append(
                    "PoseEstimator loaded but no backend is active "
                    "(check YOLO weights download / MediaPipe install)."
                )
                report.ok = False
            else:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                result = est.process(dummy)
                if result.error and result.backend == "none":
                    report.errors.append(f"Pose warm-up inference error: {result.error}")
                    report.ok = False
                else:
                    logger.info("Pose warm-up OK (backend=%s).", est.backend_name)
            est.close()
        except Exception as e:
            report.errors.append(f"Pose warm-up failed: {e}")
            report.ok = False
            report.checks["pose_backend"] = False
    else:
        report.warnings.append("Pose warm-up skipped; runtime backend not verified yet.")
        report.pose_backend = "unverified"
        report.checks["pose_import"] = bool(yolo_ok or mp_ok)

    # Emotion stack (non-fatal)
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        report.checks["emotion_stack"] = True
    except Exception as e:
        report.checks["emotion_stack"] = False
        report.warnings.append(
            f"Transformers/Torch unavailable — CV expression fallback only: {e}"
        )

    _log_report(report)
    return report


def _log_report(report: HealthReport) -> None:
    for w in report.warnings:
        logger.warning(w)
    for e in report.errors:
        logger.error(e)
    if report.ok:
        logger.info(
            "Health check passed. pose=%s checks=%s",
            report.pose_backend,
            {k: v for k, v in report.checks.items()},
        )
