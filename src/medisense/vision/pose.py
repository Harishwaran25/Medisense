"""
Pose estimation backends.

Primary: Ultralytics YOLO-pose (better for lying / non-upright patients).
Fallback: MediaPipe Pose.
Supports mid-session failover if YOLO repeatedly errors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from medisense.config import Thresholds
from medisense.vision.landmarks import landmarks_from_coco, snapshot_landmarks

logger = logging.getLogger("medisense.vision.pose")


@dataclass
class PoseResult:
    landmarks: Optional[list]
    backend: str
    person_bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 0.0
    error: Optional[str] = None
    raw_landmarks: Optional[list] = None


class PoseEstimator:
    """Unified pose front-end used by the main loop."""

    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self.backend_name = "none"
        self._yolo = None
        self._mp_pose = None
        self._mp_ctx = None
        self._yolo_fail_streak = 0
        self._device = "cpu"
        self._load()

    @property
    def ready(self) -> bool:
        return self.backend_name != "none"

    def _load(self):
        prefer = (self.cfg.pose_backend or "auto").lower().strip()
        if prefer not in ("auto", "yolo", "mediapipe"):
            logger.warning("Unknown pose_backend=%r — using auto.", prefer)
            prefer = "auto"

        if prefer in ("auto", "yolo"):
            if self._try_load_yolo():
                # Also preload MediaPipe as hot standby for failover.
                if prefer == "auto":
                    self._try_load_mediapipe(standby=True)
                return
            if prefer == "yolo":
                logger.error("YOLO requested but failed — trying MediaPipe.")

        if prefer in ("auto", "mediapipe", "yolo"):
            if self._try_load_mediapipe(standby=False):
                return
        logger.critical("No pose backend available.")

    def _try_load_yolo(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.cfg.yolo_pose_model
            logger.info("Loading YOLO pose model: %s (device=%s)", model_name, self._device)
            self._yolo = YOLO(model_name)
            # Force a tiny predict so weight download / device errors surface here.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._yolo.predict(dummy, verbose=False, device=self._device)
            self.backend_name = "yolo"
            logger.info("YOLO pose backend ready.")
            return True
        except Exception as e:
            logger.warning("YOLO pose unavailable: %s", e)
            self._yolo = None
            return False

    def _try_load_mediapipe(self, standby: bool = False) -> bool:
        if self._mp_ctx is not None:
            if not standby and self.backend_name == "none":
                self.backend_name = "mediapipe"
            return True
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose
            self._mp_ctx = self._mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
            if not standby:
                self.backend_name = "mediapipe"
                logger.info("MediaPipe pose backend ready.")
            else:
                logger.info("MediaPipe pose loaded as failover standby.")
            return True
        except Exception as e:
            logger.error("MediaPipe pose unavailable: %s", e)
            return False

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return PoseResult(landmarks=None, backend=self.backend_name, error="empty_frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return PoseResult(landmarks=None, backend=self.backend_name, error="bad_frame_shape")

        if self._yolo is not None and self.backend_name == "yolo":
            result = self._process_yolo(frame_bgr)
            if result.error:
                self._yolo_fail_streak += 1
                if self._yolo_fail_streak >= self.cfg.yolo_failover_errors:
                    self._failover_to_mediapipe()
                    if self._mp_ctx is not None:
                        return self._process_mediapipe(frame_bgr)
            else:
                self._yolo_fail_streak = 0
            return result

        if self._mp_ctx is not None:
            return self._process_mediapipe(frame_bgr)
        return PoseResult(landmarks=None, backend="none", error="no_backend")

    def _failover_to_mediapipe(self):
        logger.error(
            "YOLO failed %d times — failing over to MediaPipe.",
            self._yolo_fail_streak,
        )
        self._yolo = None
        if self._try_load_mediapipe(standby=False):
            self.backend_name = "mediapipe"
        else:
            self.backend_name = "none"

    def _process_yolo(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        try:
            results = self._yolo.predict(
                frame_bgr,
                verbose=False,
                conf=self.cfg.yolo_conf,
                iou=self.cfg.yolo_iou,
                max_det=3,
                device=self._device,
            )
        except Exception as e:
            logger.warning("YOLO inference failed: %s", e)
            return PoseResult(
                landmarks=None, backend="yolo", confidence=0.0, error=str(e)
            )

        try:
            if not results:
                return PoseResult(landmarks=None, backend="yolo")

            r0 = results[0]
            if r0.keypoints is None or len(r0.keypoints) == 0:
                return PoseResult(landmarks=None, backend="yolo")

            best_i = 0
            best_conf = 0.0
            bbox = None
            if r0.boxes is not None and len(r0.boxes):
                confs = r0.boxes.conf.detach().cpu().numpy()
                best_i = int(np.argmax(confs))
                best_conf = float(confs[best_i])
                xyxy = r0.boxes.xyxy.detach().cpu().numpy()[best_i]
                bbox = (
                    int(max(0, xyxy[0])),
                    int(max(0, xyxy[1])),
                    int(min(w, xyxy[2])),
                    int(min(h, xyxy[3])),
                )

            kpts = r0.keypoints
            xy = kpts.xy.detach().cpu().numpy()
            conf = (
                kpts.conf.detach().cpu().numpy()
                if getattr(kpts, "conf", None) is not None
                else None
            )
            if len(xy) == 0:
                return PoseResult(landmarks=None, backend="yolo", person_bbox=bbox)
            if best_i >= len(xy):
                best_i = 0
            person_xy = xy[best_i]
            person_conf = conf[best_i] if conf is not None else None
            if person_conf is not None:
                best_conf = max(best_conf, float(np.nanmean(person_conf)))

            lms = landmarks_from_coco(
                person_xy,
                person_conf,
                w,
                h,
                conf_threshold=self.cfg.yolo_kpt_conf,
            )
            raw_lms = lms
            torso_ok = sum(
                1 for i in (11, 12, 23, 24) if lms[i].visibility >= self.cfg.yolo_kpt_conf
            ) >= 3
            if not torso_ok:
                return PoseResult(
                    landmarks=None,
                    backend="yolo",
                    person_bbox=bbox,
                    confidence=best_conf,
                    raw_landmarks=raw_lms,
                )
            return PoseResult(
                landmarks=lms,
                backend="yolo",
                person_bbox=bbox,
                confidence=best_conf,
                raw_landmarks=raw_lms,
            )
        except Exception as e:
            logger.warning("YOLO result parse failed: %s", e)
            return PoseResult(landmarks=None, backend="yolo", error=str(e))

    def _process_mediapipe(self, frame_bgr: np.ndarray) -> PoseResult:
        import cv2

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_ctx.process(rgb)
            if not results.pose_landmarks:
                return PoseResult(landmarks=None, backend="mediapipe")
            lms = snapshot_landmarks(results.pose_landmarks.landmark)
            vis = float(np.mean([lms[i].visibility for i in (11, 12, 23, 24)]))
            return PoseResult(landmarks=lms, backend="mediapipe", confidence=vis)
        except Exception as e:
            logger.warning("MediaPipe inference failed: %s", e)
            return PoseResult(landmarks=None, backend="mediapipe", error=str(e))

    # Drawing deliberately lives only in medisense.ui.render. This class used
    # to draw its own skeleton and bounding box as well, so every frame carried
    # two skeletons in two different colours on top of each other.

    def close(self):
        try:
            if self._mp_ctx is not None:
                self._mp_ctx.close()
        except Exception:
            pass
        self._mp_ctx = None
        self._yolo = None
