"""
Face crop utilities for lying-patient expression analysis.

Uses OpenCV Haar (frontal + profile) with ROI biasing from YOLO bbox /
pose landmarks. Falls back to a synthetic nose/eye crop when cascades
miss (common for side-lying patients).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("medisense.vision.face")


class FaceFinder:
    def __init__(self):
        self._frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if self._frontal.empty():
            raise RuntimeError("Failed to load frontal Haar cascade")
        if self._profile.empty():
            logger.warning("Profile Haar cascade missing — frontal only.")
            self._profile = None

    def find(
        self,
        frame_bgr: np.ndarray,
        landmarks=None,
        person_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> tuple[bool, Optional[tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Returns (face_visible, bbox_xywh, face_crop_bgr).
        Never raises — returns (False, None, None) on bad input.
        """
        try:
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                return False, None, None
            h, w = frame_bgr.shape[:2]
            if h < 32 or w < 32:
                return False, None, None

            search = frame_bgr
            offset_x, offset_y = 0, 0

            roi = self._head_roi(w, h, landmarks, person_bbox)
            if roi is not None:
                x0, y0, x1, y1 = roi
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 >= 24 and y1 - y0 >= 24:
                    search = frame_bgr[y0:y1, x0:x1]
                    offset_x, offset_y = x0, y0

            faces = self._detect(search)
            if len(faces) == 0 and (offset_x or offset_y):
                faces = self._detect(frame_bgr)
                offset_x, offset_y = 0, 0

            if len(faces) == 0:
                synth = self._synthetic_face_box(w, h, landmarks)
                if synth is None:
                    return False, None, None
                return self._crop(frame_bgr, synth)

            x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            box = (int(x + offset_x), int(y + offset_y), int(fw), int(fh))
            return self._crop(frame_bgr, box)
        except Exception as e:
            logger.debug("FaceFinder error: %s", e)
            return False, None, None

    def _detect(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = list(
            self._frontal.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36)
            )
        )
        if not faces and self._profile is not None:
            faces = list(
                self._profile.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
            )
            if not faces:
                # Try mirrored profile (patient facing other way).
                flipped = cv2.flip(gray, 1)
                raw = self._profile.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
                w = gray.shape[1]
                faces = [(w - (x + fw), y, fw, fh) for (x, y, fw, fh) in raw]
        return faces

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        x, y, fw, fh = box
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        fw = max(1, min(fw, w - x))
        fh = max(1, min(fh, h - y))
        if fw < 24 or fh < 24:
            return False, None, None
        crop = frame[y:y + fh, x:x + fw]
        if crop is None or crop.size == 0:
            return False, None, None
        return True, (x, y, fw, fh), crop.copy()

    def _head_roi(self, w, h, landmarks, person_bbox):
        if landmarks is not None:
            try:
                face_idxs = [0, 2, 5, 7, 8, 11, 12]
                pts = [
                    landmarks[i]
                    for i in face_idxs
                    if i < len(landmarks) and landmarks[i].visibility > 0.3
                ]
                if len(pts) >= 2:
                    xs = [p.x for p in pts]
                    ys = [p.y for p in pts]
                    cx, cy = float(np.mean(xs)), float(np.mean(ys))
                    pad = 0.22
                    x0 = max(0, int((cx - pad) * w))
                    y0 = max(0, int((cy - pad) * h))
                    x1 = min(w, int((cx + pad) * w))
                    y1 = min(h, int((cy + pad) * h))
                    if x1 - x0 > 20 and y1 - y0 > 20:
                        return x0, y0, x1, y1
            except Exception:
                pass

        if person_bbox is not None:
            try:
                x1, y1, x2, y2 = person_bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                bh = max(1, y2 - y1)
                box_w = max(1, x2 - x1)
                if box_w > bh * 1.2:
                    mid = (x1 + x2) // 2
                    return max(0, mid - box_w // 4), y1, min(w, mid + box_w // 4), y2
                return x1, y1, x2, y1 + int(bh * 0.45)
            except Exception:
                return None
        return None

    def _synthetic_face_box(self, w, h, landmarks):
        if landmarks is None or len(landmarks) < 1:
            return None
        try:
            nose = landmarks[0]
            if nose.visibility < 0.25:
                return None
            size = int(min(w, h) * 0.18)
            cx, cy = int(nose.x * w), int(nose.y * h)
            x = max(0, cx - size // 2)
            y = max(0, cy - size // 2)
            fw = min(size, w - x)
            fh = min(size, h - y)
            if fw < 24 or fh < 24:
                return None
            return x, y, fw, fh
        except Exception:
            return None
