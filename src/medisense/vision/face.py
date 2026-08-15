"""
Face crop utilities for lying-patient expression analysis.

Uses OpenCV Haar (frontal + profile) with ROI biasing from YOLO bbox /
pose landmarks. Falls back to a synthetic nose/eye crop when cascades
miss (common for side-lying patients).

The result reports HOW the face was found, because the two paths are not
equally trustworthy. A cascade detection is a face; a synthetic crop is a
square guessed around the nose landmark, which may be an ear, a pillow, or
bedding. An expression classifier returns a confident-looking label for any
image it is handed, so feeding it synthetic crops manufactures emotions out of
pillows. Callers use `source` to decide what a crop is good enough for.

Crops are also rotated upright when eye or ear landmarks allow it. Facial
expression models are trained on upright faces, and a side-lying patient's
face arrives rotated by up to ninety degrees.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import cv2
import numpy as np

from medisense.vision.cascades import load_cascade

logger = logging.getLogger("medisense.vision.face")

FRONTAL = "frontal"
PROFILE = "profile"
LANDMARK = "landmark"


class FaceFind(NamedTuple):
    """Indexable for backwards compatibility, named for clarity."""

    found: bool
    box: Optional[tuple[int, int, int, int]]
    crop: Optional[np.ndarray]
    source: Optional[str] = None
    roll_deg: float = 0.0

    @property
    def is_detection(self) -> bool:
        """True only when a cascade actually found a face."""
        return self.source in (FRONTAL, PROFILE)


NO_FACE = FaceFind(False, None, None, None, 0.0)


class FaceFinder:
    def __init__(self):
        self._frontal = load_cascade("haarcascade_frontalface_default.xml")
        self._profile = load_cascade("haarcascade_profileface.xml")
        if self._frontal is None:
            # Landmark-derived crops still work, so degrade instead of dying.
            logger.warning(
                "Frontal Haar cascade unavailable — falling back to "
                "landmark-derived face crops only."
            )
        if self._profile is None:
            logger.warning("Profile Haar cascade missing — frontal only.")

    def find(
        self,
        frame_bgr: np.ndarray,
        landmarks=None,
        person_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> FaceFind:
        """
        Locate the patient's face.

        Never raises — returns `NO_FACE` on bad input. Check `.is_detection`
        before using the crop for anything that infers state from appearance.
        """
        try:
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                return NO_FACE
            h, w = frame_bgr.shape[:2]
            if h < 32 or w < 32:
                return NO_FACE

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

            faces, source = self._detect(search)
            if len(faces) == 0 and (offset_x or offset_y):
                faces, source = self._detect(frame_bgr)
                offset_x, offset_y = 0, 0

            if len(faces) == 0:
                synth = self._synthetic_face_box(w, h, landmarks)
                if synth is None:
                    return NO_FACE
                return self._crop(frame_bgr, synth, LANDMARK, landmarks)

            x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            box = (int(x + offset_x), int(y + offset_y), int(fw), int(fh))
            return self._crop(frame_bgr, box, source, landmarks)
        except Exception as e:
            logger.debug("FaceFinder error: %s", e)
            return NO_FACE

    def _detect(self, image_bgr: np.ndarray):
        if self._frontal is None and self._profile is None:
            return [], None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if self._frontal is not None:
            faces = list(
                self._frontal.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36)
                )
            )
            if faces:
                return faces, FRONTAL
        if self._profile is not None:
            faces = list(
                self._profile.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
            )
            if faces:
                return faces, PROFILE
            # Try mirrored profile (patient facing other way).
            flipped = cv2.flip(gray, 1)
            raw = self._profile.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
            )
            if len(raw):
                w = gray.shape[1]
                return [(w - (x + fw), y, fw, fh) for (x, y, fw, fh) in raw], PROFILE
        return [], None

    def _crop(self, frame, box, source, landmarks) -> FaceFind:
        h, w = frame.shape[:2]
        x, y, fw, fh = box
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        fw = max(1, min(fw, w - x))
        fh = max(1, min(fh, h - y))
        if fw < 24 or fh < 24:
            return NO_FACE

        roll = self._roll_degrees(landmarks)
        crop = self._upright_crop(frame, (x, y, fw, fh), roll)
        if crop is None or crop.size == 0:
            return NO_FACE
        return FaceFind(True, (x, y, fw, fh), crop, source, roll)

    def _roll_degrees(self, landmarks) -> float:
        """
        Head roll from eyes, falling back to ears.

        A bedridden patient's face is often rotated most of the way onto its
        side, which is exactly the orientation expression models handle worst.
        """
        if landmarks is None:
            return 0.0
        for a, b in ((2, 5), (7, 8)):
            try:
                left, right = landmarks[a], landmarks[b]
            except (IndexError, TypeError):
                continue
            if min(
                float(getattr(left, "visibility", 0.0) or 0.0),
                float(getattr(right, "visibility", 0.0) or 0.0),
            ) < 0.4:
                continue
            dx = float(right.x) - float(left.x)
            dy = float(right.y) - float(left.y)
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            return float(np.degrees(np.arctan2(dy, dx)))
        return 0.0

    def _upright_crop(self, frame, box, roll_deg: float):
        """Crop the face, de-rotated about its centre when roll is meaningful."""
        x, y, fw, fh = box
        if abs(roll_deg) < 12.0:
            return frame[y:y + fh, x:x + fw].copy()

        cx, cy = x + fw / 2.0, y + fh / 2.0
        # Rotate a generous window so corners are filled after rotation.
        pad = int(max(fw, fh) * 0.75)
        h, w = frame.shape[:2]
        wx0, wy0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        wx1, wy1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        window = frame[wy0:wy1, wx0:wx1]
        if window.size == 0:
            return frame[y:y + fh, x:x + fw].copy()

        m = cv2.getRotationMatrix2D((cx - wx0, cy - wy0), roll_deg, 1.0)
        rotated = cv2.warpAffine(
            window, m, (window.shape[1], window.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        rx0 = int(max(0, (cx - wx0) - fw / 2.0))
        ry0 = int(max(0, (cy - wy0) - fh / 2.0))
        crop = rotated[ry0:ry0 + fh, rx0:rx0 + fw]
        if crop.size == 0:
            return frame[y:y + fh, x:x + fw].copy()
        return crop.copy()

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
