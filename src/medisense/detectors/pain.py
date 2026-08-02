"""
Pain/distress estimation from facial expression.

Primary: Hugging Face facial-emotion classifier (distress proxy).
Fallback: OpenCV intensity / texture heuristic when the HF model is
offline, so the expression channel never silently dies.

Not a clinical pain scale (PSPI/PAINAD). Treat as distress proxy only.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np
from PIL import Image

from medisense.config import Thresholds, PAIN_WEIGHTS
from medisense.smoothing import Smoother

logger = logging.getLogger("medisense.detectors.pain")

MIN_FACE_PX = 32


def cv_distress_score(face_img_bgr: np.ndarray) -> tuple[str, float]:
    """
    Lightweight OpenCV proxy when the neural emotion model is unavailable.
    Uses contrast + edge density in the face crop (grimacing / tension
    tends to raise local contrast). Bounded and smoothed upstream.
    """
    gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    contrast = float(np.std(gray)) / 128.0
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.mean(edges > 0))
    score = min(1.0, 0.55 * contrast + 0.90 * edge_density)
    label = "tense" if score >= 0.45 else "calm"
    return label, score


class PainDetector:
    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self.emotion = "neutral"
        self.pain_score = 0.0
        self.state = "LOADING"
        self.pain_start = None
        self.mode = "loading"  # loading | neural | cv_fallback | offline
        self._lock = threading.Lock()
        self._pipe = None
        self._ready = False
        self._use_cv = False
        self.score_smooth = Smoother(6)
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="pain-worker")
        self._worker.start()

    def _worker_loop(self):
        self._load_model()
        while not self._stop.is_set():
            try:
                face = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            if face is None:
                break
            self._run_inference(face)

    def _load_model(self):
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading facial emotion model...")
            self._pipe = hf_pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                top_k=7,
            )
            self._ready = True
            self._use_cv = False
            with self._lock:
                self.state = "NORMAL"
                self.mode = "neural"
            logger.info("Emotion model ready (neural).")
        except Exception as e:
            logger.error("Neural emotion model unavailable — CV fallback: %s", e)
            self._pipe = None
            self._ready = True  # CV path is ready immediately
            self._use_cv = True
            with self._lock:
                self.state = "NORMAL"
                self.mode = "cv_fallback"

    def update(self, face_img_bgr):
        if not self._ready or face_img_bgr is None:
            return
        try:
            if not isinstance(face_img_bgr, np.ndarray) or face_img_bgr.size == 0:
                return
            if face_img_bgr.shape[0] < MIN_FACE_PX or face_img_bgr.shape[1] < MIN_FACE_PX:
                return
        except Exception:
            return

        try:
            self._jobs.put_nowait(face_img_bgr.copy())
        except queue.Full:
            try:
                _ = self._jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(face_img_bgr.copy())
            except queue.Full:
                pass

    def _run_inference(self, face_img_bgr):
        try:
            if self._use_cv or self._pipe is None:
                top, raw = cv_distress_score(face_img_bgr)
            else:
                top, raw = self._neural_score(face_img_bgr)

            score = self.score_smooth.update(float(raw))
            now = time.time()

            if score >= self.cfg.critical_pain_threshold:
                if self.pain_start is None:
                    self.pain_start = now
                dur = now - self.pain_start
                st = "CRITICAL" if dur >= self.cfg.pain_hold_seconds else "WARNING"
            elif score >= self.cfg.pain_score_threshold:
                self.pain_start = None
                st = "WARNING"
            else:
                self.pain_start = None
                st = "NORMAL"

            with self._lock:
                self.emotion = top
                self.pain_score = score
                self.state = st
        except Exception as e:
            logger.warning("Expression inference failed: %s", e)

    def _neural_score(self, face_img_bgr) -> tuple[str, float]:
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        # Normalize size for stable classifier input.
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(rgb)
        preds = self._pipe(pil)
        if not preds:
            return cv_distress_score(face_img_bgr)
        # pipeline may return list-of-dicts or nested list
        if isinstance(preds[0], list):
            preds = preds[0]
        raw = sum(
            PAIN_WEIGHTS.get(str(p.get("label", "")).lower(), 0) * float(p.get("score", 0))
            for p in preds
        )
        raw = min(max(raw, 0.0), 1.0)
        top = str(preds[0].get("label", "neutral")).lower()
        return top, raw

    def get(self) -> tuple[str, float, str]:
        with self._lock:
            return self.emotion, self.pain_score, self.state

    def get_mode(self) -> str:
        with self._lock:
            return self.mode

    def close(self):
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                _ = self._jobs.get_nowait()
                self._jobs.put_nowait(None)
            except Exception:
                pass
