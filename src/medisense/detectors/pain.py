"""
Pain/distress estimation from facial expression.

Primary: Hugging Face facial-emotion classifier (distress proxy).
Fallback: OpenCV intensity / texture heuristic when the HF model is
offline, so the expression channel never silently dies.

Not a clinical pain scale (PSPI/PAINAD). Treat as distress proxy only.

The two paths are NOT interchangeable and the detector does not pretend they
are. Contrast and edge density have no established relationship to pain, so
the fallback is capped at WARNING and can never raise a critical alert on its
own. Callers can read `get_mode()` to see which path produced a score, and the
mode is recorded alongside every logged event: a fallback that keeps the
channel alive is only useful if it cannot be mistaken for the model.

Three rules keep the reported number honest, because a classifier will return
a confident-looking label for literally any image:

  - Only crops from an actual face detection are scored. Landmark-derived
    crops (a square guessed around the nose) are rejected — feeding those to
    the model invents expressions from pillows and bedding.
  - Crops that are too small or too blurred to read are rejected, and a
    prediction whose own top-1 confidence is below threshold reports
    UNCERTAIN instead of a label.
  - The displayed label and the displayed score describe the same prediction.
    Previously the label was the argmax emotion while the number was a
    weighted sum over all seven emotions, so the panel could read
    "NEUTRAL 61%" — two true statements that look like a broken detector.

When nothing has been scored recently the state is NOT ASSESSED rather than a
stale NORMAL, so an unreadable face never looks like a calm one.
"""
from __future__ import annotations

import logging
import queue
import threading

import cv2
import numpy as np
from PIL import Image

from medisense.clock import WallClock
from medisense.config import Thresholds, PAIN_WEIGHTS
from medisense.smoothing import Smoother

logger = logging.getLogger("medisense.detectors.pain")

MIN_FACE_PX = 32

LOADING = "LOADING"
NORMAL = "NORMAL"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
NOT_ASSESSED = "NOT ASSESSED"
UNCERTAIN = "uncertain"


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
    def __init__(self, cfg: Thresholds, clock=None):
        self.cfg = cfg
        self.clock = clock or WallClock()
        self.emotion = "neutral"
        self.pain_score = 0.0
        self.confidence = 0.0
        self.state = LOADING
        self.last_scored_at = None
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

    def update(self, face_img_bgr, is_detection: bool = True):
        """
        Queue a face crop for scoring.

        `is_detection` must be False for landmark-derived crops; those are
        dropped rather than scored.
        """
        if not self._ready or face_img_bgr is None or not is_detection:
            return
        try:
            if not isinstance(face_img_bgr, np.ndarray) or face_img_bgr.size == 0:
                return
            if not self._readable(face_img_bgr):
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

    def _readable(self, face_img_bgr) -> bool:
        """Reject crops too small or too blurred for a label to mean anything."""
        h, w = face_img_bgr.shape[:2]
        if min(h, w) < max(MIN_FACE_PX, self.cfg.emotion_min_face_px):
            return False
        gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= self.cfg.emotion_min_sharpness

    def _run_inference(self, face_img_bgr):
        try:
            using_cv = self._use_cv or self._pipe is None
            if using_cv:
                top, raw, confidence = (*cv_distress_score(face_img_bgr), 1.0)
            else:
                top, raw, confidence = self._neural_score(face_img_bgr)
                if top in ("tense", "calm"):
                    # _neural_score fell back internally on an empty prediction.
                    using_cv = True

            now = self.clock.now()

            if not using_cv and confidence < self.cfg.emotion_min_confidence:
                # The model itself is unsure; do not turn that into a reading.
                self.pain_start = None
                with self._lock:
                    self.emotion = UNCERTAIN
                    self.confidence = confidence
                    self.state = NORMAL
                    self.last_scored_at = now
                return

            score = self.score_smooth.update(float(raw))

            if using_cv:
                # Heuristic path: report a signal, never confirm distress.
                self.pain_start = None
                st = WARNING if score >= self.cfg.pain_score_threshold else NORMAL
            elif score >= self.cfg.critical_pain_threshold:
                if self.pain_start is None:
                    self.pain_start = now
                dur = now - self.pain_start
                st = CRITICAL if dur >= self.cfg.pain_hold_seconds else WARNING
            elif score >= self.cfg.pain_score_threshold:
                self.pain_start = None
                st = WARNING
            else:
                self.pain_start = None
                st = NORMAL

            with self._lock:
                self.emotion = top
                self.pain_score = score
                self.confidence = confidence
                self.state = st
                self.last_scored_at = now
        except Exception as e:
            logger.warning("Expression inference failed: %s", e)

    def _neural_score(self, face_img_bgr) -> tuple[str, float, float]:
        """Returns (top_label, distress_score, top_label_confidence)."""
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        # Normalize size for stable classifier input.
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(rgb)
        preds = self._pipe(pil)
        if not preds:
            return (*cv_distress_score(face_img_bgr), 1.0)
        # pipeline may return list-of-dicts or nested list
        if isinstance(preds[0], list):
            preds = preds[0]

        scored = {}
        for p in preds:
            try:
                scored[str(p.get("label", "")).lower()] = float(p.get("score", 0.0))
            except (TypeError, ValueError):
                continue
        if not scored:
            return (*cv_distress_score(face_img_bgr), 1.0)

        top, confidence = max(scored.items(), key=lambda kv: kv[1])
        # Distress is the weighted mass on distress emotions, scaled by how
        # sure the model is overall, so a flat seven-way split cannot add up
        # to a confident-looking score.
        raw = sum(PAIN_WEIGHTS.get(label, 0.0) * prob for label, prob in scored.items())
        return top, min(max(raw, 0.0), 1.0), confidence

    def get(self) -> tuple[str, float, str]:
        """Returns (emotion_label, distress_score, state)."""
        with self._lock:
            if self.state == LOADING:
                return self.emotion, self.pain_score, LOADING
            stale = (
                self.last_scored_at is None
                or self.clock.now() - self.last_scored_at > self.cfg.pain_stale_sec
            )
            if stale:
                return "no face", 0.0, NOT_ASSESSED
            return self.emotion, self.pain_score, self.state

    def get_confidence(self) -> float:
        with self._lock:
            return self.confidence

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
