"""
Frame source resolution.

Accepts a webcam index, a video file path, or a stream URL, and reports which
kind it opened. The distinction matters twice over: a file that runs out of
frames has finished rather than failed, and a file needs a video-driven clock
so second-based thresholds stay meaningful during replay.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2

from medisense.clock import Clock, VideoClock, WallClock

logger = logging.getLogger("medisense.vision.capture")

CAMERA = "camera"
FILE = "file"
STREAM = "stream"

_STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def resolve_source(raw: Union[str, int]) -> tuple[Union[str, int], str]:
    """
    Returns (source_for_opencv, kind).

    Digits become a camera index, URLs become streams, anything else is
    treated as a file path.
    """
    if isinstance(raw, int):
        return raw, CAMERA

    text = str(raw).strip()
    if text == "":
        return 0, CAMERA
    if text.isdigit():
        return int(text), CAMERA
    if text.lower().startswith(_STREAM_PREFIXES):
        return text, STREAM
    return text, FILE


def open_capture(raw: Union[str, int]) -> tuple[cv2.VideoCapture, str, Clock]:
    """
    Open a frame source and pair it with an appropriate clock.

    Raises FileNotFoundError for a missing file and RuntimeError if the source
    cannot be opened, so startup fails with a clear reason instead of looping
    on empty reads.
    """
    source, kind = resolve_source(raw)

    if kind == FILE and not Path(str(source)).exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {kind} source {source!r}")

    if kind == CAMERA or kind == STREAM:
        # Prefer the freshest frame over a backlog for live monitoring.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    if kind == FILE:
        fps = 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0
        clock: Clock = VideoClock(fps=fps)
        logger.info("Replaying file %s at declared %.2f fps (video clock).", source, fps)
    else:
        clock = WallClock()
        logger.info("Opened %s source %r (wall clock).", kind, source)

    return cap, kind, clock


def frame_position_msec(cap: cv2.VideoCapture) -> float:
    """Container timestamp for the frame just read, or 0.0 if unsupported."""
    try:
        return float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    except Exception:
        return 0.0
