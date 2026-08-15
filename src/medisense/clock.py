"""
Time sources for duration-based detectors.

A live camera and a recorded file do not share a notion of time. Frames from
a file arrive as fast as they decode, so twenty seconds of footage can elapse
in two seconds of wall time — and every threshold expressed in seconds
(stillness, respiration, alert confirmation) silently means something
different. `VideoClock` advances on the file's own presentation timestamps so
a threshold tuned against a live feed measures the same thing during replay,
which is what makes offline precision/recall numbers comparable to live
behaviour.
"""
from __future__ import annotations

import time
from typing import Optional, Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def advance(self, position_msec: Optional[float] = None) -> float: ...


class WallClock:
    """Real time. Used for live cameras and network streams."""

    kind = "wall"

    def now(self) -> float:
        return time.time()

    def advance(self, position_msec: Optional[float] = None) -> float:
        return self.now()


class VideoClock:
    """
    Time as measured by the video being replayed.

    Prefers the container's own timestamp; falls back to frame count over
    declared FPS when a codec reports no position. Output is monotonic even
    across loop restarts, because detectors hold absolute timestamps and
    would compute negative durations if time went backwards.
    """

    kind = "video"

    def __init__(self, fps: float = 25.0):
        self.fps = fps if fps and fps > 0 else 25.0
        self._t = 0.0
        self._offset = 0.0
        self._frames = 0

    def now(self) -> float:
        return self._t

    def advance(self, position_msec: Optional[float] = None) -> float:
        self._frames += 1
        if position_msec is not None and position_msec > 0:
            raw = float(position_msec) / 1000.0
        else:
            raw = self._frames / self.fps

        shifted = raw + self._offset
        if shifted < self._t:
            # Source rewound (loop restart): keep the timeline moving forward.
            self._offset = self._t - raw
            shifted = self._t
        self._t = shifted
        return self._t
