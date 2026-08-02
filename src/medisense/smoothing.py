"""
Shared smoothing utilities.

These exist because raw per-frame pose/emotion readings are jittery —
a single noisy frame can flip a label from CALM to HIGH and back.
Averaging over a short rolling window (Smoother) and taking the
majority vote of recent labels (LabelSmoother) removes that flicker
without materially delaying real events.
"""
from collections import deque, Counter
import numpy as np


class Smoother:
    """Averages the last N numeric values to reduce frame-to-frame jitter."""

    def __init__(self, window: int):
        self.buf = deque(maxlen=window)

    def update(self, val: float) -> float:
        self.buf.append(val)
        return float(np.mean(self.buf))

    def value(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0


class LabelSmoother:
    """Returns the most common label seen in the last N updates (majority vote)."""

    def __init__(self, window: int, initial: str = "READING"):
        self.buf = deque(maxlen=window)
        self.current = initial

    def update(self, label: str) -> str:
        self.buf.append(label)
        self.current = Counter(self.buf).most_common(1)[0][0]
        return self.current

    def value(self) -> str:
        return self.current
