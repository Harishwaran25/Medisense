"""
Low-Light / Night-Vision Adaptive Contrast Enhancer.

Uses OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB
color space to enhance dark video frames for 24/7 dark-room patient monitoring.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger("medisense.vision.nightvision")


class NightVisionEnhancer:
    """Adaptive low-light booster for dark-room monitoring."""

    def __init__(self, low_light_threshold: float = 75.0, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.threshold = low_light_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.is_active = False

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool, float]:
        """
        Processes a BGR frame.

        Returns:
            (enhanced_frame, is_low_light, mean_luminance)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, False, 0.0

        # Convert to LAB color space to isolate Luminance (L channel)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        mean_luminance = float(np.mean(l_channel))
        is_low_light = mean_luminance < self.threshold
        self.is_active = is_low_light

        if is_low_light:
            # Apply CLAHE to L-channel to boost local contrast in dark areas
            enhanced_l = self.clahe.apply(l_channel)

            # Re-merge channels and convert back to BGR
            enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr, True, mean_luminance

        return frame_bgr, False, mean_luminance
