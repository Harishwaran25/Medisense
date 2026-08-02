"""Tests for Low-Light Night-Vision Adaptive Contrast Enhancer."""
import numpy as np
from medisense.vision.nightvision import NightVisionEnhancer


def test_normal_light_frame_passes_through():
    enhancer = NightVisionEnhancer(low_light_threshold=75.0)
    bright_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    out_frame, is_low_light, mean_lum = enhancer.process(bright_frame)

    assert is_low_light is False
    assert mean_lum > 75.0
    assert out_frame.shape == bright_frame.shape


def test_dark_frame_is_enhanced():
    enhancer = NightVisionEnhancer(low_light_threshold=75.0)
    dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    out_frame, is_low_light, mean_lum = enhancer.process(dark_frame)

    assert is_low_light is True
    assert mean_lum < 75.0
    assert out_frame.shape == dark_frame.shape
    # Enhanced frame mean should be brighter
    assert np.mean(out_frame) > np.mean(dark_frame)


def test_empty_frame_handling():
    enhancer = NightVisionEnhancer()
    out, is_low_light, mean_lum = enhancer.process(None)
    assert out is None
    assert is_low_light is False
    assert mean_lum == 0.0
