"""Shared pytest fixtures and configuration for all tests."""

from __future__ import annotations

import numpy as np
import pytest


# ── Markers ──────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: tests that require physical hardware (webcam, GPU, VB-Cable). "
        "Run with: pytest -m hardware",
    )
    config.addinivalue_line(
        "markers",
        "gpu: tests that require an NVIDIA GPU with CUDA. "
        "Run with: pytest -m gpu",
    )


# ── Image/Array Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def blank_bgr_frame():
    """A black 720p BGR frame (OpenCV convention)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def small_bgr_frame():
    """A small 96×96 BGR frame for unit tests."""
    return np.zeros((96, 96, 3), dtype=np.uint8)


@pytest.fixture
def face_bgr_frame():
    """A 400×300 BGR frame simulating a cropped face region."""
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    # Draw a simple face placeholder (white rectangle)
    frame[50:350, 50:250] = 200
    return frame


# ── Audio Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def silent_audio_chunk():
    """512 samples of silence at 44100 Hz (float32)."""
    return np.zeros(512, dtype=np.float32)


@pytest.fixture
def loud_audio_chunk():
    """512 samples of loud audio at 44100 Hz (float32, ~0.8 amplitude)."""
    return np.full(512, 0.8, dtype=np.float32)


@pytest.fixture
def sine_audio_16k():
    """1 second of 440 Hz sine wave at 16000 Hz sample rate (float32)."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ── Config Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def default_config(tmp_path):
    """An AppConfig instance with all defaults, config_path pointing to tmp_path."""
    from scrumsurvivor.config.settings import AppConfig
    return AppConfig()


@pytest.fixture
def config_file(tmp_path):
    """Path to a temporary config.yaml with default values written out."""
    from scrumsurvivor.config.settings import generate_default_config
    config_path = str(tmp_path / "config.yaml")
    generate_default_config(config_path)
    return config_path
