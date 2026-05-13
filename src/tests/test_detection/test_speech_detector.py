"""Tests for SpeechDetector (non-hardware)."""

from __future__ import annotations

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_detector(threshold: float = 0.05, attack_ms: int = 0, release_ms: int = 0):
    from scrumsurvivor.detection.speech_detector import SpeechDetector

    return SpeechDetector(
        threshold=threshold,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )


def test_initial_state_is_not_speaking():
    detector = _make_detector()
    assert detector.is_speaking is False


def test_loud_chunk_sets_speaking_immediately():
    """With attack_ms=0 a loud chunk should trigger speaking on first call."""
    detector = _make_detector(threshold=0.02, attack_ms=0)
    loud = np.full(512, 0.5, dtype=np.float32)
    changed = detector.update(loud)
    assert detector.is_speaking is True
    assert changed is True


def test_silent_chunk_does_not_trigger_speaking():
    detector = _make_detector(threshold=0.02)
    silent = np.zeros(512, dtype=np.float32)
    detector.update(silent)
    assert detector.is_speaking is False


def test_speech_then_silence_releases():
    """After speaking, sustained silence triggers release."""
    detector = _make_detector(threshold=0.02, attack_ms=0, release_ms=0)
    loud = np.full(512, 0.5, dtype=np.float32)
    silent = np.zeros(512, dtype=np.float32)

    detector.update(loud)
    assert detector.is_speaking is True

    changed = detector.update(silent)
    assert detector.is_speaking is False
    assert changed is True


def test_calibrate_uses_config_threshold_when_set():
    """calibrate() should skip measurement when threshold is explicitly set."""
    from scrumsurvivor.detection.speech_detector import SpeechDetector

    detector = SpeechDetector(threshold=0.123)
    mock_mic = MagicMock()
    result = detector.calibrate(mock_mic)

    assert result == pytest.approx(0.123)
    mock_mic.read.assert_not_called()


def test_calibrate_computes_threshold():
    """calibrate() measures ambient RMS and sets threshold = mean + 2*std."""
    from scrumsurvivor.detection.speech_detector import SpeechDetector
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    detector = SpeechDetector(threshold=None, sample_rate=44100)

    # Feed low-noise chunks: all ~0.01 amplitude
    chunks = [np.full(512, 0.01, dtype=np.float32) for _ in range(50)]
    chunks_iter = iter(chunks)

    mock_mic = MagicMock()
    mock_mic.read.side_effect = lambda block=False: next(chunks_iter, None)

    threshold = detector.calibrate(mock_mic)
    # With uniform chunks, std ≈ 0, so threshold ≈ mean ≈ 0.01
    assert 0.005 < threshold < 0.1
