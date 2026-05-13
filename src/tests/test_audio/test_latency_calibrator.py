"""Tests for LatencyCalibrator."""

from __future__ import annotations

import time
import numpy as np
import pytest
from unittest.mock import MagicMock


def test_calibrate_returns_positive_delay():
    from scrumsurvivor.audio.latency_calibrator import LatencyCalibrator

    # Mock engine that takes ~5ms per call
    def slow_process(face, mel):
        time.sleep(0.005)
        return np.zeros((96, 96, 3), dtype=np.uint8)

    engine = MagicMock()
    engine.process.side_effect = slow_process

    face = np.zeros((96, 96, 3), dtype=np.uint8)
    mel = np.zeros((1, 80, 16), dtype=np.float32)

    cal = LatencyCalibrator(engine, face, mel, iterations=5)
    delay = cal.calibrate()

    assert delay > 0
    assert delay > 20  # at least the safety margin


def test_calibrate_includes_safety_margin():
    from scrumsurvivor.audio.latency_calibrator import LatencyCalibrator, _SAFETY_MARGIN_MS

    engine = MagicMock()
    engine.process.return_value = np.zeros((96, 96, 3), dtype=np.uint8)

    cal = LatencyCalibrator(engine, np.zeros((96, 96, 3), dtype=np.uint8),
                            np.zeros((1, 80, 16), dtype=np.float32), iterations=3)
    delay = cal.calibrate()
    stats = cal.get_stats()

    assert delay == pytest.approx(stats["avg_ms"] + _SAFETY_MARGIN_MS, abs=1.0)


def test_get_stats_before_calibrate_raises():
    from scrumsurvivor.audio.latency_calibrator import LatencyCalibrator

    engine = MagicMock()
    cal = LatencyCalibrator(engine, np.zeros((96, 96, 3), dtype=np.uint8),
                            np.zeros((1, 80, 16), dtype=np.float32))
    with pytest.raises(RuntimeError, match="calibrate()"):
        cal.get_stats()


def test_get_stats_has_expected_keys():
    from scrumsurvivor.audio.latency_calibrator import LatencyCalibrator

    engine = MagicMock()
    engine.process.return_value = np.zeros((96, 96, 3), dtype=np.uint8)
    cal = LatencyCalibrator(engine, np.zeros((96, 96, 3), dtype=np.uint8),
                            np.zeros((1, 80, 16), dtype=np.float32), iterations=3)
    cal.calibrate()
    stats = cal.get_stats()
    for key in ("avg_ms", "min_ms", "max_ms", "std_ms", "recommended_delay_ms"):
        assert key in stats
