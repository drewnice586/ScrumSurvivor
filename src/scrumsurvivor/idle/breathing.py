"""Breathing effect — gentle vertical oscillation of the avatar image."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np


class BreathingEffect:
    """Applies a sinusoidal vertical translation to simulate breathing.

    Args:
        rate_hz: Breathing cycles per second (default ~0.25 ≈ 15 breaths/min).
        amplitude_px: Maximum vertical displacement in pixels.
    """

    def __init__(self, rate_hz: float = 0.25, amplitude_px: float = 2.5) -> None:
        self._rate_hz = rate_hz
        self._amplitude_px = amplitude_px
        self._start = time.monotonic()

    def offset_at(self, t: float | None = None) -> float:
        """Return the current vertical pixel offset at time *t* (monotonic seconds)."""
        if t is None:
            t = time.monotonic()
        phase = 2 * math.pi * self._rate_hz * (t - self._start)
        return self._amplitude_px * math.sin(phase)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Shift *frame* vertically by the current breathing offset (fills edges with black)."""
        offset = int(round(self.offset_at()))
        if offset == 0:
            return frame
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, 0], [0, 1, offset]])
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
