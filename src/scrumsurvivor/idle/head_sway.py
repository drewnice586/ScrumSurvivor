"""Head sway effect — subtle horizontal + vertical sinusoidal drift."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np


class HeadSwayEffect:
    """Two-component (dual-sine) head sway in X and Y axes.

    Args:
        x_params: ``[freq1, amp1, freq2, amp2]`` Hz / px for horizontal sway.
        y_params: ``[freq1, amp1, freq2, amp2]`` Hz / px for vertical sway.
    """

    def __init__(
        self,
        x_params: list[float] | None = None,
        y_params: list[float] | None = None,
    ) -> None:
        self._x = x_params or [0.1, 1.5, 0.17, 1.0]
        self._y = y_params or [0.13, 1.0, 0.07, 0.8]
        self._start = time.monotonic()

    def offsets_at(self, t: float | None = None) -> tuple[float, float]:
        """Return ``(dx, dy)`` pixel offsets at time *t*."""
        if t is None:
            t = time.monotonic()
        elapsed = t - self._start
        f1x, a1x, f2x, a2x = self._x
        f1y, a1y, f2y, a2y = self._y
        dx = a1x * math.sin(2 * math.pi * f1x * elapsed) + a2x * math.sin(
            2 * math.pi * f2x * elapsed
        )
        dy = a1y * math.sin(2 * math.pi * f1y * elapsed) + a2y * math.sin(
            2 * math.pi * f2y * elapsed
        )
        return dx, dy

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply the current head sway offset to *frame*."""
        dx, dy = self.offsets_at()
        tx, ty = int(round(dx)), int(round(dy))
        if tx == 0 and ty == 0:
            return frame
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
