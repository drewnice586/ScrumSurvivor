"""Blink effect — momentarily narrows the avatar's eye region."""

from __future__ import annotations

import random
import time

import cv2
import numpy as np


# Number of frames to animate a full blink (close + open)
_BLINK_FRAMES = 4


class BlinkEffect:
    """Simulates blinking by briefly darkening the eye-region strip.

    Args:
        interval_range: ``(min_s, max_s)`` range of seconds between blinks.
        frame_size: ``(width, height)`` of the avatar frame (used to compute
                    the eye-band y-range as a fraction of the height).
    """

    # Eye region as fraction of frame height (rough heuristic)
    _EYE_TOP_FRAC = 0.25
    _EYE_BOT_FRAC = 0.42

    def __init__(
        self,
        interval_range: tuple[float, float] = (3.0, 6.0),
        frame_size: tuple[int, int] = (1280, 720),
    ) -> None:
        self._min_s, self._max_s = interval_range
        self._frame_size = frame_size
        self._next_blink = time.monotonic() + self._next_interval()
        self._phase = 0         # 0 = not blinking, 1-_BLINK_FRAMES = in blink
        self._blink_progress = 0

    def _next_interval(self) -> float:
        return random.uniform(self._min_s, self._max_s)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply blink animation to *frame* and return the result."""
        now = time.monotonic()
        h, w = frame.shape[:2]

        if self._phase == 0:
            if now >= self._next_blink:
                self._phase = 1
        
        if self._phase == 0:
            return frame

        # Animate blink: darken eye band proportional to phase
        result = frame.copy()
        y1 = int(h * self._EYE_TOP_FRAC)
        y2 = int(h * self._EYE_BOT_FRAC)

        # Phase 1..2 = closing, 3..4 = opening
        half = _BLINK_FRAMES // 2
        if self._phase <= half:
            alpha = self._phase / half
        else:
            alpha = (_BLINK_FRAMES - self._phase + 1) / half

        alpha = max(0.0, min(1.0, alpha))
        result[y1:y2, :] = (result[y1:y2, :] * (1.0 - alpha)).astype(np.uint8)

        self._phase += 1
        if self._phase > _BLINK_FRAMES:
            self._phase = 0
            self._next_blink = now + self._next_interval()

        return result
