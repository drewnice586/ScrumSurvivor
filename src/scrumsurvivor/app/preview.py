"""Local preview window using OpenCV imshow."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_WINDOW_TITLE = "ScrumSurvivor Preview (press Q to close)"


class PreviewWindow:
    """Shows the pipeline output frame in a local OpenCV window.

    Only active when ``preview_enabled`` is True in config.
    Must be called from the main thread on Windows.

    Usage::

        preview = PreviewWindow(scale=0.5)
        preview.open()
        preview.update(frame_bgr)
        preview.close()
    """

    def __init__(self, scale: float = 0.5) -> None:
        self._scale = scale
        self._open = False

    def open(self) -> None:
        cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        self._open = True

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Display *frame_bgr*. Returns False if the window was closed (Q pressed)."""
        if not self._open:
            return True
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (int(w * self._scale), int(h * self._scale)))
        cv2.imshow(_WINDOW_TITLE, small)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            self.close()
            return False
        return True

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(_WINDOW_TITLE)
            self._open = False
