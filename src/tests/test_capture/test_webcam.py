"""Tests for WebcamCapture (non-hardware)."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_webcam_read_returns_none_before_first_frame():
    """Before any frame arrives, read() returns None."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        from scrumsurvivor.capture.webcam import WebcamCapture

        cam = WebcamCapture(device=0, target_fps=25)
        # Don't open — just confirm attribute initialises to None
        assert cam.read() is None


def test_webcam_is_open_false_before_open():
    from scrumsurvivor.capture.webcam import WebcamCapture

    cam = WebcamCapture(device=0)
    assert cam.is_open is False


@pytest.mark.hardware
def test_webcam_opens_and_reads_frame():
    """Integration: actually opens default webcam."""
    import cv2
    from scrumsurvivor.capture.webcam import WebcamCapture
    import time

    with WebcamCapture(device=0, target_fps=25) as cam:
        assert cam.is_open
        time.sleep(0.1)  # Give background thread time to capture a frame
        frame = cam.read()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
