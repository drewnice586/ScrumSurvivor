"""Tests for VirtualCameraOutput (non-hardware -- mocked pyvirtualcam)."""

from __future__ import annotations

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_send_raises_when_not_open():
    from scrumsurvivor.output.virtual_camera import VirtualCameraOutput

    vcam = VirtualCameraOutput()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not open"):
        vcam.send(frame)


def test_send_converts_bgr_to_rgb():
    """Ensure BGR->RGB conversion is applied before sending."""
    from scrumsurvivor.output.virtual_camera import VirtualCameraOutput

    mock_cam = MagicMock()
    pvc_mock = MagicMock()
    pvc_mock.Camera.return_value = mock_cam

    with patch.dict(sys.modules, {"pyvirtualcam": pvc_mock}):
        vcam = VirtualCameraOutput(width=4, height=4, fps=25)
        vcam.open()

        # Pure blue BGR frame -> should become pure red in RGB
        frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = 255  # blue channel in BGR

        vcam.send(frame_bgr)

        sent_frame = mock_cam.send.call_args[0][0]
        # After BGR->RGB: channel 0 = R = 0, channel 2 = B = 255
        assert sent_frame[0, 0, 0] == 0    # R
        assert sent_frame[0, 0, 2] == 255  # B

        vcam.close()


def test_send_resizes_when_dimensions_mismatch():
    """Frame is resized to camera resolution when dimensions don't match."""
    from scrumsurvivor.output.virtual_camera import VirtualCameraOutput

    mock_cam = MagicMock()
    pvc_mock = MagicMock()
    pvc_mock.Camera.return_value = mock_cam

    with patch.dict(sys.modules, {"pyvirtualcam": pvc_mock}):
        vcam = VirtualCameraOutput(width=1280, height=720, fps=25)
        vcam.open()

        small_frame = np.zeros((96, 96, 3), dtype=np.uint8)
        vcam.send(small_frame)

        sent = mock_cam.send.call_args[0][0]
        assert sent.shape == (720, 1280, 3)

        vcam.close()


@pytest.mark.hardware
def test_virtual_camera_opens_and_sends():
    """Integration: open real virtual camera and send one frame."""
    from scrumsurvivor.output.virtual_camera import VirtualCameraOutput

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with VirtualCameraOutput(width=1280, height=720, fps=25) as vcam:
        vcam.send(frame)
