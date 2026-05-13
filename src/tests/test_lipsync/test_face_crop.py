"""Tests for FaceCropManager."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


def _make_face_photo() -> np.ndarray:
    """Return a 720×400 BGR test image."""
    return np.zeros((720, 400, 3), dtype=np.uint8)


def test_get_crop_returns_96x96():
    """get_crop() must return a 96×96 image when face is detected."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = _make_face_photo()
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 100, 200, 200),
    ):
        mgr = FaceCropManager(base_photo=photo)
        crop = mgr.get_crop()
    assert crop is not None
    assert crop.shape == (96, 96, 3)


def test_get_crop_returns_none_when_no_face():
    """get_crop() returns None when face detection fails."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = _make_face_photo()
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=None,
    ):
        mgr = FaceCropManager(base_photo=photo)
        crop = mgr.get_crop()
    assert crop is None


def test_get_crop_can_use_current_frame_instead_of_static_base_photo():
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    base_photo = np.zeros((200, 200, 3), dtype=np.uint8)
    current_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    current_frame[40:140, 60:160] = 180

    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(60, 40, 100, 100),
    ):
        mgr = FaceCropManager(base_photo=base_photo)
        crop = mgr.get_crop(current_frame)

    assert crop is not None
    assert crop.shape == (96, 96, 3)
    assert crop[48, 48, 0] == 180


def test_paste_back_only_modifies_crop_region():
    """paste_back() should only change pixels inside the mouth region."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = np.full((400, 300, 3), 50, dtype=np.uint8)
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 50, 100, 100),
    ):
        mgr = FaceCropManager(base_photo=photo)
        mgr.detect()

    synced_face = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = mgr.paste_back(photo.copy(), synced_face)

    # Pixels well outside the crop rect are untouched
    assert result[0, 0, 0] == 50
    # Pixels in the mouth region (lower 55 % of crop) are blended
    x, y, w, h = 50, 50, 100, 100
    mouth_y = y + int(h * 0.75)   # well below the 45 % blend boundary
    mouth_x = x + w // 2
    assert result[mouth_y, mouth_x, 0] != 50


def test_paste_back_keeps_cheeks_from_base_photo():
    """paste_back() should not repaint wide cheek/jaw areas from the 96x96 output."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = np.full((400, 300, 3), 50, dtype=np.uint8)
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 50, 100, 100),
    ):
        mgr = FaceCropManager(base_photo=photo)
        mgr.detect()

    synced_face = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = mgr.paste_back(photo.copy(), synced_face)

    cheek_y = 50 + int(100 * 0.72)
    left_cheek_x = 50 + int(100 * 0.15)
    right_cheek_x = 50 + int(100 * 0.85)

    assert result[cheek_y, left_cheek_x, 0] == 50
    assert result[cheek_y, right_cheek_x, 0] == 50


def test_paste_back_keeps_lower_chin_from_base_photo():
    """paste_back() should not slip the generated mouth down onto the chin."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = np.full((400, 300, 3), 50, dtype=np.uint8)
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 50, 100, 100),
    ):
        mgr = FaceCropManager(base_photo=photo)
        mgr.detect()

    synced_face = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = mgr.paste_back(photo.copy(), synced_face)

    chin_y = 50 + int(100 * 0.90)
    chin_x = 50 + 100 // 2

    assert result[chin_y, chin_x, 0] == 50


def test_detect_calls_face_detector():
    """detect() should delegate to detect_face_once."""
    from scrumsurvivor.lipsync.face_crop import FaceCropManager

    photo = _make_face_photo()
    mgr = FaceCropManager(base_photo=photo)

    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 100, 200, 300),
    ) as mock_detect:
        rect = mgr.detect()
        mock_detect.assert_called_once_with(photo)
        assert rect == (50, 100, 200, 300)
