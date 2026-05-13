"""Tests for idle clip manager timing behavior."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import random
from unittest.mock import MagicMock, patch


def test_clip_manager_respects_post_speech_cooldown_before_starting_clip():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("dummy.mp4")]
    manager._loaded = True

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        manager.suppress_for(2.0)

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=11.0):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_not_called()


def test_clip_manager_can_start_clip_after_cooldown_expires():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("dummy.mp4")]
    manager._loaded = True

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        manager.suppress_for(2.0)

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=12.1):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_called_once()


def test_clip_manager_does_not_start_new_clip_when_starts_are_blocked():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("dummy.mp4")]
    manager._loaded = True
    manager.set_clip_starts_blocked(True)

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=12.1):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_not_called()


def test_speaking_compatible_frame_ignores_non_blink_clip():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("single_slow_nod.mp4")]
    manager._loaded = True
    manager._schedules_initialized = True
    manager._current_role = manager._ROLE_SUPPLEMENTAL
    manager._current_cap = MagicMock()
    manager._current_cap.isOpened.return_value = True

    frame = manager.read_speaking_compatible_frame()

    assert frame is None
    manager._current_cap.read.assert_not_called()


def test_speaking_compatible_frame_can_start_due_blink_clip():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("blink_standard.mp4")]
    manager._loaded = True
    manager._refresh_role_inventory()
    manager._schedules_initialized = True
    manager._next_due_at[manager._ROLE_BLINK] = 5.0
    manager._next_due_at[manager._ROLE_BREATHING] = float("inf")
    manager._next_due_at[manager._ROLE_SUPPLEMENTAL] = float("inf")

    expected_frame = np.full((4, 4, 3), 99, dtype=np.uint8)

    def _open_blink_clip() -> bool:
        manager._current_role = manager._ROLE_BLINK
        manager._current_clip_overlay_capable = True
        manager._current_cap = MagicMock()
        manager._current_cap.isOpened.return_value = True
        manager._current_cap.read.return_value = (True, expected_frame)
        return True

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        with patch.object(manager, "_advance_clip", side_effect=_open_blink_clip):
            frame = manager.read_speaking_compatible_frame()

    assert np.array_equal(frame, expected_frame)
    assert manager.current_clip_allows_speaking_overlay is True


def test_clip_manager_blocks_all_idle_starts_while_speech_has_priority():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("blink_standard.mp4"), Path("single_slow_nod.mp4")]
    manager._loaded = True
    manager._refresh_role_inventory()
    manager._schedules_initialized = True
    manager._next_due_at[manager._ROLE_BLINK] = 5.0
    manager._next_due_at[manager._ROLE_BREATHING] = float("inf")
    manager._next_due_at[manager._ROLE_SUPPLEMENTAL] = 5.0
    manager.set_clip_starts_blocked(True)

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_not_called()


def test_clip_manager_avoids_immediate_repeat_across_playlist_rebuild():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    clip_a = Path("a.mp4")
    clip_b = Path("b.mp4")
    clip_c = Path("c.mp4")
    manager._clips = [clip_a, clip_b, clip_c]
    manager._loaded = True
    manager._last_played_clip = clip_a

    with patch("scrumsurvivor.idle.clip_manager.random.sample", return_value=[clip_a, clip_b, clip_c]):
        manager._build_playlist()

    assert manager._playlist == [clip_b, clip_c, clip_a]


def test_clip_manager_keeps_single_clip_playlist_unchanged():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    clip = Path("only.mp4")
    manager._clips = [clip]
    manager._loaded = True
    manager._last_played_clip = clip

    manager._build_playlist()

    assert manager._playlist == [clip]


def test_clip_manager_detects_named_role_clips_from_new_asset_names(tmp_path):
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    for file_name in (
        "blink_standard.mp4",
        "blink_slow_drowsy.mp4",
        "breathing_shift.mp4",
        "single_slow_nod.mp4",
    ):
        (tmp_path / file_name).write_bytes(b"stub")

    manager = IdleClipManager(str(tmp_path))

    assert manager.load() == 4
    assert manager.has_recorded_blink_clips is True
    assert manager.has_recorded_breathing_clips is True
    assert [clip.name for clip in manager._clips_by_role[manager._ROLE_SUPPLEMENTAL]] == [
        "single_slow_nod.mp4",
    ]


def test_clip_manager_prefers_due_blink_role_before_due_supplemental_role():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused")
    manager._clips = [Path("blink_standard.mp4"), Path("single_slow_nod.mp4")]
    manager._loaded = True
    manager._refresh_role_inventory()
    manager._schedules_initialized = True
    manager._next_due_at[manager._ROLE_BLINK] = 5.0
    manager._next_due_at[manager._ROLE_BREATHING] = float("inf")
    manager._next_due_at[manager._ROLE_SUPPLEMENTAL] = 6.0

    selected_roles: list[str | None] = []

    def _capture_role() -> bool:
        selected_roles.append(manager._pending_role)
        return False

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        with patch.object(manager, "_advance_clip", side_effect=_capture_role):
            frame = manager.read_frame()

    assert frame is None
    assert selected_roles == [manager._ROLE_BLINK]


def test_clip_manager_enforces_base_pause_after_any_clip_finishes():
    from scrumsurvivor.idle.clip_manager import IdleClipManager

    manager = IdleClipManager("unused", pause_min_s=2.0, pause_max_s=2.0)
    manager._clips = [Path("single_slow_nod.mp4")]
    manager._loaded = True
    manager._refresh_role_inventory()
    manager._schedules_initialized = True
    manager._current_role = manager._ROLE_SUPPLEMENTAL
    manager._current_cap = MagicMock()
    manager._current_cap.isOpened.return_value = True
    manager._current_cap.read.return_value = (False, None)

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=10.0):
        frame = manager.read_frame()

    assert frame is None

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=11.0):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_not_called()

    with patch("scrumsurvivor.idle.clip_manager.time.monotonic", return_value=12.1):
        with patch.object(manager, "_advance_clip", return_value=False) as mock_advance:
            frame = manager.read_frame()

    assert frame is None
    mock_advance.assert_called_once()