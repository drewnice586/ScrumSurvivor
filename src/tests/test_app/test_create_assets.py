from __future__ import annotations

import numpy as np
from pathlib import Path
from unittest.mock import patch


def test_idle_clip_specs_use_named_files_for_new_workflow():
    from scrumsurvivor.create_assets import IDLE_CLIPS, _idle_clip_output_path

    expected_slugs = [
        "blink_standard",
        "blink_slow_drowsy",
        "breathing_shift",
        "single_slow_nod",
        "head_tilt_return",
        "glance_down_return",
    ]

    assert [spec.slug for spec in IDLE_CLIPS] == expected_slugs
    assert [_idle_clip_output_path(spec).name for spec in IDLE_CLIPS] == [
        f"{slug}.mp4" for slug in expected_slugs
    ]


def test_alignment_region_expands_face_rect_and_clamps_to_frame():
    from scrumsurvivor.create_assets import _alignment_region

    assert _alignment_region((720, 1280, 3), (600, 200, 100, 100)) == (460, 110, 840, 620)
    assert _alignment_region((720, 1280, 3), (10, 5, 80, 80)) == (0, 0, 202, 341)


def test_build_alignment_guide_draws_visible_outline_for_base_photo():
    from scrumsurvivor.create_assets import _build_alignment_guide

    base_photo = np.full((240, 320, 3), 255, dtype=np.uint8)
    base_photo[70:220, 110:210] = 40

    with patch(
        "scrumsurvivor.create_assets._load_face_detector",
        return_value=lambda _frame: (130, 70, 60, 60),
    ):
        guide = _build_alignment_guide(base_photo)

    assert guide.shape == base_photo.shape
    assert np.count_nonzero(guide) > 0


def test_build_camera_options_dedupes_helper_duplicates_and_preserves_capture_index():
    from scrumsurvivor.create_assets import _build_camera_options

    class _Cam:
        def __init__(self, index, name, path, backend=0):
            self.index = index
            self.name = name
            self.path = path
            self.backend = backend

    options = _build_camera_options([
        _Cam(1402, "Logitech BRIO", r"\\?\usb#vid_046d&pid_085e&mi_00#8&1d70ac2c&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global"),
        _Cam(702, "Logitech BRIO", r"\\?\usb#vid_046d&pid_085e&mi_00#8&1d70ac2c&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"),
        _Cam(703, "OBS Virtual Camera", ""),
    ])

    assert [(option.slot, option.capture_index, option.name) for option in options] == [
        (1, 702, "Logitech BRIO"),
        (2, 703, "OBS Virtual Camera"),
    ]


def test_detect_cameras_prefers_helper_and_falls_back_to_probe():
    from scrumsurvivor.create_assets import CameraOption, _detect_cameras

    helper_options = [
        CameraOption(slot=1, capture_index=702, name="Logitech BRIO"),
    ]

    with patch("scrumsurvivor.create_assets._enumerate_cameras_helper", return_value=[]):
        with patch("scrumsurvivor.create_assets._probe_camera_options", return_value=helper_options):
            assert _detect_cameras() == helper_options


def test_camera_identity_key_normalizes_helper_duplicate_paths():
    from scrumsurvivor.create_assets import _camera_identity_key

    left = _camera_identity_key(
        "Logitech BRIO",
        r"\\?\usb#vid_046d&pid_085e&mi_00#8&1d70ac2c&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global",
    )
    right = _camera_identity_key(
        "Logitech BRIO",
        r"\\?\usb#vid_046d&pid_085e&mi_00#8&1d70ac2c&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global",
    )

    assert left == right


def test_camera_selection_canvas_draws_camera_names_and_selection_panel():
    from scrumsurvivor.create_assets import CameraOption, _camera_selection_canvas

    canvas = _camera_selection_canvas(
        [
            CameraOption(1, 701, "Surface Camera Front"),
            CameraOption(2, 702, "Logitech BRIO"),
        ],
        selected_pos=1,
        width=640,
        height=360,
    )

    assert canvas.shape == (360, 640, 3)
    assert np.count_nonzero(canvas) > 0


def test_instruction_canvas_draws_step_guidance():
    from scrumsurvivor.create_assets import _instruction_canvas

    canvas = _instruction_canvas(
        "STEP 1/7 - Base Photo",
        [
            "Capture the reference frame.",
            "Look directly into the camera.",
            "Neutral expression - mouth firmly CLOSED.",
        ],
        width=640,
        height=360,
    )

    assert canvas.shape == (360, 640, 3)
    assert np.count_nonzero(canvas) > 0


def test_normalize_capture_frame_unmirrors_horizontally():
    from scrumsurvivor.create_assets import _normalize_capture_frame

    frame = np.array([
        [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
    ], dtype=np.uint8)

    normalized = _normalize_capture_frame(frame)

    assert normalized.tolist() == [[[3, 0, 0], [2, 0, 0], [1, 0, 0]]]


def test_action_key_accepts_enter_and_space():
    from scrumsurvivor.create_assets import _is_action_key

    assert _is_action_key(13) is True
    assert _is_action_key(10) is True
    assert _is_action_key(32) is True
    assert _is_action_key(ord("r")) is False


def test_selection_delta_supports_arrow_keys_and_wasd():
    from scrumsurvivor.create_assets import _selection_delta_for_key

    assert _selection_delta_for_key(0x260000) == -1
    assert _selection_delta_for_key(0x250000) == -1
    assert _selection_delta_for_key(ord("w")) == -1
    assert _selection_delta_for_key(0x280000) == 1
    assert _selection_delta_for_key(0x270000) == 1
    assert _selection_delta_for_key(ord("s")) == 1
    assert _selection_delta_for_key(ord("x")) == 0


def test_window_title_uses_ascii_text():
    from scrumsurvivor.create_assets import WINDOW

    assert WINDOW == "ScrumSurvivor - Asset Creator"
    assert WINDOW.encode("ascii")