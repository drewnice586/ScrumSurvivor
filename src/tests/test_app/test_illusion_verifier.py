"""Tests for the standalone illusion verifier helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_parse_audio_device_prefers_int_for_numeric_values():
    from illusion_verifier.devices import parse_audio_device

    assert parse_audio_device("4") == 4
    assert parse_audio_device(" CABLE Output ") == "CABLE Output"
    assert parse_audio_device("") is None


def test_pick_default_audio_input_prefers_cable_output():
    from illusion_verifier.devices import AudioDeviceInfo, pick_default_audio_input

    devices = [
        AudioDeviceInfo(id=1, name="Microphone", max_input_channels=1, default_samplerate=44_100),
        AudioDeviceInfo(id=9, name="CABLE Output (VB-Audio Virtual Cable)", max_input_channels=2, default_samplerate=44_100),
    ]

    assert pick_default_audio_input(devices) == 9


def test_pick_default_audio_input_prefers_standard_stereo_virtual_cable_over_ambiguous_matches():
    from illusion_verifier.devices import AudioDeviceInfo, pick_default_audio_input

    devices = [
        AudioDeviceInfo(id=1, name="CABLE Output (VB-Audio Virtual )", max_input_channels=16, default_samplerate=44_100),
        AudioDeviceInfo(id=29, name="CABLE Output (VB-Audio Virtual Cable)", max_input_channels=2, default_samplerate=48_000),
        AudioDeviceInfo(id=35, name="CABLE Output (VB-Audio Point)", max_input_channels=16, default_samplerate=44_100),
    ]

    assert pick_default_audio_input(devices) == 29


def test_build_recording_paths_sanitizes_label(tmp_path):
    from illusion_verifier.recorder import build_recording_paths

    paths = build_recording_paths(
        tmp_path,
        "Review Session!",
        now=datetime(2026, 4, 21, 12, 34, 56),
    )

    assert paths.video_path == tmp_path / "review_session_20260421_123456_video.mp4"
    assert paths.audio_path == tmp_path / "review_session_20260421_123456_audio.wav"
    assert paths.metadata_path == tmp_path / "review_session_20260421_123456_metadata.json"
    assert paths.review_path == tmp_path / "review_session_20260421_123456_review.mp4"


def test_resolve_ffmpeg_binary_prefers_repo_local_binary(tmp_path):
    from illusion_verifier.recorder import resolve_ffmpeg_binary

    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir()
    local_binary = ffmpeg_dir / "ffmpeg.exe"
    local_binary.write_text("stub", encoding="utf-8")

    assert resolve_ffmpeg_binary(search_root=tmp_path) == local_binary


def test_resolve_ffmpeg_binary_finds_workspace_root_when_package_lives_under_src(tmp_path):
    from illusion_verifier import recorder

    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir()
    local_binary = ffmpeg_dir / "ffmpeg.exe"
    local_binary.write_text("stub", encoding="utf-8")

    fake_module_file = tmp_path / "src" / "illusion_verifier" / "recorder.py"
    fake_module_file.parent.mkdir(parents=True)
    fake_module_file.write_text("# stub", encoding="utf-8")

    with patch.object(recorder, "__file__", str(fake_module_file)):
        assert recorder.resolve_ffmpeg_binary() == local_binary


def test_resolve_audio_input_config_uses_device_default_sample_rate():
    from illusion_verifier.recorder import resolve_audio_input_config

    with patch(
        "illusion_verifier.recorder.sd.query_devices",
        return_value={"name": "CABLE Output (VB-Audio Virtual Cable)", "default_samplerate": 48_000.0},
    ) as mock_query:
        resolved_device, sample_rate, name = resolve_audio_input_config(29)

    mock_query.assert_called_once_with(29, "input")
    assert resolved_device == 29
    assert sample_rate == 48_000
    assert name == "CABLE Output (VB-Audio Virtual Cable)"


def test_target_frame_count_matches_constant_rate_timeline():
    from illusion_verifier.recorder import _target_frame_count

    assert _target_frame_count(0.0, 25) == 1
    assert _target_frame_count(0.039, 25) == 1
    assert _target_frame_count(0.040, 25) == 2
    assert _target_frame_count(0.999, 25) == 25


def test_align_audio_to_video_timeline_can_pad_and_trim():
    from illusion_verifier.recorder import align_audio_to_video_timeline
    import numpy as np

    sample_rate = 10
    audio = np.arange(10, dtype=np.float32)

    padded = align_audio_to_video_timeline(
        audio=audio,
        audio_start_s=0.2,
        video_start_s=0.0,
        target_duration_s=1.2,
        sample_rate=sample_rate,
    )
    assert len(padded) == 12
    assert np.allclose(padded[:2], 0.0)
    assert np.allclose(padded[2:], audio)

    trimmed = align_audio_to_video_timeline(
        audio=audio,
        audio_start_s=0.0,
        video_start_s=0.3,
        target_duration_s=0.5,
        sample_rate=sample_rate,
    )
    assert len(trimmed) == 5
    assert np.allclose(trimmed, np.array([3, 4, 5, 6, 7], dtype=np.float32))


def test_stop_hotkey_registers_normalized_listener():
    from illusion_verifier.recorder import StopHotkey

    callback = MagicMock()
    mock_listener = MagicMock()

    with patch("pynput.keyboard.GlobalHotKeys", return_value=mock_listener) as mock_listener_cls:
        hotkey = StopHotkey("ctrl+shift+f10", callback)
        hotkey.start()
        hotkey.stop()

    registrations = mock_listener_cls.call_args.args[0]
    assert "<ctrl>+<shift>+<f10>" in registrations
    mock_listener.stop.assert_called_once()


def test_cli_defaults_to_hotkey_stopped_recording(tmp_path):
    import cv2
    from click.testing import CliRunner

    from illusion_verifier.devices import AudioDeviceInfo, VideoDeviceInfo
    from illusion_verifier.main import cli
    from illusion_verifier.recorder import RecordingResult

    result_value = RecordingResult(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.wav",
        metadata_path=tmp_path / "metadata.json",
        review_path=None,
        ffmpeg_used=False,
        frame_count=10,
        encoded_frame_count=10,
        audio_samples=44_100,
        video_duration_s=1.0,
        encoded_video_duration_s=1.0,
        audio_duration_s=1.0,
        effective_capture_fps=25.0,
        raw_audio_start_offset_ms=0.0,
        raw_av_duration_gap_ms=0.0,
        audio_sample_rate=48_000,
    )

    with patch(
        "illusion_verifier.main.find_video_devices",
        return_value=[
            VideoDeviceInfo(
                slot=1,
                capture_index=2,
                name="Webcam",
                backend=cv2.CAP_DSHOW,
                width=1280,
                height=720,
            )
        ],
    ):
        with patch(
            "illusion_verifier.main.list_audio_input_devices",
            return_value=[
                AudioDeviceInfo(
                    id=9,
                    name="CABLE Output (VB-Audio Virtual Cable)",
                    max_input_channels=2,
                    default_samplerate=44_100,
                )
            ],
        ):
            with patch("illusion_verifier.main.record_session", return_value=result_value) as mock_record:
                result = CliRunner().invoke(
                    cli,
                    [
                        "--video-device",
                        "2",
                        "--audio-device",
                        "9",
                        "--output-dir",
                        str(tmp_path),
                        "--no-preview",
                    ],
                )

    assert result.exit_code == 0
    assert "Recording until you press ctrl+shift+f10 or Ctrl+C in the terminal" in result.output
    assert mock_record.call_args.kwargs["duration_s"] is None
    assert mock_record.call_args.kwargs["stop_hotkey"] == "ctrl+shift+f10"
    assert mock_record.call_args.kwargs["video_backend"] == cv2.CAP_DSHOW


def test_cli_prompts_for_video_before_audio_and_shows_camera_names(tmp_path):
    import cv2
    from click.testing import CliRunner

    from illusion_verifier.devices import AudioDeviceInfo, VideoDeviceInfo
    from illusion_verifier.main import cli
    from illusion_verifier.recorder import RecordingResult

    result_value = RecordingResult(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.wav",
        metadata_path=tmp_path / "metadata.json",
        review_path=None,
        ffmpeg_used=False,
        frame_count=10,
        encoded_frame_count=10,
        audio_samples=44_100,
        video_duration_s=1.0,
        encoded_video_duration_s=1.0,
        audio_duration_s=1.0,
        effective_capture_fps=25.0,
        raw_audio_start_offset_ms=0.0,
        raw_av_duration_gap_ms=0.0,
        audio_sample_rate=48_000,
    )

    with patch(
        "illusion_verifier.main.find_video_devices",
        return_value=[
            VideoDeviceInfo(
                slot=1,
                capture_index=700,
                name="Webcam",
                backend=cv2.CAP_MSMF,
            ),
            VideoDeviceInfo(
                slot=2,
                capture_index=701,
                name="Surface Camera Front",
                backend=cv2.CAP_MSMF,
            ),
        ],
    ):
        with patch(
            "illusion_verifier.main.list_audio_input_devices",
            return_value=[
                AudioDeviceInfo(
                    id=29,
                    name="CABLE Output (VB-Audio Virtual Cable)",
                    max_input_channels=2,
                    default_samplerate=48_000,
                )
            ],
        ):
            with patch("illusion_verifier.main.record_session", return_value=result_value) as mock_record:
                result = CliRunner().invoke(
                    cli,
                    [
                        "--output-dir",
                        str(tmp_path),
                        "--no-preview",
                    ],
                    input="\n\n",
                )

    assert result.exit_code == 0
    assert "Available cameras:" in result.output
    assert "[1]  Webcam (OpenCV index 700)" in result.output
    assert result.output.index("Available cameras:") < result.output.index("Select video device")
    assert result.output.index("Select video device") < result.output.index("Audio input devices:")
    assert mock_record.call_args.kwargs["video_device"] == 700
    assert mock_record.call_args.kwargs["video_backend"] == cv2.CAP_MSMF


def test_mux_with_ffmpeg_returns_false_when_ffmpeg_missing(tmp_path):
    from illusion_verifier.recorder import mux_with_ffmpeg

    with patch("illusion_verifier.recorder.resolve_ffmpeg_binary", return_value=None):
        ok, detail = mux_with_ffmpeg(
            tmp_path / "video.mp4",
            tmp_path / "audio.wav",
            tmp_path / "review.mp4",
        )

    assert ok is False
    assert "ffmpeg" in detail.lower()


def test_mux_with_ffmpeg_invokes_subprocess(tmp_path):
    from illusion_verifier.recorder import mux_with_ffmpeg

    class _Result:
        returncode = 0
        stderr = ""

    with patch("illusion_verifier.recorder.resolve_ffmpeg_binary", return_value=Path("C:/ffmpeg/bin/ffmpeg.exe")):
        with patch("illusion_verifier.recorder.subprocess.run", return_value=_Result()) as mock_run:
            ok, detail = mux_with_ffmpeg(
                tmp_path / "video.mp4",
                tmp_path / "audio.wav",
                tmp_path / "review.mp4",
            )

    assert ok is True
    assert detail == "ok"
    cmd = mock_run.call_args.args[0]
    assert cmd[0].endswith("ffmpeg.exe")
    assert "-shortest" in cmd