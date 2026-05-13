"""Tests for VirtualAudioOutput (non-hardware — mocked sounddevice)."""

from __future__ import annotations

import numpy as np
import pytest
import sounddevice as sd
from unittest.mock import MagicMock, patch


def test_write_raises_when_not_started():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    vao = VirtualAudioOutput()
    with pytest.raises(RuntimeError, match="not started"):
        vao.write(np.zeros(512, dtype=np.float32))


def test_start_opens_stream_with_correct_params():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [
        {"name": "CABLE Input", "max_output_channels": 2, "default_samplerate": 44_100},
        {"name": "Speakers", "max_output_channels": 2},
    ]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream) as mock_os,
        patch("sounddevice.query_devices", return_value=mock_devices),
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input", sample_rate=44100)
        vao.start()

        mock_os.assert_called_once()
        call_kwargs = mock_os.call_args.kwargs
        assert call_kwargs["samplerate"] == 44100
        assert call_kwargs["device"] == 0
        assert call_kwargs["blocksize"] == 0
        assert call_kwargs["latency"] == "high"
        vao.stop()


def test_start_prefers_standard_stereo_vb_cable_and_uses_device_rate():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 16, "default_samplerate": 44_100},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2, "default_samplerate": 48_000},
        {"name": "CABLE Input (VB-Audio Point)", "max_output_channels": 16, "default_samplerate": 44_100},
    ]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream) as mock_os,
        patch("sounddevice.query_devices", return_value=mock_devices),
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input", sample_rate=44_100)
        vao.start()

        call_kwargs = mock_os.call_args.kwargs
        # Selects the stereo device (index 1), not the 16-channel one
        assert call_kwargs["device"] == 1
        # Stream opened at device native rate
        assert call_kwargs["samplerate"] == 48_000
        vao.stop()


def test_start_honors_explicit_wasapi_selector():
    from scrumsurvivor.audio.device_selector import format_output_device_selector
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 16,
            "default_samplerate": 44_100,
            "hostapi": 0,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 16,
            "default_samplerate": 44_100,
            "hostapi": 1,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 2,
            "default_samplerate": 48_000,
            "hostapi": 2,
        },
    ]
    mock_hostapis = [
        {"name": "MME"},
        {"name": "Windows DirectSound"},
        {"name": "Windows WASAPI"},
    ]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream) as mock_os,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        vao = VirtualAudioOutput(
            device_name=format_output_device_selector(
                "CABLE Input (VB-Audio Virtual Cable)",
                hostapi_name="Windows WASAPI",
                device_id=2,
            )
        )
        vao.start()

        call_kwargs = mock_os.call_args.kwargs
        assert call_kwargs["device"] == 2
        assert call_kwargs["samplerate"] == 48_000
        vao.stop()


def test_start_prefers_wasapi_exclusive_when_supported():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 2,
            "default_samplerate": 48_000,
            "hostapi": 2,
        },
    ]
    mock_hostapis = [
        {"name": "MME"},
        {"name": "Windows DirectSound"},
        {"name": "Windows WASAPI"},
    ]
    exclusive_settings = object()

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream) as mock_os,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=exclusive_settings) as mock_wasapi,
        patch("sounddevice.check_output_settings") as mock_check,
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input")
        vao.start()

        mock_wasapi.assert_called_once_with(exclusive=True)
        mock_check.assert_called_once()
        call_kwargs = mock_os.call_args.kwargs
        assert call_kwargs["extra_settings"] is exclusive_settings
        assert call_kwargs["blocksize"] == 0
        assert call_kwargs["latency"] == "high"
        vao.stop()


def test_start_uses_callback_when_delay_buffer_attached_and_rates_match():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    class _DelayBuffer:
        def pull(self, num_samples):
            return np.full(num_samples, 0.25, dtype=np.float32)

        def pull_into(self, out):
            out[:] = 0.25

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 2,
            "default_samplerate": 48_000,
            "hostapi": 2,
        },
    ]
    mock_hostapis = [
        {"name": "MME"},
        {"name": "Windows DirectSound"},
        {"name": "Windows WASAPI"},
    ]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream) as mock_os,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input")
        vao.attach_delay_buffer(_DelayBuffer())
        vao.start()

        call_kwargs = mock_os.call_args.kwargs
        assert callable(call_kwargs["callback"])
        assert vao.is_callback_driven is True

        outdata = np.zeros((16, 1), dtype=np.float32)
        call_kwargs["callback"](outdata, 16, None, sd.CallbackFlags())
        assert np.allclose(outdata[:, 0], 0.25)
        vao.stop()


def test_list_output_devices():
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_devices = [
        {"name": "Mic", "max_output_channels": 0, "max_input_channels": 1},
        {"name": "Speakers", "max_output_channels": 2, "max_input_channels": 0},
        {"name": "CABLE Input", "max_output_channels": 2, "max_input_channels": 0},
    ]
    with patch("sounddevice.query_devices", return_value=mock_devices):
        devices = VirtualAudioOutput.list_output_devices()

    names = [d["name"] for d in devices]
    assert "Speakers" in names
    assert "CABLE Input" in names
    assert "Mic" not in names  # input-only excluded


def test_write_reshapes_mono():
    """write() should reshape 1-D mono array to (N, 1) for sounddevice."""
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [{"name": "CABLE Input", "max_output_channels": 2, "default_samplerate": 44_100}]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream),
        patch("sounddevice.query_devices", return_value=mock_devices),
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input")
        vao.start()

        chunk = np.ones(512, dtype=np.float32)
        vao.write(chunk)

        written = mock_stream.write.call_args[0][0]
        # Check shape after any implicit resampling if device rate differs from default
        # Default sample_rate is 48000, while mock device rate is 44100
        expected_len = round(512 * 44100 / 48000)
        assert written.shape == (expected_len, 1)
        vao.stop()


def test_write_resamples_when_device_uses_different_rate():
    """Overlap-save soxr resampler produces upsampled output."""
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    mock_stream = MagicMock()
    mock_devices = [
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2, "default_samplerate": 48_000},
    ]

    with (
        patch("sounddevice.OutputStream", return_value=mock_stream),
        patch("sounddevice.query_devices", return_value=mock_devices),
    ):
        vao = VirtualAudioOutput(device_name="CABLE Input", sample_rate=44_100)
        vao.start()

        chunk = np.linspace(-1.0, 1.0, num=512, dtype=np.float32)
        vao.write(chunk)

        written = mock_stream.write.call_args[0][0]
        assert written.shape[1] == 1
        # 512 samples at 44100 → ~557 at 48000
        assert written.shape[0] > 512
        vao.stop()


@pytest.mark.hardware
def test_virtual_audio_real_device():
    """Integration: open VB-Cable and write 0.1 s of silence."""
    from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

    with VirtualAudioOutput(device_name="CABLE Input") as vao:
        silence = np.zeros(4410, dtype=np.float32)
        vao.write(silence)
