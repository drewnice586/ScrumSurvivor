"""Tests for DuplexAudioTransport (non-hardware)."""

from __future__ import annotations

import numpy as np
import pytest
import sounddevice as sd
from unittest.mock import MagicMock, patch


def test_start_opens_duplex_stream_when_hostapis_match():
    from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "Microphone (Logitech BRIO)",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 2,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
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
        patch("sounddevice.Stream", return_value=mock_stream) as mock_sd_stream,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=exclusive_settings),
        patch("sounddevice.check_output_settings"),
    ):
        transport = DuplexAudioTransport(
            input_device="Logitech BRIO",
            output_device_name="CABLE Input",
        )
        transport.start()

        call_kwargs = mock_sd_stream.call_args.kwargs
        assert call_kwargs["device"] == (0, 1)
        assert call_kwargs["samplerate"] == 48_000
        assert call_kwargs["blocksize"] == 512
        assert call_kwargs["latency"] == ("high", "high")
        assert call_kwargs["extra_settings"] == (None, exclusive_settings)
        transport.stop()


def test_start_rejects_hostapi_mismatch():
    from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport

    mock_devices = [
        {
            "name": "Different Microphone",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 0,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
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
        patch("sounddevice.Stream") as mock_sd_stream,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        transport = DuplexAudioTransport(input_device=0, output_device_name="CABLE Input")
        with pytest.raises(RuntimeError, match="same host API"):
            transport.start()

    mock_sd_stream.assert_not_called()


def test_start_remaps_integer_input_to_matching_hostapi_duplicate():
    from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "Microphone (Logitech BRIO)",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 0,
        },
        {
            "name": "Microphone (Logitech BRIO)",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 2,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
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
        patch("sounddevice.Stream", return_value=mock_stream) as mock_sd_stream,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        transport = DuplexAudioTransport(input_device=0, output_device_name="CABLE Input")
        transport.start()

        call_kwargs = mock_sd_stream.call_args.kwargs
        assert call_kwargs["device"] == (1, 2)
        transport.stop()


def test_callback_enqueues_input_and_pulls_output():
    from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport

    class _DelayBuffer:
        def pull_into(self, out: np.ndarray) -> None:
            out[:] = 0.25

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "Microphone (Logitech BRIO)",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 2,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
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
        patch("sounddevice.Stream", return_value=mock_stream) as mock_sd_stream,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        transport = DuplexAudioTransport(input_device=0, output_device_name="CABLE Input")
        transport.attach_delay_buffer(_DelayBuffer())
        transport.start()

        callback = mock_sd_stream.call_args.kwargs["callback"]
        indata = np.arange(4, dtype=np.float32).reshape(-1, 1)
        outdata = np.zeros((4, 1), dtype=np.float32)
        callback(indata, outdata, 4, None, sd.CallbackFlags())

        chunk = transport.read(block=False)
        assert np.array_equal(chunk, np.arange(4, dtype=np.float32))
        assert np.allclose(outdata[:, 0], 0.25)
        transport.stop()


def test_start_honors_explicit_wasapi_output_selector():
    from scrumsurvivor.audio.device_selector import format_output_device_selector
    from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport

    mock_stream = MagicMock()
    mock_devices = [
        {
            "name": "Microphone (Logitech BRIO)",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "hostapi": 2,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
            "max_output_channels": 16,
            "default_samplerate": 44_100,
            "hostapi": 0,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
            "max_output_channels": 16,
            "default_samplerate": 44_100,
            "hostapi": 1,
        },
        {
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_input_channels": 0,
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
        patch("sounddevice.Stream", return_value=mock_stream) as mock_sd_stream,
        patch("sounddevice.query_devices", return_value=mock_devices),
        patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        patch("sounddevice.WasapiSettings", return_value=object()),
        patch("sounddevice.check_output_settings"),
    ):
        transport = DuplexAudioTransport(
            input_device="Logitech BRIO",
            output_device_name=format_output_device_selector(
                "CABLE Input (VB-Audio Virtual Cable)",
                hostapi_name="Windows WASAPI",
                device_id=3,
            ),
        )
        transport.start()

        call_kwargs = mock_sd_stream.call_args.kwargs
        assert call_kwargs["device"] == (0, 3)
        assert call_kwargs["samplerate"] == 48_000
        transport.stop()