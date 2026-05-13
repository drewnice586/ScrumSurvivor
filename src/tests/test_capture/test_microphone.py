"""Tests for MicrophoneCapture (non-hardware)."""

from __future__ import annotations

import numpy as np
import pytest


def test_rms_silence():
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    silent = np.zeros(512, dtype=np.float32)
    assert MicrophoneCapture.rms(silent) == pytest.approx(0.0)


def test_rms_full_amplitude():
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    full = np.ones(512, dtype=np.float32)
    assert MicrophoneCapture.rms(full) == pytest.approx(1.0)


def test_rms_half_amplitude():
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    half = np.full(512, 0.5, dtype=np.float32)
    assert MicrophoneCapture.rms(half) == pytest.approx(0.5)


def test_read_returns_none_when_empty():
    """read(block=False) returns None on an empty queue without hardware."""
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    mic = MicrophoneCapture()  # not opened
    result = mic.read(block=False)
    assert result is None


def test_read_all_returns_empty_list():
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    mic = MicrophoneCapture()
    assert mic.read_all() == []


@pytest.mark.hardware
def test_microphone_opens_and_captures():
    """Integration: opens the default mic and reads a chunk."""
    import time
    from scrumsurvivor.capture.microphone import MicrophoneCapture

    with MicrophoneCapture() as mic:
        time.sleep(0.05)
        chunk = mic.read(block=False)
        assert chunk is not None
        assert chunk.dtype == np.float32
        assert len(chunk) == 512
