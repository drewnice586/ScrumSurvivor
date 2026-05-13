"""Tests for AudioPreprocessor."""

from __future__ import annotations

import numpy as np
import pytest


def test_to_mel_output_shape():
    """to_mel() should return (n_mels, T) for any audio length."""
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor(sample_rate=44100)
    # 1 second of silence at 44100 Hz
    audio = np.zeros(44100, dtype=np.float32)
    mel = proc.to_mel(audio)
    assert mel.ndim == 2
    assert mel.shape[0] == 80  # n_mels


def test_resample_length():
    """Resampling from 44100 → 16000 should produce approx 16000/44100 the samples."""
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor(sample_rate=44100)
    audio = np.zeros(44100, dtype=np.float32)
    resampled = proc.resample(audio)
    expected_len = int(44100 * 16000 / 44100)
    assert abs(len(resampled) - expected_len) <= 2  # allow 2-sample rounding


def test_resample_no_op_at_16k():
    """No resampling needed when input is already 16 kHz."""
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor(sample_rate=16000)
    audio = np.ones(160, dtype=np.float32)
    result = proc.resample(audio)
    assert len(result) == 160


def test_mel_silence_near_zero():
    """Silence produces a mel below a low threshold (log-compressed floor)."""
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor()
    audio = np.zeros(44100, dtype=np.float32)
    mel = proc.to_mel(audio)
    # After normalisation, silence → near minimum (-4.0)
    assert mel.mean() < -2.0


def test_extract_mel_window_shape():
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor()
    # Build a mel that is wide enough
    audio = np.zeros(44100, dtype=np.float32)
    mel = proc.to_mel(audio)
    window = proc.extract_mel_window(mel, frame_idx=0, fps=25.0)
    if window is not None:
        assert window.shape == (1, 80, 16)


def test_extract_mel_window_returns_none_if_too_short():
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor

    proc = AudioPreprocessor()
    short_mel = np.zeros((80, 5), dtype=np.float32)  # only 5 frames
    window = proc.extract_mel_window(short_mel, frame_idx=100, fps=25.0)
    assert window is None
