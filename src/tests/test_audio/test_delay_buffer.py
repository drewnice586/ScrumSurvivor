"""Tests for AudioDelayBuffer."""

from __future__ import annotations

import numpy as np
import pytest


def test_pull_returns_silence_before_primed():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    buf = AudioDelayBuffer(delay_ms=100, sample_rate=44100)
    result = buf.pull(512)
    assert np.all(result == 0.0)


def test_is_primed_false_initially():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    buf = AudioDelayBuffer(delay_ms=100, sample_rate=44100)
    assert buf.is_primed is False


def test_is_primed_true_after_enough_data():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    delay_ms = 10.0
    sample_rate = 44100
    buf = AudioDelayBuffer(delay_ms=delay_ms, sample_rate=sample_rate)
    delay_samples = int(delay_ms / 1000 * sample_rate)

    # Push exactly enough data
    chunk = np.zeros(delay_samples + 1, dtype=np.float32)
    buf.push(chunk)
    assert buf.is_primed is True


def test_delay_shifts_audio():
    """Data pushed in should appear in pull after the delay has elapsed."""
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    delay_ms = 10.0
    sr = 44100
    buf = AudioDelayBuffer(delay_ms=delay_ms, sample_rate=sr)
    delay_samples = int(delay_ms / 1000 * sr)
    chunk_size = 512

    # Fill the buffer enough to prime it
    silence = np.zeros(delay_samples, dtype=np.float32)
    buf.push(silence)

    # Now push a distinguish-able signal
    signal = np.ones(chunk_size, dtype=np.float32) * 0.5
    buf.push(signal)

    # A true delay line preserves both the configured startup delay and the
    # leading silence that was part of the source signal itself.
    pulled_startup_silence = buf.pull(delay_samples)
    assert np.allclose(pulled_startup_silence, 0.0)

    pulled_source_silence = buf.pull(delay_samples)
    assert np.allclose(pulled_source_silence, 0.0)

    # Pull signal portion once the delayed source reaches it.
    pulled_signal = buf.pull(chunk_size)
    assert np.allclose(pulled_signal, 0.5, atol=0.01)


def test_set_delay_resets_buffer():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    buf = AudioDelayBuffer(delay_ms=100, sample_rate=44100)
    buf.push(np.ones(4410, dtype=np.float32))
    assert buf.is_primed is True

    buf.set_delay(200)
    assert buf.is_primed is False


def test_pull_returns_float32():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    buf = AudioDelayBuffer(delay_ms=5, sample_rate=44100)
    result = buf.pull(512)
    assert result.dtype == np.float32


def test_pre_priming_pulls_do_not_destroy_future_delayed_audio():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    sr = 1000
    buf = AudioDelayBuffer(delay_ms=100, sample_rate=sr)

    # Consumer runs before enough audio exists yet.
    early = buf.pull(50)
    assert np.allclose(early, 0.0)

    # Push 150 ms of identifiable audio.
    signal = np.arange(150, dtype=np.float32)
    buf.push(signal)

    # The next 50 ms should still be silence because they map to negative time.
    still_silent = buf.pull(50)
    assert np.allclose(still_silent, 0.0)

    # Because the consumer already advanced 50 samples before any source audio
    # existed, the source timeline contains an explicit 50-sample silent gap
    # before the pushed signal begins.
    delayed = buf.pull(100)
    assert np.allclose(delayed[:50], 0.0)
    assert np.allclose(delayed[50:], np.arange(50, dtype=np.float32))

    remaining = buf.pull(100)
    assert np.allclose(remaining, np.arange(50, 150, dtype=np.float32))


def test_many_startup_pulls_do_not_leave_buffer_permanently_silent():
    from scrumsurvivor.audio.delay_buffer import AudioDelayBuffer

    sr = 44_100
    block = 512
    buf = AudioDelayBuffer(delay_ms=265, sample_rate=sr)

    # Simulate the output pump running during startup before the main loop
    # begins feeding audio into the delay buffer.
    startup_pulls = int((2.5 * sr) // block)
    for _ in range(startup_pulls):
        assert np.allclose(buf.pull(block), 0.0)

    # Feed constant non-zero audio for long enough to cross the configured delay.
    for _ in range(int((2.0 * sr) // block)):
        buf.push(np.full(block, 0.5, dtype=np.float32))

    heard = False
    for _ in range(int((3.0 * sr) // block)):
        if np.any(np.abs(buf.pull(block)) > 1e-6):
            heard = True
            break

    assert heard is True
