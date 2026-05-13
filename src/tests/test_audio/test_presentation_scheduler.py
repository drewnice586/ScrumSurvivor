from __future__ import annotations

import numpy as np


def test_base_delay_schedules_audio_after_fixed_offset() -> None:
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(base_delay_ms=100, sample_rate=1000)
    scheduler.push_chunk(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert np.allclose(scheduler.pull(100), 0.0)
    assert np.array_equal(
        scheduler.pull(3),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_hold_releases_blocked_audio_after_base_delay() -> None:
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(base_delay_ms=50, sample_rate=1000)
    scheduler.begin_hold()
    scheduler.push_chunk(np.array([0.25, 0.5], dtype=np.float32))

    assert np.allclose(scheduler.pull(200), 0.0)

    scheduler.release_hold()

    assert np.allclose(scheduler.pull(50), 0.0)
    assert np.array_equal(
        scheduler.pull(2),
        np.array([0.25, 0.5], dtype=np.float32),
    )


def test_hold_absorbs_future_audio_already_scheduled_at_base_delay() -> None:
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(base_delay_ms=50, sample_rate=1000)
    scheduler.push_chunk(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    assert np.allclose(scheduler.pull(25), 0.0)

    scheduler.begin_hold()

    assert np.allclose(scheduler.pull(100), 0.0)

    scheduler.release_hold()

    assert np.allclose(scheduler.pull(50), 0.0)
    assert np.array_equal(
        scheduler.pull(3),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_hold_does_not_reorder_audio_scheduled_before_block() -> None:
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(base_delay_ms=100, sample_rate=100)
    scheduler.push_chunk(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    first_pull = scheduler.pull(11)
    assert np.array_equal(
        first_pull,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    scheduler.begin_hold()
    scheduler.push_chunk(np.array([2.0, 2.0], dtype=np.float32))

    assert np.allclose(scheduler.pull(10), 0.0)

    scheduler.release_hold()

    assert np.allclose(scheduler.pull(10), 0.0)
    assert np.array_equal(
        scheduler.pull(4),
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    )


def test_recent_output_window_tracks_presented_audio() -> None:
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(base_delay_ms=200, sample_rate=10)
    scheduler.push_chunk(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    scheduler.pull(2)
    scheduler.pull(3)

    assert np.array_equal(
        scheduler.recent_output_window(5),
        np.array([0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_output_gated_to_silence_when_speech_detector_not_speaking() -> None:
    """When a speech detector is attached and says not speaking,
    pull_into should return silence and store silence in history."""
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    class _FakeDetector:
        def __init__(self) -> None:
            self.is_speaking = False

        def update(self, _chunk: np.ndarray) -> None:
            pass  # never flips to speaking

    detector = _FakeDetector()
    scheduler = AudioPresentationScheduler(
        base_delay_ms=0, sample_rate=100, speech_detector=detector,
    )
    scheduler.push_chunk(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    out = scheduler.pull(3)
    assert np.allclose(out, 0.0), "Output should be silence when detector says not speaking"

    history = scheduler.recent_output_window(3)
    assert np.allclose(history, 0.0), "History should also be silence (lip sync sees gated audio)"


def test_output_passes_through_when_speech_detector_speaking() -> None:
    """When the speech detector says speaking, audio passes through ungated."""
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    class _FakeDetector:
        def __init__(self) -> None:
            self.is_speaking = True

        def update(self, _chunk: np.ndarray) -> None:
            pass  # always speaking

    detector = _FakeDetector()
    scheduler = AudioPresentationScheduler(
        base_delay_ms=0, sample_rate=100, speech_detector=detector,
    )
    scheduler.push_chunk(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    out = scheduler.pull(3)
    assert np.array_equal(out, np.array([0.5, 0.5, 0.5], dtype=np.float32))

    history = scheduler.recent_output_window(3)
    assert np.array_equal(history, np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_output_gain_scales_audio() -> None:
    """output_gain reduces output amplitude (and history) proportionally."""
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(
        base_delay_ms=0, sample_rate=100, output_gain=0.5,
    )
    scheduler.push_chunk(np.array([1.0, 0.5, 0.2], dtype=np.float32))

    out = scheduler.pull(3)
    assert np.allclose(out, np.array([0.5, 0.25, 0.1], dtype=np.float32))

    history = scheduler.recent_output_window(3)
    assert np.allclose(history, np.array([0.5, 0.25, 0.1], dtype=np.float32))


def test_output_gain_default_passthrough() -> None:
    """Default gain (1.0) passes audio through unchanged."""
    from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler

    scheduler = AudioPresentationScheduler(
        base_delay_ms=0, sample_rate=100,
    )
    scheduler.push_chunk(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    out = scheduler.pull(3)
    assert np.array_equal(out, np.array([0.5, 0.5, 0.5], dtype=np.float32))