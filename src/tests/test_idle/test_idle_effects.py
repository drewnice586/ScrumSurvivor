"""Tests for Phase 2 idle animation effects."""

from __future__ import annotations

import math
import numpy as np
import pytest


# ── BreathingEffect ───────────────────────────────────────────────────────────

def test_breathing_zero_at_start():
    from scrumsurvivor.idle.breathing import BreathingEffect

    effect = BreathingEffect(rate_hz=0.25, amplitude_px=2.5)
    # At t == start, sin(0) = 0
    assert effect.offset_at(t=effect._start) == pytest.approx(0.0)


def test_breathing_max_at_quarter_period():
    from scrumsurvivor.idle.breathing import BreathingEffect

    rate = 0.25
    amp = 5.0
    effect = BreathingEffect(rate_hz=rate, amplitude_px=amp)
    t_quarter = effect._start + 1.0 / (4 * rate)
    assert effect.offset_at(t=t_quarter) == pytest.approx(amp, abs=0.01)


def test_breathing_apply_returns_same_shape():
    from scrumsurvivor.idle.breathing import BreathingEffect

    effect = BreathingEffect()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = effect.apply(frame)
    assert result.shape == frame.shape


# ── HeadSwayEffect ────────────────────────────────────────────────────────────

def test_head_sway_zero_at_start():
    from scrumsurvivor.idle.head_sway import HeadSwayEffect

    effect = HeadSwayEffect()
    dx, dy = effect.offsets_at(t=effect._start)
    assert dx == pytest.approx(0.0, abs=0.01)
    assert dy == pytest.approx(0.0, abs=0.01)


def test_head_sway_apply_returns_same_shape():
    from scrumsurvivor.idle.head_sway import HeadSwayEffect

    effect = HeadSwayEffect()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = effect.apply(frame)
    assert result.shape == frame.shape


# ── NoiseEffect ───────────────────────────────────────────────────────────────

def test_noise_zero_intensity_returns_same():
    from scrumsurvivor.idle.noise import NoiseEffect

    effect = NoiseEffect(intensity=0)
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    result = effect.apply(frame)
    assert np.array_equal(result, frame)


def test_noise_varies_pixels():
    from scrumsurvivor.idle.noise import NoiseEffect

    effect = NoiseEffect(intensity=10)
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = effect.apply(frame)
    # Very unlikely that all pixels are exactly 128 after noise
    assert not np.array_equal(result, frame)


def test_noise_stays_in_bounds():
    from scrumsurvivor.idle.noise import NoiseEffect

    effect = NoiseEffect(intensity=50)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = effect.apply(frame)
    assert result.min() >= 0
    assert result.max() <= 255


# ── BlinkEffect ───────────────────────────────────────────────────────────────

def test_blink_apply_returns_same_shape():
    from scrumsurvivor.idle.blink import BlinkEffect

    effect = BlinkEffect(interval_range=(3.0, 6.0))
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    result = effect.apply(frame)
    assert result.shape == frame.shape


# ── IdleCompositor ─────────────────────────────────────────────────────────────

def test_idle_compositor_uses_base_when_no_clips():
    from scrumsurvivor.idle.idle_compositor import IdleCompositor

    base = np.full((720, 1280, 3), 42, dtype=np.uint8)
    compositor = IdleCompositor(base_image=base)

    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = compositor.process(raw)
    assert result.shape == (720, 1280, 3)
    # Without effects applied, base content should be mostly preserved
    assert np.array_equal(result, base)


def test_idle_compositor_applies_noise():
    from scrumsurvivor.idle.idle_compositor import IdleCompositor
    from scrumsurvivor.idle.noise import NoiseEffect

    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    noise = NoiseEffect(intensity=10)
    compositor = IdleCompositor(base_image=base, noise=noise)

    raw = np.zeros((100, 100, 3), dtype=np.uint8)
    result = compositor.process(raw)
    assert not np.array_equal(result, base)


def test_idle_compositor_skips_fallback_blink_and_breathing_when_recorded_clips_exist():
    from scrumsurvivor.idle.idle_compositor import IdleCompositor

    class _Effect:
        def __init__(self):
            self.called = False

        def apply(self, frame):
            self.called = True
            return frame + 1

    class _ClipManager:
        has_clips = True
        has_recorded_blink_clips = True
        has_recorded_breathing_clips = True
        is_clip_playing = False

        def read_frame(self):
            return None

    base = np.zeros((20, 20, 3), dtype=np.uint8)
    breathing = _Effect()
    blink = _Effect()
    compositor = IdleCompositor(
        base_image=base,
        clip_manager=_ClipManager(),
        breathing=breathing,
        blink=blink,
    )

    result = compositor.process(base)

    assert np.array_equal(result, base)
    assert breathing.called is False
    assert blink.called is False


def test_idle_compositor_crossfades_from_base_into_clip_frames():
    from scrumsurvivor.idle.idle_compositor import IdleCompositor

    class _ClipManager:
        has_clips = True
        has_recorded_blink_clips = False
        has_recorded_breathing_clips = False
        is_clip_playing = True

        def __init__(self):
            self._frames = [None, np.full((10, 10, 3), 200, dtype=np.uint8)]

        def read_frame(self):
            if self._frames:
                return self._frames.pop(0)
            return np.full((10, 10, 3), 200, dtype=np.uint8)

    base = np.zeros((10, 10, 3), dtype=np.uint8)
    compositor = IdleCompositor(
        base_image=base,
        clip_manager=_ClipManager(),
        transition_frames=4,
    )

    first = compositor.process(base)
    second = compositor.process(base)

    assert np.array_equal(first, base)
    assert second[0, 0, 0] > 0
    assert second[0, 0, 0] < 200


def test_idle_compositor_crossfades_from_clip_back_to_base():
    from scrumsurvivor.idle.idle_compositor import IdleCompositor

    clip_frame = np.full((10, 10, 3), 180, dtype=np.uint8)

    class _ClipManager:
        has_clips = True
        has_recorded_blink_clips = False
        has_recorded_breathing_clips = False
        is_clip_playing = False

        def __init__(self):
            self._frames = [clip_frame, None]

        def read_frame(self):
            if self._frames:
                return self._frames.pop(0)
            return None

    base = np.zeros((10, 10, 3), dtype=np.uint8)
    compositor = IdleCompositor(
        base_image=base,
        clip_manager=_ClipManager(),
        transition_frames=4,
    )

    first = compositor.process(base)
    second = compositor.process(base)

    assert np.array_equal(first, clip_frame)
    assert second[0, 0, 0] > 0
    assert second[0, 0, 0] < 180
