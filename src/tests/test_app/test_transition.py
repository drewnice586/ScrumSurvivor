"""Tests for CrossfadeTransition."""

from __future__ import annotations

import numpy as np
import pytest


def test_crossfade_is_done_after_n_frames():
    from scrumsurvivor.compositor.transition import CrossfadeTransition

    t = CrossfadeTransition(n_frames=3)
    from_f = np.zeros((4, 4, 3), dtype=np.uint8)
    to_f = np.full((4, 4, 3), 255, dtype=np.uint8)
    t.start(from_f, to_f)

    for _ in range(3):
        t.next_frame()

    assert t.is_done is True


def test_crossfade_blends_with_smoothstep_easing():
    from scrumsurvivor.compositor.transition import CrossfadeTransition

    t = CrossfadeTransition(n_frames=4)
    from_f = np.zeros((4, 4, 3), dtype=np.uint8)
    to_f = np.full((4, 4, 3), 200, dtype=np.uint8)
    t.start(from_f, to_f)

    frame1 = t.next_frame()
    # t = 1/4 = 0.25 -> smoothstep(t) = t^2 * (3 - 2t) = 0.15625
    # expected ~= 200 * 0.15625 = 31.25
    assert frame1[0, 0, 0] == pytest.approx(31.25, abs=2)


def test_crossfade_raises_before_start():
    from scrumsurvivor.compositor.transition import CrossfadeTransition

    t = CrossfadeTransition(n_frames=5)
    with pytest.raises(RuntimeError, match="start()"):
        t.next_frame()


def test_crossfade_output_shape():
    from scrumsurvivor.compositor.transition import CrossfadeTransition

    t = CrossfadeTransition(n_frames=2)
    f1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    f2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    t.start(f1, f2)
    result = t.next_frame()
    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8
