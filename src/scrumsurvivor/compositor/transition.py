"""Crossfade transition between video frames."""

from __future__ import annotations

import numpy as np


class CrossfadeTransition:
    """Produces a smoothstep crossfade blend between two frames over *n_frames* frames.

    Usage::

        t = CrossfadeTransition(n_frames=5)
        t.start(current_frame, target_frame)
        while not t.is_done:
            output = t.next_frame()

    The transition is stateful — only one transition can run at a time.
    """

    def __init__(self, n_frames: int = 5) -> None:
        self._n_frames = max(1, n_frames)
        self._from_frame: np.ndarray | None = None
        self._to_frame: np.ndarray | None = None
        self._step = 0

    def start(self, from_frame: np.ndarray, to_frame: np.ndarray) -> None:
        """Begin a new crossfade from *from_frame* to *to_frame*."""
        self._from_frame = from_frame.astype(np.float32)
        self._to_frame = to_frame.astype(np.float32)
        self._step = 0

    @property
    def is_done(self) -> bool:
        """True once all crossfade frames have been emitted."""
        return self._step >= self._n_frames

    @property
    def is_active(self) -> bool:
        return self._from_frame is not None and not self.is_done

    def next_frame(self) -> np.ndarray:
        """Return the next blended frame and advance the step counter.

        Raises RuntimeError if no transition is active.
        """
        if self._from_frame is None or self._to_frame is None:
            raise RuntimeError("Call start() before next_frame().")
        alpha = (self._step + 1) / self._n_frames
        # Smoothstep easing: ease-in/out to avoid visible hard clipping at state switches.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        blended = self._from_frame * (1.0 - alpha) + self._to_frame * alpha
        self._step += 1
        if self.is_done:
            self._from_frame = None
            self._to_frame = None
        return blended.astype(np.uint8)
