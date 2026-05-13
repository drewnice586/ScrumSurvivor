"""Noise effect — subtle pixel-level grain to simulate webcam sensor noise."""

from __future__ import annotations

import numpy as np


class NoiseEffect:
    """Adds uniform random noise in the range ``[-intensity, +intensity]``
    to each pixel channel.

    Args:
        intensity: Half-range of noise in pixel values (integer).
    """

    def __init__(self, intensity: int = 3) -> None:
        self._intensity = intensity

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return *frame* with per-pixel noise applied."""
        if self._intensity == 0:
            return frame
        noise = np.random.randint(
            -self._intensity,
            self._intensity + 1,
            size=frame.shape,
            dtype=np.int16,
        )
        noisy = frame.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
