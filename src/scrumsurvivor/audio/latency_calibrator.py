"""Latency calibrator — measures Wav2Lip inference time to recommend audio delay."""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_SAFETY_MARGIN_MS = 20.0


class LatencyCalibrator:
    """Measures Wav2Lip processing latency by running several dummy inferences.

    Args:
        wav2lip_engine: A loaded :class:`Wav2LipEngine` instance.
        face_crop: Representative face crop image (BGr, 96×96).
        sample_mel: Mel-spectrogram window ``(1, 80, 16)``.
        iterations: Number of inference runs to average (default 20).
    """

    def __init__(
        self,
        wav2lip_engine,
        face_crop: np.ndarray,
        sample_mel: np.ndarray,
        iterations: int = 20,
    ) -> None:
        self._engine = wav2lip_engine
        self._face = face_crop
        self._mel = sample_mel
        self._iterations = iterations
        self._stats: dict | None = None

    def calibrate(self) -> float:
        """Run inferences and return the recommended audio delay in milliseconds."""
        times_ms: list[float] = []
        for _ in range(self._iterations):
            t0 = time.perf_counter()
            self._engine.process(self._face, self._mel)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms)
        avg = float(arr.mean())
        recommended = avg + _SAFETY_MARGIN_MS

        self._stats = {
            "avg_ms": avg,
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "std_ms": float(arr.std()),
            "recommended_delay_ms": recommended,
        }
        logger.info(
            "Latency calibration: avg=%.1f ms, std=%.1f ms → recommended delay=%.1f ms",
            avg,
            arr.std(),
            recommended,
        )
        return recommended

    def get_stats(self) -> dict:
        """Return calibration statistics. Call :meth:`calibrate` first."""
        if self._stats is None:
            raise RuntimeError("calibrate() must be called before get_stats().")
        return self._stats
