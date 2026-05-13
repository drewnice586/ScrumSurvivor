"""Sync verifier — manual A/V sync check tool."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SYNC_THRESHOLD_MS = 50.0  # A/V considered in sync if offset < this


class SyncVerifier:
    """Manual sync verification tool invoked via ``calibrate --verify-sync``.

    Measures the time offset between audio onset (when the delayed audio
    starts) and video onset (when the lip-animation starts on the virtual
    camera feed).

    Args:
        audio_delay_ms: The currently configured audio delay.
        wav2lip_avg_latency_ms: The measured Wav2Lip processing latency.
    """

    def __init__(
        self,
        audio_delay_ms: float,
        wav2lip_avg_latency_ms: float,
    ) -> None:
        self._audio_delay_ms = audio_delay_ms
        self._wav2lip_latency_ms = wav2lip_avg_latency_ms

    def run_test(self, duration_s: float = 5.0) -> dict:
        """Estimate A/V sync quality from configured delay values.

        In Phase 4 this is a heuristic check: if audio delay ≈ Wav2Lip
        latency, sync is considered good.  Full frame-accurate verification
        requires capturing virtual camera output which is done manually.

        Returns:
            dict with keys ``offset_ms``, ``in_sync``, and ``recommendation``.
        """
        offset_ms = self._audio_delay_ms - self._wav2lip_latency_ms
        in_sync = abs(offset_ms) < _SYNC_THRESHOLD_MS

        recommendation = None
        if not in_sync:
            if offset_ms > 0:
                recommendation = (
                    f"Audio is {offset_ms:.0f} ms too late relative to video. "
                    f"Try reducing audio_delay_ms by {offset_ms:.0f} in config.yaml"
                )
            else:
                recommendation = (
                    f"Audio is {abs(offset_ms):.0f} ms too early relative to video. "
                    f"Try increasing audio_delay_ms by {abs(offset_ms):.0f} in config.yaml"
                )

        result = {
            "audio_delay_ms": self._audio_delay_ms,
            "wav2lip_latency_ms": self._wav2lip_latency_ms,
            "offset_ms": offset_ms,
            "in_sync": in_sync,
            "recommendation": recommendation,
        }

        logger.info(
            "Sync check: audio_delay=%.1fms wav2lip_latency=%.1fms offset=%.1fms in_sync=%s",
            self._audio_delay_ms,
            self._wav2lip_latency_ms,
            offset_ms,
            in_sync,
        )
        return result
