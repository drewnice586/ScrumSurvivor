"""Speech detector with self-calibrating RMS threshold."""

from __future__ import annotations

import logging
import time

import numpy as np

from scrumsurvivor.capture.microphone import MicrophoneCapture

logger = logging.getLogger(__name__)

# Calibration parameters
_CALIBRATION_DURATION_S = 2.5    # seconds of ambient noise to measure
_CALIBRATION_MULTIPLIER = 2.0    # threshold = mean + multiplier * std


class SpeechDetector:
    """Detects speech using an RMS-based threshold with hysteresis.

    Attributes
    ----------
    threshold:
        The RMS value above which audio is considered speech.
        Set by ``calibrate()`` or via the config ``speech_threshold``.
    is_speaking:
        Current speech state (True = speaking, False = silent).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold: float | None = None,
        attack_ms: int = 80,
        release_ms: int = 300,
        chunk_frames: int = 512,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_frames = chunk_frames
        self._attack_ms = attack_ms
        self._release_ms = release_ms

        self.threshold: float = threshold if threshold is not None else 0.02
        self._threshold_from_config: float | None = threshold

        # Hysteresis state
        self.is_speaking: bool = False
        self._state_entered_at: float = time.monotonic()

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, mic: MicrophoneCapture) -> float:
        """Measure ambient noise from *mic* for ~2.5 s and set threshold.

        Returns the computed threshold value.
        """
        if self._threshold_from_config is not None:
            logger.info(
                "speech_threshold set in config (%s) — skipping calibration.",
                self._threshold_from_config,
            )
            self.threshold = self._threshold_from_config
            return self.threshold

        logger.info(
            "Calibrating ambient noise level (%.1f s)…  please stay quiet.",
            _CALIBRATION_DURATION_S,
        )
        chunks_needed = int(
            _CALIBRATION_DURATION_S * self._sample_rate / self._chunk_frames
        )
        rms_values: list[float] = []
        deadline = time.monotonic() + _CALIBRATION_DURATION_S + 0.5  # safety margin

        while len(rms_values) < chunks_needed and time.monotonic() < deadline:
            chunk = mic.read(block=True)
            if chunk is not None:
                rms_values.append(MicrophoneCapture.rms(chunk))

        if len(rms_values) < 5:
            logger.warning(
                "Calibration got only %d samples — using default threshold 0.02",
                len(rms_values),
            )
            self.threshold = 0.02
            return self.threshold

        arr = np.array(rms_values)
        computed = float(arr.mean() + _CALIBRATION_MULTIPLIER * arr.std())
        # Floor: never let threshold fall below 0.005 — if the calibration mic
        # is silent (e.g. wrong device / VB-Cable), a zero threshold would
        # cause the pipeline to always be in SPEAKING state.
        _MIN_THRESHOLD = 0.005
        if computed < _MIN_THRESHOLD:
            logger.warning(
                "Calibrated threshold %.4f is below minimum %.4f — "
                "mic may be wrong device (got: mean=%.4f std=%.4f). "
                "Using minimum threshold. Check microphone_device in config.yaml.",
                computed,
                _MIN_THRESHOLD,
                arr.mean(),
                arr.std(),
            )
            computed = _MIN_THRESHOLD
        self.threshold = computed
        logger.info(
            "Calibration complete: mean_rms=%.4f  std=%.4f  threshold=%.4f",
            arr.mean(),
            arr.std(),
            self.threshold,
        )
        return self.threshold

    # ── Real-time update ──────────────────────────────────────────────────────

    def update(self, chunk: np.ndarray) -> bool:
        """Process one audio *chunk* and update :attr:`is_speaking`.

        Returns True if the state changed this call.
        """
        rms = MicrophoneCapture.rms(chunk)
        now = time.monotonic()

        if not self.is_speaking:
            if rms >= self.threshold:
                # check if we have been above threshold long enough
                # (attack hysteresis: we switch to speaking after attack_ms)
                # We track time above threshold via _potential_onset
                if not hasattr(self, "_onset_start") or self._onset_start is None:  # type: ignore[attr-defined]
                    self._onset_start = now  # type: ignore[attr-defined]
                elapsed_ms = (now - self._onset_start) * 1000  # type: ignore[attr-defined]
                if elapsed_ms >= self._attack_ms:
                    self.is_speaking = True
                    self._state_entered_at = now
                    self._onset_start = None  # type: ignore[attr-defined]
                    logger.debug("Speech STARTED (rms=%.4f, threshold=%.4f)", rms, self.threshold)
                    return True
            else:
                self._onset_start = None  # type: ignore[attr-defined]
        else:
            if rms < self.threshold:
                if not hasattr(self, "_offset_start") or self._offset_start is None:  # type: ignore[attr-defined]
                    self._offset_start = now  # type: ignore[attr-defined]
                elapsed_ms = (now - self._offset_start) * 1000  # type: ignore[attr-defined]
                if elapsed_ms >= self._release_ms:
                    self.is_speaking = False
                    self._state_entered_at = now
                    self._offset_start = None  # type: ignore[attr-defined]
                    logger.debug("Speech ENDED (rms=%.4f, threshold=%.4f)", rms, self.threshold)
                    return True
            else:
                self._offset_start = None  # type: ignore[attr-defined]

        return False
