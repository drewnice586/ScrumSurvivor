"""Audio delay buffer — ring buffer that delays audio by a fixed number of milliseconds."""

from __future__ import annotations

import threading

import numpy as np


class AudioDelayBuffer:
    """Thread-safe ring buffer that delays audio by *delay_ms* milliseconds.

    Uses a pre-allocated numpy array as a circular buffer for efficiency.
    Push audio chunks from the capture thread; pull delayed audio from the
    output thread.  Before the buffer has been primed (initial fill), pull
    returns silence — this initial gap is intentional and acceptable.

    Args:
        delay_ms: Desired audio delay in milliseconds.
        sample_rate: Audio sample rate (Hz).
        channels: Number of audio channels (default 1 = mono).
    """

    def __init__(
        self, delay_ms: float, sample_rate: int = 48_000, channels: int = 1
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._lock = threading.Lock()
        self.set_delay(delay_ms)

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, audio_chunk: np.ndarray) -> None:
        """Append samples from *audio_chunk* (float32 mono) into the buffer."""
        with self._lock:
            n = len(audio_chunk)
            if n == 0:
                return

            if self._write_total < self._output_total:
                gap = self._output_total - self._write_total
                self._write_silence(self._write_total, gap)
                self._write_total = self._output_total

            if n >= self._capacity:
                audio_chunk = audio_chunk[-self._capacity :]
                n = len(audio_chunk)

            write_pos = self._write_total % self._capacity
            if write_pos + n <= self._capacity:
                self._buf[write_pos:write_pos + n] = audio_chunk
            else:
                first = self._capacity - write_pos
                self._buf[write_pos:] = audio_chunk[:first]
                self._buf[: n - first] = audio_chunk[first:]

            self._write_total += n

    def pull(self, num_samples: int) -> np.ndarray:
        """Return *num_samples* of delayed audio (float32).

        Returns silence when the buffer is not yet primed.
        """
        out = np.zeros(num_samples, dtype=np.float32)
        self.pull_into(out)
        return out

    def pull_into(self, out: np.ndarray) -> None:
        """Fill *out* with delayed audio samples in-place."""
        if out.ndim != 1:
            raise ValueError("pull_into expects a 1-D float32 array")

        with self._lock:
            out.fill(0.0)
            output_start = self._output_total
            output_end = output_start + len(out)
            source_start = output_start - self._delay_samples
            source_end = output_end - self._delay_samples

            valid_start = max(0, source_start)
            valid_end = min(self._write_total, source_end)

            if valid_end > valid_start:
                offset = valid_start - source_start
                length = valid_end - valid_start
                out[offset : offset + length] = self._read_from_ring(valid_start, length)

            self._output_total = output_end

    @property
    def is_primed(self) -> bool:
        """True once the buffer has accumulated at least *delay_samples* of audio."""
        with self._lock:
            return self._write_total >= self._delay_samples

    @property
    def available_samples(self) -> int:
        """Current source/output separation in samples.

        In a stable fixed-delay configuration this should hover around
        ``delay_samples`` with some jitter margin above it.
        """
        with self._lock:
            return max(0, self._write_total - self._output_total)

    @property
    def delay_samples(self) -> int:
        """Configured delay length in samples."""
        return self._delay_samples

    def set_delay(self, delay_ms: float) -> None:
        """Resize the buffer for a new *delay_ms* value.

        Calling this clears buffered audio and resets the primed state.
        """
        self._delay_ms = delay_ms
        self._delay_samples = max(0, int(delay_ms / 1000.0 * self._sample_rate))
        with self._lock:
            # Keep several seconds of history so output timing jitter does not
            # convert into overwrite/underflow artifacts.
            self._capacity = max(self._delay_samples + (self._sample_rate * 2), self._sample_rate * 10, 4096)
            self._buf = np.zeros(self._capacity, dtype=np.float32)
            self._write_total = 0
            self._output_total = 0

    def _read_from_ring(self, start: int, length: int) -> np.ndarray:
        """Read *length* samples starting at absolute source index *start*."""
        read_pos = start % self._capacity
        if read_pos + length <= self._capacity:
            return self._buf[read_pos : read_pos + length].copy()

        first = self._capacity - read_pos
        out = np.empty(length, dtype=np.float32)
        out[:first] = self._buf[read_pos:]
        out[first:] = self._buf[: length - first]
        return out

    def _write_silence(self, start: int, length: int) -> None:
        """Mark a source gap as explicit silence in the ring buffer."""
        if length <= 0:
            return
        if length >= self._capacity:
            self._buf.fill(0.0)
            return

        write_pos = start % self._capacity
        if write_pos + length <= self._capacity:
            self._buf[write_pos : write_pos + length] = 0.0
            return

        first = self._capacity - write_pos
        self._buf[write_pos:] = 0.0
        self._buf[: length - first] = 0.0
