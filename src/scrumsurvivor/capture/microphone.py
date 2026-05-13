"""Microphone capture module with a thread-safe ring buffer."""

from __future__ import annotations

import logging
import queue
import threading
from types import TracebackType

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_CHUNK_FRAMES = 512  # frames per callback (about 10.7 ms at 48 kHz)


class MicrophoneCapture:
    """Continuous microphone capture using sounddevice.

    Audio chunks (float32, mono) are placed in a bounded FIFO queue.
    Callers read chunks with ``read()`` or drain the queue with
    ``read_all()``.  Old chunks are automatically dropped when the
    queue is full to avoid unbounded memory growth.

    Usage::

        with MicrophoneCapture(sample_rate=48000) as mic:
            chunk = mic.read(block=False)  # np.ndarray float32 or None
    """

    QUEUE_MAXSIZE = 32  # max buffered chunks before dropping

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._stream: sd.InputStream | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def open(self) -> "MicrophoneCapture":
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=_CHUNK_FRAMES,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(
            "Opened microphone (device=%s, rate=%d, channels=%d)",
            self._device,
            self._sample_rate,
            self._channels,
        )
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Microphone closed.")

    def __enter__(self) -> "MicrophoneCapture":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, block: bool = False, timeout: float | None = None) -> np.ndarray | None:
        """Return a single audio chunk (float32 mono) or *None* if queue is empty."""
        try:
            if block:
                if timeout is None:
                    return self._queue.get(block=True)
                return self._queue.get(block=True, timeout=timeout)
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def read_all(self) -> list[np.ndarray]:
        """Drain the entire queue and return all available chunks."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @staticmethod
    def rms(chunk: np.ndarray) -> float:
        """Return the root-mean-square amplitude of *chunk*."""
        return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("sounddevice callback status: %s", status)
        mono = indata[:, 0].copy() if self._channels > 1 else indata[:, 0].copy()
        # Drop oldest chunk if full to avoid blocking the audio thread
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            pass  # extremely unlikely after the drain above
