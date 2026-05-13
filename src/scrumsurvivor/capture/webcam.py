"""Webcam capture module."""

from __future__ import annotations

import logging
import threading
from types import TracebackType

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCapture:
    """Thread-safe webcam capture using cv2.VideoCapture.

    Reads frames in a background thread so the pipeline never blocks
    waiting for a sensor read. Drops stale frames — callers always get
    the most recent frame.

    Usage::

        with WebcamCapture(device=0, target_fps=25) as cam:
            frame = cam.read()   # numpy BGR uint8 or None on first call
    """

    def __init__(self, device: int = 0, target_fps: int = 25, backend: int | None = None) -> None:
        self._device = device
        self._target_fps = target_fps
        self._backend = backend  # None → cv2.CAP_DSHOW; set to cam.backend from cv2_enumerate_cameras when needed
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def open(self) -> "WebcamCapture":
        """Open the capture device and start the background reader thread."""
        backend = self._backend if self._backend is not None else cv2.CAP_DSHOW
        device = self._device
        # cv2_enumerate_cameras returns composite indices (e.g. 1402 = MSMF:1400 + device:2).
        # When the backend is passed as a separate argument, the device index must be raw.
        if isinstance(device, int) and backend > 0 and device >= backend:
            device = device - backend
        self._cap = cv2.VideoCapture(device, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam device {self._device!r} (raw={device}, backend={backend}). "
                "Check that the webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(
            "Opened webcam device %s (raw=%d, backend=%d, requested %d fps)",
            self._device, device, backend, self._target_fps,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="webcam-reader"
        )
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the background thread and release the capture device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Webcam device %d closed.", self._device)

    def __enter__(self) -> "WebcamCapture":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame (may be *None* before the first frame arrives)."""
        with self._lock:
            return self._frame

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Webcam read failed — device may have been disconnected.")
                break
            with self._lock:
                self._frame = frame
