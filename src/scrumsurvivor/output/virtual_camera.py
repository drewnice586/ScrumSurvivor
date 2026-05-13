"""Virtual camera output via pyvirtualcam."""

from __future__ import annotations

import logging
from types import TracebackType

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VirtualCameraOutput:
    """Sends BGR frames to a virtual camera device via pyvirtualcam.

    Handles BGR→RGB conversion (pyvirtualcam expects RGB) and optional
    backend selection.

    Usage::

        with VirtualCameraOutput(width=1280, height=720, fps=25) as vcam:
            vcam.send(bgr_frame)
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
        backend: str | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._camera = None

    # ── Context manager ───────────────────────────────────────────────────────

    def open(self) -> "VirtualCameraOutput":
        import pyvirtualcam

        kwargs: dict = dict(width=self._width, height=self._height, fps=self._fps)
        if self._backend:
            kwargs["backend"] = self._backend

        self._camera = pyvirtualcam.Camera(**kwargs)
        logger.info(
            "Virtual camera opened: %dx%d @ %d fps (device=%s)",
            self._width,
            self._height,
            self._fps,
            getattr(self._camera, "device", "unknown"),
        )
        return self

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
            logger.info("Virtual camera closed.")

    def __enter__(self) -> "VirtualCameraOutput":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, frame_bgr: np.ndarray) -> None:
        """Send one *frame_bgr* (H×W×3 uint8, BGR) to the virtual camera."""
        if self._camera is None:
            raise RuntimeError("VirtualCameraOutput is not open. Use as a context manager.")
        # Resize if necessary
        h, w = frame_bgr.shape[:2]
        if w != self._width or h != self._height:
            frame_bgr = cv2.resize(frame_bgr, (self._width, self._height))
        # pyvirtualcam expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._camera.send(frame_rgb)
        self._camera.sleep_until_next_frame()
