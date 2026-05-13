"""Frame compositor — places the animated overlay onto the background."""

from __future__ import annotations

import cv2
import numpy as np


class FrameCompositor:
    """Composites the avatar overlay onto a background image.

    In Phase 1 there is no separate background — the virtual camera sees
    the full overlay frame.  In future phases a background image can be
    supplied and the avatar blended on top.

    Args:
        background: Static BGR background image, or *None* to use black.
        output_size: ``(width, height)`` of the output frame.
    """

    def __init__(
        self,
        background: np.ndarray | None = None,
        output_size: tuple[int, int] = (1280, 720),
    ) -> None:
        self._output_size = output_size
        w, h = output_size
        if background is not None:
            self._background = cv2.resize(background, (w, h))
        else:
            self._background = np.zeros((h, w, 3), dtype=np.uint8)

    def compose(self, overlay: np.ndarray) -> np.ndarray:
        """Place *overlay* on the background and return the composite frame.

        *overlay* is resized to match the output resolution if needed.
        """
        w, h = self._output_size
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        return overlay.copy()

    def compose_with_background(
        self, overlay: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Blend *overlay* on top of the stored background using *mask*.

        Args:
            overlay: BGR image same size as output.
            mask: Float32 alpha mask (H×W), 1.0 = fully overlay.
                  If *None*, overlay completely replaces background.
        """
        w, h = self._output_size
        overlay_r = cv2.resize(overlay, (w, h)) if overlay.shape[:2] != (h, w) else overlay

        if mask is None:
            return overlay_r.copy()

        mask_3 = cv2.resize(mask, (w, h))[:, :, np.newaxis]
        result = (
            self._background.astype(np.float32) * (1.0 - mask_3)
            + overlay_r.astype(np.float32) * mask_3
        )
        return result.astype(np.uint8)
