"""Face crop manager for Wav2Lip lipsync — static detection at startup."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from scrumsurvivor.detection.face_detector import FaceRect, detect_face_once

logger = logging.getLogger(__name__)

# Target size for Wav2Lip model input face crop
_WAV2LIP_FACE_SIZE = (96, 96)

# Replace only a focused mouth ellipse inside the face crop.
# The old lower-face blend replaced too much of the cheeks and jaw from the
# 96x96 Wav2Lip output, which could make the speaking face look uncanny.
_MOUTH_CENTER_X_RATIO = 0.50
_MOUTH_CENTER_Y_RATIO = 0.66
_MOUTH_WIDTH_RATIO = 0.24
_MOUTH_HEIGHT_RATIO = 0.12


class FaceCropManager:
    """Manages the face crop region used for Wav2Lip inference.

    Face detection runs via Haar cascade on startup to obtain a tight
    bounding box around the face.

    Args:
        base_photo: BGR image of the avatar base photo.
    """

    def __init__(self, base_photo: np.ndarray) -> None:
        self._base_photo = base_photo
        self._rect: FaceRect | None = None

    def detect(self) -> FaceRect | None:
        """Run face detection on the base photo and store the result."""
        if self._rect is not None:
            return self._rect
        rect = detect_face_once(self._base_photo)
        if rect is not None:
            # Sanity-check: reject rects that are unreasonably large
            # (> 60 % of image width or height). That means the cascade
            # picked up a false positive or the image is weird.
            img_h, img_w = self._base_photo.shape[:2]
            _, _, rw, rh = rect
            if rw > img_w * 0.6 or rh > img_h * 0.6:
                logger.warning(
                    "Auto-detected face rect %s looks too large "
                    "(image is %dx%d). Rejecting.",
                    rect, img_w, img_h,
                )
                return None
        self._rect = rect
        return rect

    def get_crop(self, source_frame: np.ndarray | None = None) -> np.ndarray | None:
        """Return the face crop region resized to Wav2Lip input size (96×96).

        Returns *None* if no face has been detected.
        """
        if self._rect is None:
            self.detect()
        if self._rect is None:
            return None
        x, y, w, h = self._rect
        frame = self._base_photo if source_frame is None else source_frame
        crop = frame[y : y + h, x : x + w]
        return cv2.resize(crop, _WAV2LIP_FACE_SIZE)

    def paste_back(
        self, full_frame: np.ndarray, lip_synced_face: np.ndarray
    ) -> np.ndarray:
        """Paste *only the focused mouth region* of the lip-synced face back.

        The upper face, cheeks, and most of the jaw stay from the high-res
        original. Only a soft elliptical region around the mouth is replaced
        from the Wav2Lip output so the mouth moves without turning the entire
        lower face into a resized 96x96 reconstruction.

        Args:
            full_frame: BGR image to paste into.
            lip_synced_face: Wav2Lip output face (96×96 BGR).

        Returns:
            New frame with the lip-synced mouth region blended in.
        """
        if self._rect is None:
            return full_frame

        x, y, w, h = self._rect
        result = full_frame.copy()

        # Resize lip-synced output back to original crop size
        face_resized = cv2.resize(lip_synced_face, (w, h), interpolation=cv2.INTER_LANCZOS4)

        mask = np.zeros((h, w), dtype=np.float32)
        mouth_center = (
            int(round(w * _MOUTH_CENTER_X_RATIO)),
            int(round(h * _MOUTH_CENTER_Y_RATIO)),
        )
        mouth_axes = (
            max(6, int(round(w * _MOUTH_WIDTH_RATIO))),
            max(4, int(round(h * _MOUTH_HEIGHT_RATIO))),
        )
        cv2.ellipse(mask, mouth_center, mouth_axes, 0, 0, 360, 1.0, -1)

        # Gaussian blur the mask for extra smoothness
        feather = max(5, min(w, h) // 10)
        ksize = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        region = result[y : y + h, x : x + w].astype(np.float32)
        face_f = face_resized.astype(np.float32)
        mask_3 = mask[:, :, np.newaxis]
        blended = region * (1.0 - mask_3) + face_f * mask_3
        result[y : y + h, x : x + w] = blended.astype(np.uint8)
        return result

    @property
    def rect(self) -> FaceRect | None:
        return self._rect

