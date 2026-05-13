"""Static face detection — runs once at startup against the base photo.

MediaPipe 0.10+ dropped the ``solutions`` API entirely, so we use OpenCV's
bundled Haar cascade (always available, no downloads required).  The result is
stored in config and reused every frame — face detection never runs at runtime.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FaceRect = tuple[int, int, int, int]  # (x, y, w, h)


def detect_face_once(frame_bgr: np.ndarray, padding: float = 0.2) -> FaceRect | None:
    """Detect the (largest) face in *frame_bgr* using OpenCV Haar cascade.

    Returns a padded bounding box ``(x, y, w, h)`` clamped to image bounds,
    or *None* if no face is found.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    # Try increasingly relaxed parameters until a face is found
    for scale, neighbours in [(1.1, 5), (1.05, 3), (1.05, 1)]:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbours,
            minSize=(80, 80),
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        logger.warning(
            "No face detected in the provided frame. "
            "Ensure base_photo.png shows a clear, front-facing face. "
            "As a fallback you can set face_crop_rect manually in config.yaml."
        )
        return None

    # Pick the largest face (most likely the subject)
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Add padding
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x = max(0, fx - pad_x)
    y = max(0, fy - pad_y)
    x2 = min(w, fx + fw + pad_x)
    y2 = min(h, fy + fh + pad_y)

    rect: FaceRect = (x, y, x2 - x, y2 - y)
    logger.info("Face detected at %s", rect)
    return rect


def crop_face(frame_bgr: np.ndarray, rect: FaceRect) -> np.ndarray:
    """Crop *frame_bgr* to *rect* ``(x, y, w, h)``."""
    x, y, w, h = rect
    return frame_bgr[y : y + h, x : x + w]
