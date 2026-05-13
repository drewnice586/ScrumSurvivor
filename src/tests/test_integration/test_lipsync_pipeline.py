"""Integration test: lipsync pipeline (mocked Wav2Lip model)."""

from __future__ import annotations

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class _FakeTensor:
    """Minimal numpy-backed tensor mock for torch-free lipsync tests."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def squeeze(self, d=None):
        return _FakeTensor(self._arr.squeeze(d))

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def unsqueeze(self, d):
        return _FakeTensor(np.expand_dims(self._arr, d))

    def to(self, device):
        return self

    def __mul__(self, other):
        return _FakeTensor(self._arr * other)


def _build_torch_mock_for_engine():
    """Build a torch mock that works with Wav2LipEngine.process()."""
    torch_mock = MagicMock(name="torch")
    torch_mock.from_numpy.side_effect = lambda arr: _FakeTensor(arr)
    torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return torch_mock


def test_lipsync_pipeline_produces_valid_frame():
    """End-to-end: audio -> mel -> process -> paste_back -> valid frame."""
    from scrumsurvivor.lipsync.audio_preprocessor import AudioPreprocessor
    from scrumsurvivor.lipsync.face_crop import FaceCropManager
    from scrumsurvivor.lipsync.wav2lip_engine import Wav2LipEngine

    # 1. Build a fake base photo with a known crop region
    base_photo = np.full((400, 300, 3), 100, dtype=np.uint8)
    with patch(
        "scrumsurvivor.lipsync.face_crop.detect_face_once",
        return_value=(50, 50, 100, 100),
    ):
        crop_mgr = FaceCropManager(base_photo=base_photo)
        crop_mgr.detect()

    # 2. Mock engine (no real torch needed)
    torch_mock = _build_torch_mock_for_engine()
    mock_model = MagicMock()
    mock_model.return_value = _FakeTensor(np.full((1, 3, 96, 96), 0.8, dtype=np.float32))

    engine = Wav2LipEngine.__new__(Wav2LipEngine)
    engine._model = mock_model
    engine._device = "cpu"
    engine._ready = True

    # 3. Build small audio chunk (0.2 s at 44100 Hz)
    audio = np.zeros(8820, dtype=np.float32)

    # 4. Convert audio -> mel
    proc = AudioPreprocessor(sample_rate=44100)
    mel = proc.to_mel(audio)
    mel_window = proc.extract_mel_window(mel, frame_idx=0, fps=25.0)

    if mel_window is None:
        pytest.skip("Mel window too short for this audio length")

    # 5. Get face crop and run engine with torch mocked
    face_crop = crop_mgr.get_crop()
    assert face_crop is not None

    with patch.dict(sys.modules, {"torch": torch_mock}):
        synced_face = engine.process(face_crop, mel_window)

    assert synced_face.shape == (96, 96, 3)

    # 6. Paste back
    full_frame = base_photo.copy()
    result = crop_mgr.paste_back(full_frame, synced_face)
    assert result.shape == base_photo.shape
    assert result.dtype == np.uint8

