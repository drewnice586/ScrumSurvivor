"""Audio pre-processor: raw PCM → mel-spectrogram for Wav2Lip input."""

from __future__ import annotations

import logging

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# Wav2Lip mel-spectrogram parameters (must match training config)
_WAV2LIP_SR = 16_000       # 16 kHz
_N_FFT = 800
_HOP_LENGTH = 200
_WIN_LENGTH = 800
_N_MELS = 80
_FMIN = 55.0
_FMAX = 7600.0

# Number of mel frames required per Wav2Lip inference step
_MEL_FRAMES_PER_STEP = 16
# Relationship between mel frame index and video frame index:
# mel_idx = frame_idx * mel_idx_multiplier
# mel_idx_multiplier = 80 / fps = 3.2 at 25 fps


class AudioPreprocessor:
    """Converts raw audio PCM to mel-spectrograms matching Wav2Lip training params.

    Args:
        sample_rate: Input audio sample rate (e.g. 48000).
        n_mels: Number of mel bands (default 80).
    """

    def __init__(self, sample_rate: int = 48_000, n_mels: int = _N_MELS) -> None:
        self._input_sr = sample_rate
        self._n_mels = n_mels
        self._target_rms: float = 0.1  # standard speech RMS for consistent mel features
        self._mel_basis: np.ndarray | None = None
        self._build_mel_basis()

    def _build_mel_basis(self) -> None:
        from librosa.filters import mel as librosa_mel

        self._mel_basis = librosa_mel(
            sr=_WAV2LIP_SR,
            n_fft=_N_FFT,
            n_mels=self._n_mels,
            fmin=_FMIN,
            fmax=_FMAX,
        )

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample *audio* from :attr:`_input_sr` to 16 kHz."""
        if self._input_sr == _WAV2LIP_SR:
            return audio
        target_len = int(len(audio) * _WAV2LIP_SR / self._input_sr)
        resampled = scipy.signal.resample(audio, target_len)
        return resampled.astype(np.float32)

    def to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert *audio* (float32, any sample rate) to a mel-spectrogram.

        Matches the exact Wav2Lip training pipeline:
          1. Resample to 16 kHz
          2. Normalise RMS to a standard speech level (compensates for quiet mics)
          3. Preemphasis (coeff=0.97) — boosts high-frequency articulatory cues
          4. Short-time Fourier transform (librosa-style, Hann window)
          5. Mel filterbank projection
          6. Amplitude→dB, subtract ref_level_db=20
          7. Symmetric normalisation to [-4, 4]

        Returns an array of shape ``(n_mels, T)``.
        """
        import librosa

        audio_16k = self.resample(audio)

        # Normalise RMS to standard speech level so Wav2Lip gets
        # consistently strong mel features regardless of mic gain.
        # Wav2Lip was trained on speech with ~0.1 RMS.
        current_rms = float(np.sqrt(np.mean(audio_16k ** 2)))
        if current_rms > 1e-6:
            audio_16k = audio_16k * (self._target_rms / current_rms)

        # Preemphasis — matches Wav2Lip hparams.preemphasis=0.97
        audio_emph = scipy.signal.lfilter([1, -0.97], [1], audio_16k).astype(np.float32)

        # STFT via librosa (Hann window, same as Wav2Lip)
        D = librosa.stft(
            audio_emph,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
        )
        magnitude = np.abs(D)  # (freq_bins, time)

        assert self._mel_basis is not None
        mel = np.dot(self._mel_basis, magnitude)  # (n_mels, time)

        # Amplitude → dB (min_level_db = -100  →  min_level = 1e-5)
        log_mel = 20.0 * np.log10(np.maximum(1e-5, mel))

        # Subtract ref_level_db=20 (Wav2Lip training uses this shift)
        log_mel = log_mel - 20.0

        # Symmetric normalisation: max_abs_value=4, min_level_db=-100
        # formula: clip(8 * ((S + 100) / 100) - 4, -4, 4)
        log_mel = np.clip(8.0 * ((log_mel + 100.0) / 100.0) - 4.0, -4.0, 4.0)

        return log_mel.astype(np.float32)

    def extract_mel_window(
        self, mel: np.ndarray, frame_idx: int, fps: float = 25.0
    ) -> np.ndarray | None:
        """Extract the :const:`_MEL_FRAMES_PER_STEP`-frame mel window for *frame_idx*.

        Returns shape ``(1, n_mels, _MEL_FRAMES_PER_STEP)`` or *None* if the
        mel is too short.
        """
        mel_idx_multiplier = _WAV2LIP_SR / _HOP_LENGTH / fps
        start_idx = int(frame_idx * mel_idx_multiplier)
        end_idx = start_idx + _MEL_FRAMES_PER_STEP
        if end_idx > mel.shape[1]:
            return None
        window = mel[:, start_idx:end_idx]  # (n_mels, 16)
        return window[np.newaxis]           # (1, n_mels, 16)
