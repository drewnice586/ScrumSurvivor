"""Virtual audio output via sounddevice → VB-Cable."""

from __future__ import annotations

import logging
from math import isclose
from types import TracebackType
from typing import Protocol

import numpy as np
import sounddevice as sd
import soxr

from scrumsurvivor.audio.device_selector import (
    output_device_score,
    parse_output_device_selector,
    resolve_hostapi_name,
    selector_matches_device,
)

logger = logging.getLogger(__name__)

_CABLE_KEYWORDS = ("cable", "vb-cable", "vb cable")
_ROBUST_OUTPUT_LATENCY = "high"

# Overlap-save context kept across chunks so the soxr sinc filter sees
# real audio history instead of zero-padding at each chunk boundary.
_RESAMPLE_OVERLAP = 256


class _DelayBufferLike(Protocol):
    def pull(self, num_samples: int) -> np.ndarray: ...

    def pull_into(self, out: np.ndarray) -> None: ...


class VirtualAudioOutput:
    """Writes delayed audio to the VB-Cable virtual audio device.

    VB-Cable on Windows creates two virtual devices:
    - "CABLE Input"  — we write audio here (this class).
    - "CABLE Output" — Teams/Zoom uses this as the microphone input.

    Args:
        device_name: Name (or partial name) of the VB-Cable input device.
        sample_rate: Audio sample rate (Hz).
        channels: Number of channels (default 1 = mono).
        blocksize: Output stream buffer size in frames.
    """

    def __init__(
        self,
        device_name: str = "CABLE Input",
        sample_rate: int = 48_000,
        channels: int = 1,
        blocksize: int = 512,
    ) -> None:
        self._device_name = device_name
        self._sample_rate = sample_rate
        self._device_sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._stream: sd.OutputStream | None = None
        self._selected_device_id: int | str | None = None
        self._selected_device_label: str = device_name
        self._selected_hostapi_name: str = "unknown"
        self._stream_mode: str = "shared"
        self._delay_buffer: _DelayBufferLike | None = None
        self._callback_driven = False
        self._underflow_count = 0
        self._resample_tail: np.ndarray = np.zeros(0, dtype=np.float32)

    # ── Context manager ───────────────────────────────────────────────────────

    def attach_delay_buffer(self, delay_buffer: _DelayBufferLike | None) -> None:
        """Attach *delay_buffer* so PortAudio can pull audio in a callback."""
        self._delay_buffer = delay_buffer

    def start(self) -> None:
        device_id, device_label, device_sample_rate, device_info = self._find_device()
        self._selected_device_id = device_id
        self._selected_device_label = device_label
        self._device_sample_rate = device_sample_rate
        self._resample_tail = np.zeros(0, dtype=np.float32)
        stream_kwargs = self._build_stream_kwargs(device_id, device_info)
        self._callback_driven = False
        if self._delay_buffer is not None and self._device_sample_rate == self._sample_rate:
            stream_kwargs["callback"] = self._output_callback
            self._callback_driven = True
        self._stream = sd.OutputStream(**stream_kwargs)
        self._stream.start()
        actual_latency = getattr(self._stream, "latency", None)
        logger.info(
            "VirtualAudioOutput opened on device %r (id=%s, rate=%d, device_native=%d, mode=%s, hostapi=%s, latency=%r)",
            self._selected_device_label,
            device_id,
            self._sample_rate,
            self._device_sample_rate,
            self._stream_mode,
            self._selected_hostapi_name,
            actual_latency,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("VirtualAudioOutput stopped.")

    def __enter__(self) -> "VirtualAudioOutput":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, audio_chunk: np.ndarray) -> bool:
        """Write *audio_chunk* (float32) to the virtual audio device.

        Returns ``True`` if PortAudio reported an output underflow since the
        previous write call.
        """
        if self._stream is None:
            raise RuntimeError("VirtualAudioOutput is not started.")
        # sounddevice OutputStream.write() expects (frames, channels)
        if audio_chunk.ndim == 1:
            data = audio_chunk.reshape(-1, self._channels)
        else:
            data = audio_chunk
        data = data.astype(np.float32, copy=False)
        if self._device_sample_rate != self._sample_rate:
            data = self._resample_chunk(data)
        underflowed = bool(self._stream.write(data))
        if underflowed:
            self._underflow_count += 1
        return underflowed

    @property
    def blocksize(self) -> int:
        """Configured output block size in samples."""
        return self._blocksize

    @property
    def is_callback_driven(self) -> bool:
        """True when PortAudio is driving output via callback mode."""
        return self._callback_driven

    @property
    def underflow_count(self) -> int:
        """Number of underflow/status events observed by the output path."""
        return self._underflow_count

    @property
    def block_duration_s(self) -> float:
        """Duration of one output block in seconds."""
        return self._blocksize / self._sample_rate

    @staticmethod
    def list_output_devices() -> list[dict]:
        """Return all available output audio devices."""
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0:
                result.append({"id": i, "name": dev["name"]})
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_device(self) -> tuple[int | str, str, int, dict]:
        """Find the best VB-Cable input device index by name.

        Falls back to the configured *device_name* string if exact match not found
        (sounddevice supports partial name matching).
        """
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        selector = parse_output_device_selector(self._device_name)
        name_lower = selector.name.lower()

        if selector.device_id is not None and 0 <= selector.device_id < len(devices):
            preferred_device = devices[selector.device_id]
            preferred_hostapi_name = resolve_hostapi_name(preferred_device, hostapis)
            if selector_matches_device(
                selector,
                selector.device_id,
                preferred_device,
                preferred_hostapi_name,
            ):
                device_rate = int(round(float(preferred_device.get("default_samplerate", self._sample_rate))))
                if device_rate <= 0:
                    device_rate = self._sample_rate
                return selector.device_id, str(preferred_device["name"]), device_rate, preferred_device

        candidates: list[tuple[int, dict]] = []
        for i, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) <= 0:
                continue
            if name_lower in dev["name"].lower():
                candidates.append((i, dev))

        if not candidates:
            for i, dev in enumerate(devices):
                if dev.get("max_output_channels", 0) > 0 and any(
                    kw in dev["name"].lower() for kw in _CABLE_KEYWORDS
                ):
                    candidates.append((i, dev))

        if candidates:
            device_id, device_info = max(
                candidates,
                key=lambda item: self._score_device(
                    item[1],
                    resolve_hostapi_name(item[1], hostapis),
                    selector,
                    item[0],
                ),
            )
            device_rate = int(round(float(device_info.get("default_samplerate", self._sample_rate))))
            if device_rate <= 0:
                device_rate = self._sample_rate
            return device_id, str(device_info["name"]), device_rate, device_info

        logger.warning(
            "VB-Cable device %r not found. "
            "Install VB-Cable from https://vb-audio.com/Cable/ — "
            "run installer as Administrator, then reboot. "
            "In Teams → Settings → Devices, select 'CABLE Output' as microphone.",
            self._device_name,
        )
        # Return the configured name and let sounddevice try
        return self._device_name, self._device_name, self._sample_rate, {}

    def _build_stream_kwargs(self, device_id: int | str, device_info: dict) -> dict:
        """Build OutputStream kwargs using host-preferred buffering when possible."""
        hostapi_name = self._get_hostapi_name(device_info)
        self._selected_hostapi_name = hostapi_name
        self._stream_mode = "shared"

        kwargs: dict = {
            "device": device_id,
            "samplerate": self._device_sample_rate,
            "channels": self._channels,
            "dtype": "float32",
            # Let the host API choose its preferred buffering; this is the
            # PortAudio-recommended starting point for robust streams.
            "blocksize": 0,
            "latency": _ROBUST_OUTPUT_LATENCY,
        }

        if self._should_try_wasapi_exclusive(device_id, device_info, hostapi_name):
            extra_settings = sd.WasapiSettings(exclusive=True)
            try:
                sd.check_output_settings(
                    device=device_id,
                    samplerate=self._device_sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
            except Exception as exc:
                logger.info(
                    "WASAPI exclusive mode unavailable for %r: %s; falling back to shared mode.",
                    self._selected_device_label,
                    exc,
                )
            else:
                kwargs["extra_settings"] = extra_settings
                self._stream_mode = "wasapi-exclusive"

        return kwargs

    def _output_callback(self, outdata: np.ndarray, frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        """PortAudio callback used for the single-rate live output path."""
        if getattr(status, "output_underflow", False):
            self._underflow_count += 1

        if self._channels != 1:
            outdata.fill(0.0)
            return

        if self._delay_buffer is None:
            outdata.fill(0.0)
            return

        mono = outdata[:, 0]
        try:
            self._delay_buffer.pull_into(mono)
        except Exception:
            self._underflow_count += 1
            outdata.fill(0.0)

    def _get_hostapi_name(self, device_info: dict) -> str:
        try:
            hostapis = sd.query_hostapis()
            return resolve_hostapi_name(device_info, hostapis)
        except Exception:
            logger.debug("Unable to resolve host API name for device %r.", self._selected_device_label, exc_info=True)
            return "unknown"

    @staticmethod
    def _should_try_wasapi_exclusive(
        device_id: int | str,
        device_info: dict,
        hostapi_name: str,
    ) -> bool:
        return (
            isinstance(device_id, int)
            and bool(device_info)
            and hasattr(sd, "WasapiSettings")
            and "wasapi" in hostapi_name.lower()
        )

    def _score_device(
        self,
        dev: dict,
        hostapi_name: str,
        selector,
        device_id: int,
    ) -> int:
        score = output_device_score(dev, hostapi_name)
        if selector.hostapi_name and selector.hostapi_name.lower() == hostapi_name.lower():
            score += 50
        if selector.device_id is not None and selector.device_id == device_id:
            score += 200
        return score

    def _output_blocksize(self) -> int:
        """Scale blocksize proportionally when device rate differs from source."""
        if isclose(self._sample_rate, self._device_sample_rate):
            return self._blocksize
        return max(1, int(round(self._blocksize * self._device_sample_rate / self._sample_rate)))

    def _resample_chunk(self, data: np.ndarray) -> np.ndarray:
        """Resample *data* (N, channels) from source to device rate.

        Uses soxr one-shot resampling with an overlap-save context so the
        sinc filter sees real audio history at chunk boundaries instead of
        zero-padding.  This eliminates the metallic/robotic artefacts that
        per-chunk linear interpolation or the bursty soxr streaming API
        would introduce.
        """
        if len(data) == 0:
            return np.zeros((0, self._channels), dtype=np.float32)

        mono = data[:, 0]

        # Prepend overlap tail from the previous call
        if len(self._resample_tail) > 0:
            extended = np.concatenate([self._resample_tail, mono])
            tail_len = len(self._resample_tail)
        else:
            extended = mono
            tail_len = 0

        # High-quality band-limited resample of the extended block
        resampled = soxr.resample(extended, self._sample_rate, self._device_sample_rate)

        # Discard output samples that correspond to the overlap tail
        # (they were already emitted in the previous call)
        if tail_len > 0:
            skip = round(tail_len * self._device_sample_rate / self._sample_rate)
            output = resampled[skip:]
        else:
            output = resampled

        # Save new tail for the next call
        if len(mono) >= _RESAMPLE_OVERLAP:
            self._resample_tail = mono[-_RESAMPLE_OVERLAP:].copy()
        else:
            self._resample_tail = mono.copy()

        return output.reshape(-1, 1).astype(np.float32, copy=False)


