"""Shared full-duplex microphone/VB-Cable transport."""

from __future__ import annotations

import logging
import queue
from types import TracebackType
from typing import Protocol

import numpy as np
import sounddevice as sd

from scrumsurvivor.audio.device_selector import (
    output_device_score,
    parse_output_device_selector,
    resolve_hostapi_name,
    selector_matches_device,
)

logger = logging.getLogger(__name__)

_CABLE_KEYWORDS = ("cable", "vb-cable", "vb cable")
_ROBUST_STREAM_LATENCY = "high"


class _DelayBufferLike(Protocol):
    def pull_into(self, out: np.ndarray) -> None: ...


class DuplexAudioTransport:
    """Capture microphone input and feed VB-Cable in one PortAudio stream.

    This mirrors the proven prototype topology: one full-duplex stream on a
    shared host API so capture and presentation run on the same PortAudio clock.
    The transport exposes a microphone-like queue interface for the pipeline and
    a virtual-audio-like delay-buffer attachment for scheduled output.
    """

    QUEUE_MAXSIZE = 32

    def __init__(
        self,
        input_device: int | str | None = None,
        output_device_name: str = "CABLE Input",
        sample_rate: int = 48_000,
        channels: int = 1,
        blocksize: int = 512,
    ) -> None:
        self._input_device = input_device
        self._output_device_name = output_device_name
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._stream: sd.Stream | None = None
        self._delay_buffer: _DelayBufferLike | None = None
        self._selected_input_id: int | str | None = None
        self._selected_input_label: str = "default"
        self._selected_output_id: int | str | None = None
        self._selected_output_label: str = output_device_name
        self._selected_hostapi_name: str = "unknown"
        self._stream_mode: str = "shared"
        self._underflow_count = 0
        self._input_overflow_count = 0
        self._queue_drop_count = 0

    def attach_delay_buffer(self, delay_buffer: _DelayBufferLike | None) -> None:
        self._delay_buffer = delay_buffer

    def open(self) -> "DuplexAudioTransport":
        devices = sd.query_devices()
        output_id, output_label, output_info = self._resolve_output_device(devices)
        input_id, input_label, input_info = self._resolve_input_device(
            devices,
            preferred_hostapi_index=output_info.get("hostapi"),
        )

        input_hostapi = self._get_hostapi_name(input_info)
        output_hostapi = self._get_hostapi_name(output_info)
        if input_info.get("hostapi") != output_info.get("hostapi"):
            raise RuntimeError(
                "Duplex audio requires input/output on the same host API "
                f"(input={input_hostapi!r}, output={output_hostapi!r})."
            )

        self._selected_input_id = input_id
        self._selected_input_label = input_label
        self._selected_output_id = output_id
        self._selected_output_label = output_label
        self._selected_hostapi_name = output_hostapi

        stream_kwargs = {
            "device": (input_id, output_id),
            "samplerate": self._sample_rate,
            "channels": (self._channels, self._channels),
            "dtype": "float32",
            "blocksize": self._blocksize,
            "latency": (_ROBUST_STREAM_LATENCY, _ROBUST_STREAM_LATENCY),
            "callback": self._duplex_callback,
            "clip_off": True,
            "dither_off": True,
        }

        extra_settings = self._build_output_extra_settings(output_id, output_info, output_hostapi)
        if extra_settings is not None:
            stream_kwargs["extra_settings"] = (None, extra_settings)

        self._stream = sd.Stream(**stream_kwargs)
        self._stream.start()
        actual_latency = getattr(self._stream, "latency", None)
        logger.info(
            "DuplexAudioTransport opened (input=%r id=%s, output=%r id=%s, rate=%d, block=%d, hostapi=%s, mode=%s, latency=%r)",
            self._selected_input_label,
            self._selected_input_id,
            self._selected_output_label,
            self._selected_output_id,
            self._sample_rate,
            self._blocksize,
            self._selected_hostapi_name,
            self._stream_mode,
            actual_latency,
        )
        return self

    def start(self) -> None:
        self.open()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("DuplexAudioTransport stopped.")

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "DuplexAudioTransport":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def read(self, block: bool = False, timeout: float | None = None) -> np.ndarray | None:
        try:
            if block:
                if timeout is None:
                    return self._queue.get(block=True)
                return self._queue.get(block=True, timeout=timeout)
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def read_all(self) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @property
    def block_duration_s(self) -> float:
        return self._blocksize / self._sample_rate

    @property
    def is_callback_driven(self) -> bool:
        return True

    @property
    def underflow_count(self) -> int:
        return self._underflow_count

    @property
    def queue_drop_count(self) -> int:
        return self._queue_drop_count

    @property
    def selected_input_label(self) -> str:
        return self._selected_input_label

    def _duplex_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        _frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if getattr(status, "input_overflow", False):
            self._input_overflow_count += 1
        if getattr(status, "output_underflow", False):
            self._underflow_count += 1
        if status:
            logger.debug("duplex transport callback status: %s", status)

        mono = indata[:, 0].copy() if self._channels > 1 else indata[:, 0].copy()
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue_drop_count += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            self._queue_drop_count += 1

        outdata.fill(0.0)
        if self._delay_buffer is None or self._channels != 1:
            return

        try:
            self._delay_buffer.pull_into(outdata[:, 0])
        except Exception:
            self._underflow_count += 1
            outdata.fill(0.0)

    def _resolve_output_device(self, devices: list[dict]) -> tuple[int, str, dict]:
        hostapis = sd.query_hostapis()
        selector = parse_output_device_selector(self._output_device_name)
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
                return selector.device_id, str(preferred_device["name"]), preferred_device

        candidates: list[tuple[int, dict]] = []
        for index, device_info in enumerate(devices):
            if device_info.get("max_output_channels", 0) <= 0:
                continue
            if name_lower in str(device_info["name"]).lower():
                candidates.append((index, device_info))

        if not candidates:
            for index, device_info in enumerate(devices):
                if device_info.get("max_output_channels", 0) <= 0:
                    continue
                if any(keyword in str(device_info["name"]).lower() for keyword in _CABLE_KEYWORDS):
                    candidates.append((index, device_info))

        if not candidates:
            raise RuntimeError(
                f"VB-Cable output device {self._output_device_name!r} was not found."
            )

        device_id, device_info = max(
            candidates,
            key=lambda item: self._score_output_device(
                item[1],
                resolve_hostapi_name(item[1], hostapis),
                selector,
                item[0],
            ),
        )
        return device_id, str(device_info["name"]), device_info

    def _resolve_input_device(
        self,
        devices: list[dict],
        preferred_hostapi_index: int | None,
    ) -> tuple[int, str, dict]:
        if isinstance(self._input_device, int):
            device_id = self._input_device
            device_info = devices[device_id]
            if device_info.get("max_input_channels", 0) <= 0:
                raise RuntimeError(f"Input device {device_id!r} has no input channels.")
            device_id, device_info = self._remap_input_device_to_hostapi(
                device_id,
                device_info,
                devices,
                preferred_hostapi_index,
            )
            return device_id, str(device_info["name"]), device_info

        if self._input_device is None:
            default_device = getattr(sd.default, "device", None)
            if isinstance(default_device, (list, tuple)):
                device_id = int(default_device[0])
            else:
                device_id = int(default_device)
            if device_id < 0:
                raise RuntimeError("No default input device is configured.")
            device_info = devices[device_id]
            if device_info.get("max_input_channels", 0) <= 0:
                raise RuntimeError(f"Default input device {device_id!r} has no input channels.")
            device_id, device_info = self._remap_input_device_to_hostapi(
                device_id,
                device_info,
                devices,
                preferred_hostapi_index,
            )
            return device_id, str(device_info["name"]), device_info

        query = self._input_device.lower()
        candidates: list[tuple[int, dict]] = []
        for index, device_info in enumerate(devices):
            if device_info.get("max_input_channels", 0) <= 0:
                continue
            if query in str(device_info["name"]).lower():
                candidates.append((index, device_info))

        if not candidates:
            raise RuntimeError(f"Input device {self._input_device!r} was not found.")

        def _score(item: tuple[int, dict]) -> int:
            _, device_info = item
            name = str(device_info["name"]).lower()
            score = 0
            if name == query:
                score += 100
            if preferred_hostapi_index is not None and device_info.get("hostapi") == preferred_hostapi_index:
                score += 50
            score += int(device_info.get("max_input_channels", 0))
            return score

        device_id, device_info = max(candidates, key=_score)
        return device_id, str(device_info["name"]), device_info

    def _remap_input_device_to_hostapi(
        self,
        device_id: int,
        device_info: dict,
        devices: list[dict],
        preferred_hostapi_index: int | None,
    ) -> tuple[int, dict]:
        if preferred_hostapi_index is None:
            return device_id, device_info
        if device_info.get("hostapi") == preferred_hostapi_index:
            return device_id, device_info

        original_name = str(device_info["name"])
        for candidate_id, candidate_info in enumerate(devices):
            if candidate_id == device_id:
                continue
            if candidate_info.get("max_input_channels", 0) <= 0:
                continue
            if candidate_info.get("hostapi") != preferred_hostapi_index:
                continue
            if not self._names_match(str(candidate_info["name"]), original_name):
                continue

            logger.info(
                "Remapped microphone device %r from hostapi %s to matching hostapi %s endpoint %s for duplex audio.",
                original_name,
                self._get_hostapi_name(device_info),
                self._get_hostapi_name(candidate_info),
                candidate_id,
            )
            return candidate_id, candidate_info

        return device_id, device_info

    def _build_output_extra_settings(
        self,
        device_id: int,
        device_info: dict,
        hostapi_name: str,
    ):
        self._stream_mode = "shared"
        if not self._should_try_wasapi_exclusive(device_id, device_info, hostapi_name):
            return None

        extra_settings = sd.WasapiSettings(exclusive=True)
        try:
            sd.check_output_settings(
                device=device_id,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
        except Exception as exc:
            logger.info(
                "WASAPI exclusive mode unavailable for %r: %s; falling back to shared mode.",
                self._selected_output_label,
                exc,
            )
            return None

        self._stream_mode = "wasapi-exclusive"
        return extra_settings

    def _get_hostapi_name(self, device_info: dict) -> str:
        hostapi_index = device_info.get("hostapi")
        if hostapi_index is None:
            return "unknown"
        try:
            return str(sd.query_hostapis()[int(hostapi_index)].get("name", "unknown"))
        except Exception:
            logger.debug("Unable to resolve host API name.", exc_info=True)
            return "unknown"

    @staticmethod
    def _should_try_wasapi_exclusive(device_id: int, device_info: dict, hostapi_name: str) -> bool:
        return (
            bool(device_info)
            and hasattr(sd, "WasapiSettings")
            and isinstance(device_id, int)
            and "wasapi" in hostapi_name.lower()
        )

    @staticmethod
    def _score_output_device(
        device_info: dict,
        hostapi_name: str,
        selector,
        device_id: int,
    ) -> int:
        score = output_device_score(device_info, hostapi_name)
        if selector.hostapi_name and selector.hostapi_name.lower() == hostapi_name.lower():
            score += 50
        if selector.device_id is not None and selector.device_id == device_id:
            score += 200
        return score

    @staticmethod
    def _names_match(left: str, right: str) -> bool:
        left_norm = left.strip().lower()
        right_norm = right.strip().lower()
        return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm