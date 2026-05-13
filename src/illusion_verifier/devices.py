"""Device discovery helpers for the illusion verifier."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import sounddevice as sd

try:
    from cv2_enumerate_cameras import enumerate_cameras as _enumerate_cameras_helper
except ImportError:
    _enumerate_cameras_helper = None


@dataclass(frozen=True)
class VideoDeviceInfo:
    slot: int
    capture_index: int
    name: str
    backend: int = cv2.CAP_DSHOW
    width: int | None = None
    height: int | None = None
    path: str = ""

    @property
    def index(self) -> int:
        return self.capture_index


@dataclass(frozen=True)
class AudioDeviceInfo:
    id: int
    name: str
    max_input_channels: int
    default_samplerate: float


def _camera_identity_key(name: str, path: str) -> str:
    normalized_name = name.strip().lower()
    normalized_path = path.strip().lower()
    if normalized_path:
        return normalized_path.split("#{", 1)[0]
    return f"name:{normalized_name}"


def _build_video_devices(cameras: list[object]) -> list[VideoDeviceInfo]:
    deduped: dict[str, VideoDeviceInfo] = {}
    for camera in cameras:
        capture_index = int(getattr(camera, "index"))
        name = str(getattr(camera, "name", "")).strip() or f"Camera {capture_index}"
        backend = int(getattr(camera, "backend", cv2.CAP_ANY) or cv2.CAP_ANY)
        path = str(getattr(camera, "path", "") or "")
        key = _camera_identity_key(name, path)
        device = VideoDeviceInfo(
            slot=0,
            capture_index=capture_index,
            name=name,
            backend=backend,
            path=path,
        )
        existing = deduped.get(key)
        if existing is None or device.capture_index < existing.capture_index:
            deduped[key] = device

    ordered = sorted(deduped.values(), key=lambda device: device.capture_index)
    return [
        VideoDeviceInfo(
            slot=position,
            capture_index=device.capture_index,
            name=device.name,
            backend=device.backend,
            width=device.width,
            height=device.height,
            path=device.path,
        )
        for position, device in enumerate(ordered, start=1)
    ]


def _probe_video_devices(max_index: int) -> list[VideoDeviceInfo]:
    devices: list[VideoDeviceInfo] = []
    for position, capture_index in enumerate(range(max(0, max_index)), start=1):
        cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            devices.append(
                VideoDeviceInfo(
                    slot=position,
                    capture_index=capture_index,
                    name=f"Camera {capture_index}",
                    backend=cv2.CAP_DSHOW,
                    width=width,
                    height=height,
                )
            )
        finally:
            cap.release()
    return devices


def find_video_devices(max_index: int = 10) -> list[VideoDeviceInfo]:
    """Return working video inputs, preferring helper-enumerated friendly names."""
    if _enumerate_cameras_helper is not None:
        try:
            devices = _build_video_devices(list(_enumerate_cameras_helper()))
        except Exception:
            devices = []
        if devices:
            return devices
    return _probe_video_devices(max_index)


def resolve_video_device(index: int, devices: list[VideoDeviceInfo]) -> VideoDeviceInfo:
    """Resolve a CLI-supplied capture index to detected camera metadata when available."""
    for device in devices:
        if device.capture_index == index:
            return device
    return VideoDeviceInfo(
        slot=0,
        capture_index=index,
        name=f"Camera {index}",
        backend=cv2.CAP_DSHOW,
    )


def select_video_device_by_slot(slot: int, devices: list[VideoDeviceInfo]) -> VideoDeviceInfo | None:
    """Resolve an interactive slot selection into the chosen camera."""
    for device in devices:
        if device.slot == slot:
            return device
    return None


def list_audio_input_devices() -> list[AudioDeviceInfo]:
    """Return all audio devices that expose at least one input channel."""
    devices: list[AudioDeviceInfo] = []
    for device_id, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        devices.append(
            AudioDeviceInfo(
                id=device_id,
                name=str(dev["name"]),
                max_input_channels=int(dev["max_input_channels"]),
                default_samplerate=float(dev.get("default_samplerate", 44_100)),
            )
        )
    return devices


def pick_default_audio_input(devices: list[AudioDeviceInfo]) -> int | None:
    """Pick a likely review source, preferring VB-Cable output if present."""
    if not devices:
        return None

    def _score(device: AudioDeviceInfo) -> int:
        name = device.name.lower()
        score = 0

        if "cable output" in name:
            score += 100
        if "vb-audio virtual cable" in name:
            score += 80
        elif "virtual cable" in name:
            score += 40
        elif "vb-audio" in name:
            score += 20

        if "vb-audio point" in name or "audio point" in name:
            score -= 60

        if device.max_input_channels == 2:
            score += 20
        elif device.max_input_channels == 1:
            score += 10
        elif device.max_input_channels > 2:
            score -= 10

        if int(device.default_samplerate) == 48_000:
            score += 5

        if name == "cable output (vb-audio virtual cable)":
            score += 25

        return score

    return max(devices, key=_score).id


def parse_audio_device(value: str | None) -> int | str | None:
    """Convert a CLI audio-device value into int or name substring."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)
    return stripped