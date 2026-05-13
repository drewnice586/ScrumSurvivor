"""Helpers for persisting and resolving audio device selectors."""

from __future__ import annotations

from dataclasses import dataclass

_SELECTOR_SEPARATOR = " | "


@dataclass(frozen=True, slots=True)
class OutputDeviceSelector:
    name: str
    hostapi_name: str | None = None
    device_id: int | None = None


def format_output_device_selector(
    name: str,
    hostapi_name: str | None = None,
    device_id: int | None = None,
) -> str:
    parts = [name]
    if hostapi_name:
        parts.append(f"hostapi={hostapi_name}")
    if device_id is not None:
        parts.append(f"id={device_id}")
    return _SELECTOR_SEPARATOR.join(parts)


def parse_output_device_selector(raw_value: str) -> OutputDeviceSelector:
    parts = [part.strip() for part in raw_value.split(_SELECTOR_SEPARATOR) if part.strip()]
    if not parts:
        return OutputDeviceSelector(name=raw_value)

    name = parts[0]
    hostapi_name: str | None = None
    device_id: int | None = None

    for token in parts[1:]:
        key, separator, value = token.partition("=")
        if separator != "=":
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "hostapi" and normalized_value:
            hostapi_name = normalized_value
        elif normalized_key == "id":
            try:
                device_id = int(normalized_value)
            except ValueError:
                continue

    return OutputDeviceSelector(name=name, hostapi_name=hostapi_name, device_id=device_id)


def resolve_hostapi_name(device_info: dict, hostapis: list[dict]) -> str:
    hostapi_index = device_info.get("hostapi")
    if hostapi_index is None:
        return "unknown"
    try:
        return str(hostapis[int(hostapi_index)].get("name", "unknown"))
    except (IndexError, TypeError, ValueError):
        return "unknown"


def output_device_score(device_info: dict, hostapi_name: str | None = None) -> int:
    name = str(device_info["name"]).lower()
    score = 0

    if "cable input" in name:
        score += 100
    if "vb-audio virtual cable" in name:
        score += 80
    elif "virtual cable" in name:
        score += 40
    elif "vb-audio" in name:
        score += 20

    if "vb-audio point" in name or "audio point" in name:
        score -= 60

    max_output_channels = int(device_info.get("max_output_channels", 0))
    if max_output_channels == 2:
        score += 25
    elif max_output_channels == 1:
        score += 10
    elif max_output_channels > 2:
        score -= 10

    default_rate = int(round(float(device_info.get("default_samplerate", 0.0))))
    if default_rate == 48_000:
        score += 5

    if name == "cable input (vb-audio virtual cable)":
        score += 25

    normalized_hostapi = (hostapi_name or "").lower()
    if "wasapi" in normalized_hostapi:
        score += 15
    elif "directsound" in normalized_hostapi:
        score -= 5
    elif "mme" in normalized_hostapi:
        score -= 10

    return score


def selector_matches_device(
    selector: OutputDeviceSelector,
    device_id: int,
    device_info: dict,
    hostapi_name: str,
) -> bool:
    if device_info.get("max_output_channels", 0) <= 0:
        return False

    device_name = str(device_info["name"])
    if selector.name and selector.name.lower() not in device_name.lower():
        return False
    if selector.hostapi_name and selector.hostapi_name.lower() != hostapi_name.lower():
        return False
    if selector.device_id is not None and selector.device_id != device_id:
        return False
    return True