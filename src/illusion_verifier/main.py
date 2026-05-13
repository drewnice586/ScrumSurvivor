"""CLI for recording the virtual camera + audio cable output for review."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

import click

from illusion_verifier.devices import (
    VideoDeviceInfo,
    find_video_devices,
    list_audio_input_devices,
    parse_audio_device,
    pick_default_audio_input,
    resolve_video_device,
    select_video_device_by_slot,
)
from illusion_verifier.recorder import record_session

_CONFIG_FILE = Path("config.yaml")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )


def _load_last_devices() -> dict:
    try:
        data = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8")) or {}
        return {
            "video_capture_index": data.get("illusion_verifier_last_video_capture_index"),
            "audio_device_id": data.get("illusion_verifier_last_audio_device_id"),
        }
    except Exception:
        return {}


def _save_last_devices(video_capture_index: int, audio_device_id: int) -> None:
    try:
        data = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8")) or {}
        data["illusion_verifier_last_video_capture_index"] = video_capture_index
        data["illusion_verifier_last_audio_device_id"] = audio_device_id
        _CONFIG_FILE.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        pass


def _print_video_devices(devices: list[VideoDeviceInfo]) -> None:
    click.echo(f"Detecting cameras... found {len(devices)}.\n")
    if not devices:
        click.echo("  (none detected)")
        return
    click.echo("  Available cameras:")
    for device in devices:
        click.echo(
            f"    [{device.slot}]  {device.name} (OpenCV index {device.capture_index})"
        )


def _prompt_for_video_device(devices: list[VideoDeviceInfo], default_capture_index: int | None = None) -> VideoDeviceInfo:
    default_slot = devices[0].slot
    if default_capture_index is not None:
        for d in devices:
            if d.capture_index == default_capture_index:
                default_slot = d.slot
                break
    valid_slots = ", ".join(str(device.slot) for device in devices)
    while True:
        selected_slot = click.prompt("Select video device", type=int, default=default_slot)
        device = select_video_device_by_slot(selected_slot, devices)
        if device is not None:
            return device
        click.echo(f"Invalid video selection. Choose one of: {valid_slots}")


def _print_audio_devices(devices) -> list[int]:
    click.echo("Audio input devices:")
    if not devices:
        click.echo("  (none detected)")
        return []
    for device in devices:
        click.echo(
            f"  [{device.id}]  {device.name} "
            f"(inputs={device.max_input_channels}, rate={int(device.default_samplerate)})"
        )
    return [device.id for device in devices]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--list-devices", is_flag=True, default=False, help="List video/audio inputs and exit.")
@click.option("--video-device", type=int, default=None, help="OpenCV video device index.")
@click.option("--audio-device", type=str, default=None, help="Audio input id or partial device name.")
@click.option("--duration", type=float, default=0.0, show_default=True, help="Seconds to record. Use 0 or negative to run until the stop hotkey.")
@click.option("--stop-hotkey", type=str, default="ctrl+shift+f10", show_default=True, help="Global hotkey that stops the recording.")
@click.option("--fps", type=int, default=25, show_default=True, help="Video writer FPS.")
@click.option("--width", type=int, default=None, help="Requested capture width.")
@click.option("--height", type=int, default=None, help="Requested capture height.")
@click.option("--label", type=str, default="illusion", show_default=True, help="Recording label prefix.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("illusion_verifier") / "recordings",
    show_default=True,
    help="Where review artifacts are saved.",
)
@click.option("--preview/--no-preview", default=True, show_default=True, help="Show live preview while recording.")
def cli(
    list_devices: bool,
    video_device: int | None,
    audio_device: str | None,
    duration: float,
    stop_hotkey: str,
    fps: int,
    width: int | None,
    height: int | None,
    label: str,
    output_dir: Path,
    preview: bool,
) -> None:
    """Record the routed virtual camera and virtual audio cable for later review."""
    _setup_logging()

    detected_video_devices = find_video_devices()
    last = _load_last_devices()

    if list_devices:
        _print_video_devices(detected_video_devices)
        click.echo()
        _print_audio_devices(list_audio_input_devices())
        return

    if not detected_video_devices:
        raise click.ClickException("No video devices detected.")

    if width is None and height is not None or width is not None and height is None:
        raise click.ClickException("Specify both --width and --height together.")
    if not stop_hotkey.strip():
        raise click.ClickException("Stop hotkey must not be empty.")

    stop_hotkey = stop_hotkey.strip()

    if video_device is None:
        _print_video_devices(detected_video_devices)
        selected_video_device = _prompt_for_video_device(
            detected_video_devices,
            default_capture_index=last.get("video_capture_index"),
        )
    else:
        selected_video_device = resolve_video_device(video_device, detected_video_devices)

    parsed_audio_device = parse_audio_device(audio_device)
    if parsed_audio_device is None:
        audio_devices = list_audio_input_devices()
        _print_audio_devices(audio_devices)
        if not audio_devices:
            raise click.ClickException("No audio input devices detected.")
        # Prefer last-used device, then VB-Cable heuristic
        last_audio = last.get("audio_device_id")
        valid_ids = {d.id for d in audio_devices}
        if last_audio is not None and last_audio in valid_ids:
            default_audio = last_audio
        else:
            default_audio = pick_default_audio_input(audio_devices)
        if default_audio is None:
            raise click.ClickException("No audio input devices detected.")
        parsed_audio_device = click.prompt(
            "Select audio input id",
            type=int,
            default=default_audio,
        )

    _save_last_devices(
        video_capture_index=selected_video_device.capture_index,
        audio_device_id=int(parsed_audio_device),
    )

    click.echo()
    click.echo("Sync review tip: start with a clap or say 'one two three / pa pa pa'.")
    if duration > 0:
        click.echo(f"Recording for {duration:.1f} seconds or until you press {stop_hotkey}...")
    elif preview:
        click.echo(
            f"Recording until you press {stop_hotkey}, Q in the preview window, or Ctrl+C in the terminal..."
        )
    else:
        click.echo(f"Recording until you press {stop_hotkey} or Ctrl+C in the terminal...")

    result = record_session(
        video_device=selected_video_device.capture_index,
        audio_device=parsed_audio_device,
        output_dir=output_dir,
        duration_s=duration if duration > 0 else None,
        fps=fps,
        preview=preview,
        label=label,
        stop_hotkey=stop_hotkey,
        frame_size=(width, height) if width is not None and height is not None else None,
        video_backend=selected_video_device.backend,
    )

    click.echo()
    click.echo(f"Video file:   {result.video_path}")
    click.echo(f"Audio file:   {result.audio_path}")
    click.echo(f"Metadata:     {result.metadata_path}")
    if result.review_path is not None:
        click.echo(f"Review video: {result.review_path}")
    else:
        click.echo("Review video: not muxed (install ffmpeg on PATH to auto-create one MP4)")
    click.echo(
        f"Captured {result.frame_count} source frames -> {result.encoded_frame_count} encoded frames, "
        f"capture_fps={result.effective_capture_fps:.2f}."
    )
    click.echo(
        f"Capture span: {result.video_duration_s:.2f}s video, "
        f"encoded review span: {result.encoded_video_duration_s:.2f}s, "
        f"audio span: {result.audio_duration_s:.2f}s @ {result.audio_sample_rate} Hz."
    )
    click.echo(
        f"Raw capture offsets: audio start {result.raw_audio_start_offset_ms:+.1f} ms vs first video frame, "
        f"A/V end gap {result.raw_av_duration_gap_ms:+.1f} ms before alignment."
    )