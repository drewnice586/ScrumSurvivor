"""Standalone video+audio recorder for reviewing the ScrumSurvivor illusion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import subprocess
import threading
import time
import wave

import cv2
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingPaths:
    video_path: Path
    audio_path: Path
    metadata_path: Path
    review_path: Path


@dataclass(frozen=True)
class RecordingResult:
    video_path: Path
    audio_path: Path
    metadata_path: Path
    review_path: Path | None
    ffmpeg_used: bool
    frame_count: int
    encoded_frame_count: int
    audio_samples: int
    video_duration_s: float
    encoded_video_duration_s: float
    audio_duration_s: float
    effective_capture_fps: float
    raw_audio_start_offset_ms: float
    raw_av_duration_gap_ms: float
    audio_sample_rate: int


class AudioCapture:
    """Capture mono float32 audio and preserve per-chunk timing for analysis."""

    def __init__(
        self,
        device: int | str | None,
        sample_rate: int,
        channels: int = 1,
        blocksize: int = 512,
        start_time: float | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._start_time = start_time if start_time is not None else time.monotonic()
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._chunk_timestamps_s: list[float] = []
        self._chunk_sizes: list[int] = []

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def snapshot(self) -> tuple[np.ndarray, list[float], list[int]]:
        with self._lock:
            if self._chunks:
                audio = np.concatenate(self._chunks)
            else:
                audio = np.zeros(0, dtype=np.float32)
            return audio, list(self._chunk_timestamps_s), list(self._chunk_sizes)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("Verifier audio callback status: %s", status)
        mono = indata[:, 0].copy() if self._channels > 1 else indata[:, 0].copy()
        timestamp_s = time.monotonic() - self._start_time
        with self._lock:
            self._chunks.append(mono)
            self._chunk_timestamps_s.append(timestamp_s)
            self._chunk_sizes.append(len(mono))


class StopHotkey:
    """Listen for a global hotkey and stop the current recording when pressed."""

    _SPECIAL_KEYS = {
        "space",
        "enter",
        "return",
        "tab",
        "backspace",
        "delete",
        "del",
        "insert",
        "home",
        "end",
        "page_up",
        "page_down",
        "up",
        "down",
        "left",
        "right",
        "esc",
        "escape",
    }

    def __init__(self, hotkey: str, callback: Callable[[], None]) -> None:
        self._hotkey = hotkey.strip()
        self._hotkey_str = self._normalise_hotkey(self._hotkey)
        self._callback = callback
        self._listener = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _normalise_hotkey(hotkey: str) -> str:
        parts = hotkey.lower().split("+")
        normalised = []
        modifiers = {"ctrl", "shift", "alt", "cmd", "super"}
        for part in parts:
            stripped = part.strip()
            if stripped in modifiers:
                normalised.append(f"<{stripped}>")
            elif stripped.startswith("f") and stripped[1:].isdigit():
                normalised.append(f"<{stripped}>")
            elif stripped in StopHotkey._SPECIAL_KEYS:
                normalised.append(f"<{stripped}>")
            else:
                normalised.append(stripped)
        return "+".join(normalised)

    def start(self) -> None:
        from pynput import keyboard

        def _on_activate() -> None:
            logger.info("Verifier stop hotkey %r pressed.", self._hotkey)
            self._callback()

        self._listener = keyboard.GlobalHotKeys({self._hotkey_str: _on_activate})
        self._thread = threading.Thread(
            target=self._listener.run,
            daemon=True,
            name="verifier-stop-hotkey",
        )
        self._thread.start()
        logger.info("Verifier stop hotkey active on %r", self._hotkey)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Verifier stop hotkey deactivated.")


def sanitize_label(label: str) -> str:
    """Make a CLI-supplied label safe for file names."""
    clean = [ch.lower() if ch.isalnum() else "_" for ch in label.strip()]
    result = "".join(clean).strip("_")
    return result or "illusion"


def _target_frame_count(elapsed_s: float, fps: int) -> int:
    """Return how many constant-rate frames should exist by *elapsed_s*."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if elapsed_s <= 0:
        return 1
    return int(elapsed_s * fps) + 1


def align_audio_to_video_timeline(
    audio: np.ndarray,
    audio_start_s: float,
    video_start_s: float,
    target_duration_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Trim or pad audio so it shares the same zero-time and duration as video."""
    target_samples = max(0, int(round(target_duration_s * sample_rate)))

    if len(audio) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if audio_start_s > video_start_s:
        prepend = int(round((audio_start_s - video_start_s) * sample_rate))
        aligned = np.concatenate([np.zeros(prepend, dtype=np.float32), audio])
    else:
        trim = int(round((video_start_s - audio_start_s) * sample_rate))
        aligned = audio[trim:] if trim < len(audio) else np.zeros(0, dtype=np.float32)

    if len(aligned) < target_samples:
        aligned = np.concatenate(
            [aligned, np.zeros(target_samples - len(aligned), dtype=np.float32)]
        )
    elif len(aligned) > target_samples:
        aligned = aligned[:target_samples]

    return aligned


def build_recording_paths(
    output_dir: Path,
    label: str,
    now: datetime | None = None,
) -> RecordingPaths:
    """Create the output paths for a verifier recording session."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = f"{sanitize_label(label)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return RecordingPaths(
        video_path=output_dir / f"{stem}_video.mp4",
        audio_path=output_dir / f"{stem}_audio.wav",
        metadata_path=output_dir / f"{stem}_metadata.json",
        review_path=output_dir / f"{stem}_review.mp4",
    )


def resolve_audio_input_config(device: int | str | None) -> tuple[int | str | None, int, str]:
    """Resolve an audio input device and the sample rate to open it with."""
    fallback_rate = 44_100
    if device is None:
        info = sd.query_devices(None, "input")
        resolved_device = None
    else:
        info = sd.query_devices(device, "input")
        resolved_device = device

    sample_rate = int(round(float(info.get("default_samplerate", fallback_rate))))
    if sample_rate <= 0:
        sample_rate = fallback_rate

    name = str(info.get("name", resolved_device if resolved_device is not None else "default input"))
    return resolved_device, sample_rate, name


def resolve_ffmpeg_binary(search_root: Path | None = None) -> Path | None:
    """Resolve ffmpeg, preferring a repo-local binary over PATH."""
    if search_root is not None:
        candidate_roots = [search_root]
    else:
        candidate_roots = list(Path(__file__).resolve().parents)

    for root in candidate_roots:
        local_candidates = (
            root / "ffmpeg" / "ffmpeg.exe",
            root / "ffmpeg" / "ffmpeg",
        )
        for candidate in local_candidates:
            if candidate.is_file():
                return candidate

    ffmpeg = shutil.which("ffmpeg")
    return Path(ffmpeg) if ffmpeg is not None else None


def mux_with_ffmpeg(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    """Mux the recorded video and WAV into one review MP4 when ffmpeg exists."""
    ffmpeg = resolve_ffmpeg_binary()
    if ffmpeg is None:
        return False, "ffmpeg not found on PATH or in repo-local ffmpeg/"

    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg failed"
    return True, "ok"


def write_wave_file(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 mono audio as a 16-bit PCM WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def record_session(
    video_device: int,
    audio_device: int | str | None,
    output_dir: Path,
    duration_s: float | None,
    fps: int,
    preview: bool,
    label: str,
    stop_hotkey: str | None = None,
    frame_size: tuple[int, int] | None = None,
    video_backend: int = cv2.CAP_DSHOW,
) -> RecordingResult:
    """Capture a review session from the selected video and audio inputs."""
    paths = build_recording_paths(output_dir, label)
    start_time = time.monotonic()
    resolved_audio_device, audio_sample_rate, audio_device_name = resolve_audio_input_config(audio_device)
    audio_capture = AudioCapture(
        device=resolved_audio_device,
        sample_rate=audio_sample_rate,
        start_time=start_time,
    )
    logger.info(
        "Verifier audio input: %r at %d Hz",
        audio_device_name,
        audio_sample_rate,
    )

    if video_backend == cv2.CAP_ANY:
        cap = cv2.VideoCapture(video_device)
    else:
        cap = cv2.VideoCapture(video_device, video_backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video device {video_device}.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if frame_size is not None:
        width, height = frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Illusion Verifier (stop hotkey / Q)"
    frame_timestamps_s: list[float] = []
    writer: cv2.VideoWriter | None = None
    capture_zero_s: float | None = None
    encoded_frame_count = 0
    stop_requested = threading.Event()
    stop_listener = StopHotkey(stop_hotkey, stop_requested.set) if stop_hotkey else None

    try:
        audio_capture.start()
        if stop_listener is not None:
            stop_listener.start()
        if preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            if stop_requested.is_set():
                logger.info("Verifier recording stopped by hotkey before next frame read.")
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            elapsed_s = time.monotonic() - start_time

            if stop_requested.is_set():
                logger.info("Verifier recording stopped by hotkey at %.2fs.", elapsed_s)
                break

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(paths.video_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer for {paths.video_path}")

            if capture_zero_s is None:
                capture_zero_s = elapsed_s

            rel_elapsed_s = elapsed_s - capture_zero_s
            target_frame_count = _target_frame_count(rel_elapsed_s, fps)
            while encoded_frame_count < target_frame_count:
                writer.write(frame)
                encoded_frame_count += 1

            frame_timestamps_s.append(elapsed_s)

            if preview:
                preview_frame = frame.copy()
                cv2.putText(
                    preview_frame,
                    f"REC {elapsed_s:6.2f}s  frames={len(frame_timestamps_s)}",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if stop_hotkey:
                    cv2.putText(
                        preview_frame,
                        f"Stop: {stop_hotkey} or Q",
                        (20, 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break

            if duration_s is not None and elapsed_s >= duration_s:
                break

    except KeyboardInterrupt:
        logger.info("Verifier recording interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        audio_capture.stop()
        if stop_listener is not None:
            stop_listener.stop()
        if preview:
            cv2.destroyAllWindows()

    audio, chunk_timestamps_s, chunk_sizes = audio_capture.snapshot()

    sample_rate = audio_sample_rate
    video_start_s = frame_timestamps_s[0] if frame_timestamps_s else 0.0
    video_duration_s = (
        frame_timestamps_s[-1] - video_start_s if frame_timestamps_s else 0.0
    )
    encoded_video_duration_s = encoded_frame_count / fps if encoded_frame_count else 0.0
    effective_capture_fps = (
        (len(frame_timestamps_s) - 1) / video_duration_s
        if len(frame_timestamps_s) > 1 and video_duration_s > 0
        else 0.0
    )

    first_audio_s = chunk_timestamps_s[0] if chunk_timestamps_s else video_start_s
    raw_audio_start_offset_ms = (first_audio_s - video_start_s) * 1000.0
    raw_audio_end_s = first_audio_s + (len(audio) / sample_rate)
    raw_av_duration_gap_ms = (raw_audio_end_s - (video_start_s + video_duration_s)) * 1000.0

    aligned_audio = align_audio_to_video_timeline(
        audio=audio,
        audio_start_s=first_audio_s,
        video_start_s=video_start_s,
        target_duration_s=encoded_video_duration_s,
        sample_rate=sample_rate,
    )
    write_wave_file(paths.audio_path, aligned_audio, sample_rate)

    muxed = False
    review_path: Path | None = None
    ok, detail = mux_with_ffmpeg(paths.video_path, paths.audio_path, paths.review_path)
    if ok:
        muxed = True
        review_path = paths.review_path
    else:
        logger.info("Verifier mux skipped: %s", detail)

    audio_duration_s = len(aligned_audio) / sample_rate
    metadata = {
        "video_device": video_device,
        "video_backend": video_backend,
        "audio_device": audio_device,
        "fps_requested": fps,
        "audio_sample_rate": sample_rate,
        "frame_count": len(frame_timestamps_s),
        "encoded_frame_count": encoded_frame_count,
        "video_duration_s": video_duration_s,
        "encoded_video_duration_s": encoded_video_duration_s,
        "effective_capture_fps": effective_capture_fps,
        "audio_samples": int(len(aligned_audio)),
        "audio_duration_s": audio_duration_s,
        "frame_timestamps_s": frame_timestamps_s,
        "audio_chunk_timestamps_s": chunk_timestamps_s,
        "audio_chunk_sizes": chunk_sizes,
        "raw_audio_start_offset_ms": raw_audio_start_offset_ms,
        "raw_av_duration_gap_ms": raw_av_duration_gap_ms,
        "video_path": str(paths.video_path),
        "audio_path": str(paths.audio_path),
        "review_path": str(review_path) if review_path is not None else None,
        "ffmpeg_used": muxed,
        "sync_review_tip": "Start each take with a clap or say 'one two three' to evaluate mouth/audio alignment.",
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return RecordingResult(
        video_path=paths.video_path,
        audio_path=paths.audio_path,
        metadata_path=paths.metadata_path,
        review_path=review_path,
        ffmpeg_used=muxed,
        frame_count=len(frame_timestamps_s),
        encoded_frame_count=encoded_frame_count,
        audio_samples=int(len(aligned_audio)),
        video_duration_s=video_duration_s,
        encoded_video_duration_s=encoded_video_duration_s,
        audio_duration_s=audio_duration_s,
        effective_capture_fps=effective_capture_fps,
        raw_audio_start_offset_ms=raw_audio_start_offset_ms,
        raw_av_duration_gap_ms=raw_av_duration_gap_ms,
        audio_sample_rate=sample_rate,
    )