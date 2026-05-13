from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np


@dataclass(slots=True)
class _ScheduledChunk:
    start: int
    data: np.ndarray

    @property
    def end(self) -> int:
        return self.start + len(self.data)


class AudioPresentationScheduler:
    """Schedule microphone audio against one presentation timeline.

    Audio normally plays after a fixed base delay. When a speaking event begins
    during an idle clip, the pipeline can temporarily hold scheduling and later
    release the held audio so playback starts after the clip has ended.
    """

    def __init__(
        self,
        base_delay_ms: float,
        sample_rate: int = 48_000,
        history_seconds: float = 5.0,
        speech_detector=None,
        output_gain: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._base_delay_samples = max(0, int(round(base_delay_ms / 1000.0 * sample_rate)))
        self._lock = threading.Lock()
        self._scheduled_chunks: deque[_ScheduledChunk] = deque()
        self._hold_chunks: list[np.ndarray] = []
        self._hold_samples = 0
        self._hold_active = False
        self._capture_total = 0
        self._output_total = 0
        self._next_scheduled_output = 0
        self._has_received_audio = False
        self._hold_audio_output_end: int = 0  # output position where last hold-release audio finishes
        self._speech_detector = speech_detector
        self._output_gain = max(0.0, float(output_gain))

        history_capacity = max(int(round(history_seconds * sample_rate)), 4096)
        self._history = np.zeros(history_capacity, dtype=np.float32)
        self._history_capacity = history_capacity
        self._history_total = 0

        # Panic passthrough state
        self._passthrough = False
        self._base_delay_samples_saved = self._base_delay_samples

    def set_passthrough(self, enabled: bool) -> None:
        """Enable or disable zero-delay passthrough mode (panic button audio).

        When *enabled* is True all future pushed audio is scheduled for
        immediate output instead of the normal ``base_delay_ms`` offset.
        Any queued-but-not-yet-output chunks and the current hold are
        discarded so the switch is instant.

        When *enabled* is False the original delay is restored.
        """
        with self._lock:
            if enabled and not self._passthrough:
                self._passthrough = True
                # Release any hold immediately — we want continuous audio
                if self._hold_active:
                    self._hold_active = False
                    self._hold_chunks.clear()
                    self._hold_samples = 0
                # Drop scheduled chunks that are still in the future so new
                # audio from the mic plays right away without backlog noise.
                self._scheduled_chunks = deque(
                    c for c in self._scheduled_chunks if c.end <= self._output_total
                )
                self._next_scheduled_output = self._output_total
            elif not enabled and self._passthrough:
                self._passthrough = False
                # Restore normal delay; new audio will be re-delayed from here.
                self._next_scheduled_output = self._output_total

    @property
    def passthrough(self) -> bool:
        return self._passthrough

    # Maximum number of samples that may be scheduled ahead of the current
    # output position.  If the backlog exceeds this (e.g. because a slow
    # render frame stalled the video loop while audio kept arriving), all
    # future-scheduled chunks are discarded and the pointer is reset to
    # output_total + base_delay so audio resumes in real time.
    _MAX_BACKLOG_SAMPLES: int = 48_000 * 2  # 2 seconds

    def push_chunk(self, audio_chunk: np.ndarray) -> None:
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if len(chunk) == 0:
            return

        with self._lock:
            capture_start = self._capture_total
            self._capture_total += len(chunk)
            self._has_received_audio = True

            if self._hold_active:
                self._hold_chunks.append(chunk.copy())
                self._hold_samples += len(chunk)
                return

            if self._passthrough:
                # Schedule for immediate output — no delay
                scheduled_start = max(self._output_total, self._next_scheduled_output)
            else:
                desired_start = capture_start + self._base_delay_samples
                # Never schedule in the past (output_total may have advanced far
                # ahead during the pipeline initialisation phase before any audio
                # was pushed).  If desired_start is behind output_total, schedule
                # at output_total so audio plays immediately without being dropped.
                desired_start = max(desired_start, self._output_total)

                # Backlog guard: if slow render frames caused audio to pile up
                # far ahead of what is currently being output, snap back.
                # This prevents a permanent delay accumulation after any stall.
                if self._next_scheduled_output - self._output_total > self._MAX_BACKLOG_SAMPLES:
                    # Discard all chunks that have not yet started playing.
                    self._scheduled_chunks = deque(
                        c for c in self._scheduled_chunks
                        if c.end <= self._output_total
                    )
                    self._next_scheduled_output = self._output_total + self._base_delay_samples

                scheduled_start = max(desired_start, self._next_scheduled_output)
            self._append_chunk_locked(chunk, scheduled_start)

    def begin_hold(self) -> None:
        with self._lock:
            if self._hold_active:
                return
            self._absorb_future_audio_into_hold_locked()
            self._hold_active = True

    def release_hold(self) -> bool:
        with self._lock:
            if not self._hold_active:
                return False

            self._hold_active = False
            if not self._hold_chunks:
                return False

            release_start = max(
                self._output_total + self._base_delay_samples,
                self._next_scheduled_output,
            )
            held_audio = np.concatenate(self._hold_chunks)
            self._hold_chunks.clear()
            self._hold_samples = 0
            self._append_chunk_locked(held_audio, release_start)
            self._hold_audio_output_end = release_start + len(held_audio)
            return True

    def pull(self, num_samples: int) -> np.ndarray:
        out = np.zeros(num_samples, dtype=np.float32)
        self.pull_into(out)
        return out

    def pull_into(self, out: np.ndarray) -> None:
        if out.ndim != 1:
            raise ValueError("pull_into expects a 1-D float32 array")

        with self._lock:
            out.fill(0.0)
            output_start = self._output_total
            output_end = output_start + len(out)

            while self._scheduled_chunks and self._scheduled_chunks[0].end <= output_start:
                self._scheduled_chunks.popleft()

            for chunk in self._scheduled_chunks:
                if chunk.start >= output_end:
                    break

                overlap_start = max(output_start, chunk.start)
                overlap_end = min(output_end, chunk.end)
                if overlap_end <= overlap_start:
                    continue

                dst_start = overlap_start - output_start
                src_start = overlap_start - chunk.start
                length = overlap_end - overlap_start
                out[dst_start : dst_start + length] = chunk.data[src_start : src_start + length]

            self._output_total = output_end

        # Apply output gain to reduce noise amplitude while keeping speech
        # clearly audible. This is applied before the detector gate so the
        # detector sees true levels, but the output/history sees reduced.
        if self._output_gain != 1.0:
            out *= self._output_gain

        # Feed detector with gain-adjusted audio, then gate the output:
        # replace with silence when not speaking. This prevents mouse
        # clicks and background noise from reaching VB-Cable and lip-sync.
        if self._speech_detector is not None:
            self._speech_detector.update(out)
            if not self._speech_detector.is_speaking:
                out.fill(0.0)

        # Append the (possibly gated) output to history so that lip-sync
        # also sees silence during non-speech.
        with self._lock:
            self._append_history_locked(out)

    def recent_output_window(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            available = min(num_samples, self._history_total, self._history_capacity)
            out = np.zeros(num_samples, dtype=np.float32)
            if available == 0:
                return out

            start = self._history_total - available
            read_pos = start % self._history_capacity
            if read_pos + available <= self._history_capacity:
                out[-available:] = self._history[read_pos : read_pos + available]
                return out

            first = self._history_capacity - read_pos
            out[-available : -available + first] = self._history[read_pos:]
            out[-(available - first) :] = self._history[: available - first]
            return out

    @property
    def available_samples(self) -> int:
        with self._lock:
            scheduled = max(0, self._next_scheduled_output - self._output_total)
            return scheduled + self._hold_samples

    @property
    def is_primed(self) -> bool:
        return self._has_received_audio

    @property
    def hold_active(self) -> bool:
        with self._lock:
            return self._hold_active

    @property
    def base_delay_samples(self) -> int:
        return self._base_delay_samples

    @property
    def has_scheduled_hold_audio(self) -> bool:
        """True while hold-release speech audio is still being consumed by the output.

        Resets to False once the output timeline has passed the end of the last
        released hold, so this never stays True after the held speech has played.
        """
        with self._lock:
            return self._output_total < self._hold_audio_output_end

    @property
    def presentation_is_speaking(self) -> bool:
        if self._speech_detector is None:
            return False
        return bool(self._speech_detector.is_speaking)

    def _append_chunk_locked(self, chunk: np.ndarray, start: int) -> None:
        self._scheduled_chunks.append(_ScheduledChunk(start=start, data=chunk.copy()))
        self._next_scheduled_output = start + len(chunk)

    def _absorb_future_audio_into_hold_locked(self) -> None:
        if not self._scheduled_chunks:
            self._next_scheduled_output = max(self._next_scheduled_output, self._output_total)
            return

        future_chunks: list[np.ndarray] = []
        for chunk in self._scheduled_chunks:
            if chunk.end <= self._output_total:
                continue

            if chunk.start >= self._output_total:
                future_chunks.append(chunk.data)
                continue

            offset = self._output_total - chunk.start
            future_chunks.append(chunk.data[offset:])

        self._scheduled_chunks.clear()
        self._next_scheduled_output = self._output_total

        for future in future_chunks:
            if len(future) == 0:
                continue
            self._hold_chunks.append(future.copy())
            self._hold_samples += len(future)

    def _append_history_locked(self, chunk: np.ndarray) -> None:
        length = len(chunk)
        if length == 0:
            return

        if length >= self._history_capacity:
            self._history[:] = chunk[-self._history_capacity :]
            self._history_total += length
            return

        write_pos = self._history_total % self._history_capacity
        if write_pos + length <= self._history_capacity:
            self._history[write_pos : write_pos + length] = chunk
        else:
            first = self._history_capacity - write_pos
            self._history[write_pos:] = chunk[:first]
            self._history[: length - first] = chunk[first:]
        self._history_total += length