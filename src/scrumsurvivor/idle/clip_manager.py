"""Clip manager — reads and cycles pre-rendered idle animation clips."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import time

logger = logging.getLogger(__name__)


class IdleClipManager:
    """Loads idle animation clips from a directory and vends frames.

    The manager cycles through clips in a random order. When all clips
    have been played once, it shuffles and starts again.

    Args:
        clips_dir: Path to directory containing .mp4 / .avi idle clips.
        loop: If True (default), loop endlessly. If False, stop after one round.
    """

    SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov")
    _ROLE_BLINK = "blink"
    _ROLE_BREATHING = "breathing"
    _ROLE_SUPPLEMENTAL = "supplemental"
    _ROLE_ORDER = (_ROLE_BLINK, _ROLE_BREATHING, _ROLE_SUPPLEMENTAL)
    _BLINK_CLIP_STEMS = {"blink_standard", "blink_slow_drowsy"}
    _BREATHING_CLIP_STEMS = {"breathing_shift"}

    def __init__(self, clips_dir: str, loop: bool = True,
                 pause_min_s: float = 3.0, pause_max_s: float = 8.0,
                 blink_interval_range: tuple[float, float] = (4.0, 8.0),
                 breathing_interval_range: tuple[float, float] = (10.0, 15.0)) -> None:
        self._clips_dir = Path(clips_dir)
        self._loop = loop
        self._pause_min_s = max(0.0, pause_min_s)
        self._pause_max_s = max(self._pause_min_s, pause_max_s)
        self._blink_interval_range = self._normalize_interval_range(blink_interval_range)
        self._breathing_interval_range = self._normalize_interval_range(breathing_interval_range)
        self._clips: list[Path] = []
        self._playlist: list[Path] = []
        self._clips_by_role: dict[str, list[Path]] = {
            role: [] for role in self._ROLE_ORDER
        }
        self._playlists_by_role: dict[str, list[Path]] = {
            role: [] for role in self._ROLE_ORDER
        }
        self._last_played_by_role: dict[str, Path | None] = {
            role: None for role in self._ROLE_ORDER
        }
        self._next_due_at: dict[str, float] = {
            role: 0.0 for role in self._ROLE_ORDER
        }
        self._current_cap: cv2.VideoCapture | None = None
        self._current_role: str | None = None
        self._current_clip_overlay_capable = False
        self._pending_role: str | None = None
        self._pending_overlay_capable = False
        self._loaded = False
        self._suppressed_until: float = 0.0
        self._next_clip_start_allowed_at: float = 0.0
        self._clip_starts_blocked = False
        self._last_played_clip: Path | None = None
        self._schedules_initialized = False

    def load(self) -> int:
        """Scan *clips_dir* for clips.  Returns the number of clips found."""
        if not self._clips_dir.exists():
            logger.warning("Idle clips directory not found: %s", self._clips_dir)
            return 0
        self._clips = [
            p
            for p in sorted(self._clips_dir.iterdir())
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if not self._clips:
            logger.warning("No idle clips found in %s", self._clips_dir)
            return 0
        self._refresh_role_inventory()
        self._initialize_schedules(time.monotonic(), start_immediately=False)
        logger.info(
            "Loaded %d idle clips from %s (%d blink, %d breathing, %d supplemental)",
            len(self._clips),
            self._clips_dir,
            len(self._clips_by_role[self._ROLE_BLINK]),
            len(self._clips_by_role[self._ROLE_BREATHING]),
            len(self._clips_by_role[self._ROLE_SUPPLEMENTAL]),
        )
        self._loaded = True
        return len(self._clips)

    @property
    def has_clips(self) -> bool:
        return bool(self._loaded and self._clips)

    @property
    def is_clip_playing(self) -> bool:
        """True when a clip is actively being played (not in a pause between clips)."""
        return self._current_cap is not None and self._current_cap.isOpened()

    @property
    def has_recorded_blink_clips(self) -> bool:
        return bool(self._clips_by_role[self._ROLE_BLINK])

    @property
    def has_recorded_breathing_clips(self) -> bool:
        return bool(self._clips_by_role[self._ROLE_BREATHING])

    @property
    def has_speaking_compatible_clips(self) -> bool:
        """True when at least one recorded clip can continue during speech."""
        return bool(self._clips_by_role[self._ROLE_BLINK])

    @property
    def current_clip_allows_speaking_overlay(self) -> bool:
        """Blink clips may keep running while lipsync/audio continue."""
        return self._current_role == self._ROLE_BLINK and self._current_clip_overlay_capable

    def suppress_for(self, seconds: float) -> None:
        """Prevent a new idle clip from starting for at least *seconds* seconds."""
        duration = max(0.0, seconds)
        if duration == 0.0:
            return

        until = time.monotonic() + duration
        if until <= self._suppressed_until:
            return

        self._suppressed_until = until

        logger.info(
            "Idle clip cooldown active for %.1f s (suppressed until %.3f)",
            duration,
            self._suppressed_until,
        )

    def set_clip_starts_blocked(self, blocked: bool) -> None:
        """Prevent new idle clips from starting while speech has priority."""
        self._clip_starts_blocked = bool(blocked)

    def read_speaking_compatible_frame(self) -> np.ndarray | None:
        """Return the next blink frame allowed to continue during speech.

        This is the narrow speaking-time clip path: only blink clips may
        continue or start while speech has priority.
        """
        if not self._clips:
            return None

        now = time.monotonic()
        self._ensure_schedules_initialized(now)

        if self._current_cap is not None and self._current_cap.isOpened():
            if self._current_role != self._ROLE_BLINK or not self._current_clip_overlay_capable:
                return None
            return self._read_active_frame()

        if now < self._suppressed_until:
            return None

        if now < self._next_clip_start_allowed_at:
            return None

        self._pending_role = self._next_due_role(now, allowed_roles={self._ROLE_BLINK})
        if self._pending_role is None:
            return None

        self._pending_overlay_capable = True
        if not self._advance_clip():
            failed_role = self._pending_role
            self._pending_role = None
            self._pending_overlay_capable = False
            if failed_role is not None:
                self._schedule_next_due(failed_role, now)
            return None

        return self._read_active_frame()

    def read_frame(self) -> np.ndarray | None:
        """Return the next frame from the current clip, or *None* during pauses.

        Returns *None* between clips so the idle compositor falls back to the
        static base image until the next named role clip becomes due.
        """
        if not self._clips:
            return None

        now = time.monotonic()
        self._ensure_schedules_initialized(now)

        if self._current_cap is not None and self._current_cap.isOpened():
            return self._read_active_frame()

        if now < self._suppressed_until:
            return None

        if now < self._next_clip_start_allowed_at:
            return None

        if self._clip_starts_blocked:
            return None

        self._pending_role = self._next_due_role(now)
        if self._pending_role is None:
            return None

        self._pending_overlay_capable = False
        if not self._advance_clip():
            failed_role = self._pending_role
            self._pending_role = None
            self._pending_overlay_capable = False
            if failed_role is not None:
                self._schedule_next_due(failed_role, now)
            return None

        return self._read_active_frame()

    def _build_playlist(self, role: str | None = None) -> None:
        selected_role = role or self._pending_role or self._ROLE_SUPPLEMENTAL
        if not any(self._clips_by_role.values()) and self._clips:
            self._refresh_role_inventory()

        clips = list(self._clips_by_role.get(selected_role, []))
        if selected_role == self._ROLE_SUPPLEMENTAL and not clips:
            clips = list(self._clips)
        if len(clips) <= 1:
            playlist = clips
        else:
            playlist = random.sample(clips, len(clips))
            last_played = self._last_played_by_role.get(selected_role)
            if selected_role == self._ROLE_SUPPLEMENTAL and last_played is None:
                last_played = self._last_played_clip
            if last_played is not None and playlist[0] == last_played:
                playlist.append(playlist.pop(0))

        self._playlists_by_role[selected_role] = playlist
        if selected_role == self._ROLE_SUPPLEMENTAL:
            self._playlist = self._playlists_by_role[selected_role]

    def _advance_clip(self) -> bool:
        """Open the next due role clip. Returns False if no eligible clip can be opened."""
        role = self._pending_role or self._ROLE_SUPPLEMENTAL
        if not self._clips_by_role.get(role):
            return False

        attempts_remaining = max(1, len(self._clips_by_role[role]))
        while attempts_remaining > 0:
            playlist = self._playlists_by_role[role]
            if not playlist:
                if not self._loop and self._last_played_by_role.get(role) is not None:
                    return False
                self._build_playlist(role)
                playlist = self._playlists_by_role[role]
                if not playlist:
                    return False

            clip_path = playlist.pop(0)
            cap = cv2.VideoCapture(str(clip_path))
            attempts_remaining -= 1
            if not cap.isOpened():
                logger.warning("Cannot open idle clip: %s — skipping.", clip_path)
                continue

            self._current_cap = cap
            self._current_role = role
            self._current_clip_overlay_capable = self._pending_overlay_capable
            self._last_played_by_role[role] = clip_path
            if role == self._ROLE_SUPPLEMENTAL:
                self._last_played_clip = clip_path
            logger.debug("Now playing idle clip: %s (%s)", clip_path.name, role)
            return True

        return False

    def _close_current(self) -> None:
        if self._current_cap is not None:
            self._current_cap.release()
            self._current_cap = None
        self._current_role = None
        self._current_clip_overlay_capable = False
        self._pending_overlay_capable = False

    def _read_active_frame(self) -> np.ndarray | None:
        assert self._current_cap is not None
        ret, frame = self._current_cap.read()
        if ret:
            return frame

        finished_role = self._current_role
        finished_at = time.monotonic()
        self._close_current()
        if finished_role is not None:
            self._schedule_next_due(finished_role, finished_at)
        self._schedule_next_clip_start_allowed(finished_at)
        return None

    @staticmethod
    def _normalize_interval_range(interval_range: tuple[float, float]) -> tuple[float, float]:
        min_s, max_s = interval_range
        min_s = max(0.0, float(min_s))
        max_s = max(min_s, float(max_s))
        return min_s, max_s

    @classmethod
    def _classify_clip(cls, clip_path: Path) -> str:
        stem = clip_path.stem.lower()
        if stem in cls._BLINK_CLIP_STEMS:
            return cls._ROLE_BLINK
        if stem in cls._BREATHING_CLIP_STEMS:
            return cls._ROLE_BREATHING
        return cls._ROLE_SUPPLEMENTAL

    def _refresh_role_inventory(self) -> None:
        self._clips_by_role = {role: [] for role in self._ROLE_ORDER}
        self._playlists_by_role = {role: [] for role in self._ROLE_ORDER}
        for clip_path in self._clips:
            role = self._classify_clip(clip_path)
            self._clips_by_role[role].append(clip_path)
        self._playlist = self._playlists_by_role[self._ROLE_SUPPLEMENTAL]

    def _ensure_schedules_initialized(self, now: float) -> None:
        if self._schedules_initialized:
            return
        self._refresh_role_inventory()
        self._initialize_schedules(now, start_immediately=True)

    def _initialize_schedules(self, now: float, start_immediately: bool) -> None:
        for role in self._ROLE_ORDER:
            if not self._clips_by_role[role]:
                self._next_due_at[role] = float("inf")
                continue
            if start_immediately:
                self._next_due_at[role] = now
            else:
                self._schedule_next_due(role, now)
        self._schedules_initialized = True

    def _schedule_next_due(self, role: str, now: float) -> None:
        if not self._clips_by_role.get(role):
            self._next_due_at[role] = float("inf")
            return

        if role == self._ROLE_BLINK:
            min_s, max_s = self._blink_interval_range
        elif role == self._ROLE_BREATHING:
            min_s, max_s = self._breathing_interval_range
        else:
            min_s, max_s = self._pause_min_s, self._pause_max_s

        delay = min_s if min_s == max_s else random.uniform(min_s, max_s)
        self._next_due_at[role] = max(now + delay, self._suppressed_until)

    def _schedule_next_clip_start_allowed(self, now: float) -> None:
        delay = (
            self._pause_min_s
            if self._pause_min_s == self._pause_max_s
            else random.uniform(self._pause_min_s, self._pause_max_s)
        )
        self._next_clip_start_allowed_at = max(now + delay, self._suppressed_until)

    def _next_due_role(
        self,
        now: float,
        allowed_roles: set[str] | None = None,
    ) -> str | None:
        due_roles = [
            role
            for role in self._ROLE_ORDER
            if (
                (allowed_roles is None or role in allowed_roles)
                and self._clips_by_role[role]
                and now >= self._next_due_at.get(role, float("inf"))
            )
        ]
        if not due_roles:
            return None

        role_priority = {role: index for index, role in enumerate(self._ROLE_ORDER)}
        return min(
            due_roles,
            key=lambda role: (self._next_due_at.get(role, float("inf")), role_priority[role]),
        )
