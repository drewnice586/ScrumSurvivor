"""Main processing pipeline — coordinates all sub-systems."""

from __future__ import annotations

from contextlib import ExitStack
import logging
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np

from scrumsurvivor.config.settings import AppConfig

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()              # Static photo or idle animation clip
    SPEAKING = auto()          # Scheduled speech presentation is active
    SPEAKING_LIVE = auto()     # Backward-compat alias; routed via scheduled audio
    SPEAKING_BUFFERED = auto() # Backward-compat alias; routed via scheduled audio


class Pipeline:
    """Main pipeline that wires capture, detection, and virtual camera together."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._running = False

        # Injected by higher phases
        self._idle_processor = None        # Phase 2 — IdleCompositor
        self._lipsync_engine = None        # Phase 3 — Wav2LipEngine
        self._face_crop_manager = None     # Phase 3 — FaceCropManager
        self._audio_preprocessor = None   # Phase 3 — AudioPreprocessor
        self._frame_compositor = None     # Phase 5 — FrameCompositor
        self._transition = None           # Phase 5 — CrossfadeTransition
        self._base_photo: np.ndarray | None = None  # loaded by CLI run

        # Audio rolling buffer (float32 capture history for diagnostics and future use)
        self._audio_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._audio_lock = threading.Lock()
        self._idle_cooldown_lock = threading.Lock()
        self._pending_idle_cooldown_reason: str | None = None
        self._latest_audio_chunk_size = 0
        self._presentation_speech_detector = None

        # Phase 4 — scheduled audio presentation + virtual audio output
        self._audio_delay_buffer = None
        self._virtual_audio = None
        self._audio_input_stop = threading.Event()
        self._audio_input_thread: threading.Thread | None = None
        self._audio_output_stop = threading.Event()
        self._audio_output_thread: threading.Thread | None = None

        # Lazy-initialised hardware handles
        self._microphone = None
        self._speech_detector = None
        self._virtual_camera = None
        self._preview = None

        # Initialization phase — True until the main loop is ready to run
        self._initializing = True
        self._init_cam_thread: threading.Thread | None = None
        self._init_cam_stop = threading.Event()

        # Tracks the last time live speech was detected so that new idle clips
        # are blocked during the audio base-delay window even if the live
        # detector briefly drops between words.
        self._live_speech_last_seen_time: float = 0.0

        # Panic button state
        self._panic_mode = False
        self._panic_webcam = None
        self._panic_last_webcam_frame: np.ndarray | None = None
        self._panic_activation_frame: np.ndarray | None = None
        self._panic_crossfade = None          # CrossfadeTransition, created in run()
        self._panic_needs_enter_crossfade = False
        self._panic_toggle_requested = threading.Event()
        self._panic_hotkey_listener = None
        self._panic_toggle_cooldown_until = 0.0  # monotonic timestamp

    # ── Phase injectors ───────────────────────────────────────────────────────

    def set_idle_processor(self, processor) -> None:
        self._idle_processor = processor

    def set_lipsync_engine(self, engine) -> None:
        self._lipsync_engine = engine

    def set_face_crop_manager(self, manager) -> None:
        self._face_crop_manager = manager

    def set_audio_preprocessor(self, preprocessor) -> None:
        self._audio_preprocessor = preprocessor

    def set_frame_compositor(self, compositor) -> None:
        self._frame_compositor = compositor

    def set_base_photo(self, photo: np.ndarray) -> None:
        self._base_photo = photo

    def set_transition(self, transition) -> None:
        self._transition = transition

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    def run(self) -> None:
        """Open all hardware resources and run the main loop until stopped."""
        from scrumsurvivor.detection.speech_detector import SpeechDetector
        from scrumsurvivor.output.virtual_camera import VirtualCameraOutput
        from scrumsurvivor.audio.presentation_scheduler import AudioPresentationScheduler
        from scrumsurvivor.audio.virtual_audio import VirtualAudioOutput

        config = self._config
        w, h = config.output_resolution
        fps = config.target_fps

        self._running = True

        with ExitStack() as stack:
            mic, duplex_audio = self._open_audio_transport(stack)
            self._microphone = mic
            self._virtual_audio = duplex_audio

            # Try to open virtual camera — not required (preview-only mode if absent)
            try:
                vcam_ctx = VirtualCameraOutput(width=w, height=h, fps=fps)
                vcam_ctx.open()
                self._virtual_camera = vcam_ctx
            except Exception as exc:
                logger.warning(
                    "Virtual camera unavailable (%s). "
                    "Running in preview-only mode. "
                    "Check: is another process holding the OBS Virtual Camera device? "
                    "Is the OBS Virtual Camera driver installed?",
                    exc,
                )
                self._virtual_camera = None

            # Broadcast a static 'initializing' frame while setup completes so
            # Teams / OBS shows something sensible instead of a black screen.
            # The thread is stopped (and joined) before the main loop takes over.
            if self._virtual_camera is not None:
                self._init_cam_stop.clear()
                self._init_cam_thread = threading.Thread(
                    target=self._init_cam_loop,
                    args=(self._make_init_frame(),),
                    daemon=True,
                    name="init-cam-feed",
                )
                self._init_cam_thread.start()
                logger.info("Init frame broadcast started on virtual camera.")

            # Calibrate speech detection
            speech = SpeechDetector(
                sample_rate=config.sample_rate,
                threshold=config.speech_threshold,
                attack_ms=config.speech_attack_ms,
                release_ms=config.speech_release_ms,
            )
            speech.calibrate(mic)
            self._speech_detector = speech
            # The presentation detector runs on the *delayed* audio output.
            # It needs a much longer release than the live detector so that
            # natural word pauses (0.3–1 s) don't flip the pipeline state
            # between SPEAKING and IDLE every sentence fragment, which causes
            # visual stuttering and lip-sync re-initialisation.
            self._presentation_speech_detector = SpeechDetector(
                sample_rate=config.sample_rate,
                threshold=speech.threshold,
                attack_ms=0,
                release_ms=config.presentation_release_ms,
            )

            # Set up scheduled audio presentation + virtual audio output (VB-Cable)
            delay_ms = config.audio_delay_ms if config.audio_delay_ms is not None else 265
            self._audio_delay_buffer = AudioPresentationScheduler(
                base_delay_ms=delay_ms,
                sample_rate=config.sample_rate,
                speech_detector=self._presentation_speech_detector,
                output_gain=config.output_gain,
            )
            try:
                if duplex_audio is not None:
                    duplex_audio.attach_delay_buffer(self._audio_delay_buffer)
                    logger.info(
                        "Duplex audio callback output active; PortAudio drives shared capture/output timing."
                    )
                else:
                    vaudio = VirtualAudioOutput(
                        device_name=config.virtual_audio_device,
                        sample_rate=config.sample_rate,
                    )
                    vaudio.attach_delay_buffer(self._audio_delay_buffer)
                    vaudio.start()
                    self._virtual_audio = vaudio
                    if vaudio.is_callback_driven:
                        logger.info("Virtual audio callback output active; PortAudio drives VB-Cable timing.")
                    else:
                        self._start_audio_output_thread()
                logger.info(
                    "Audio presentation delay active: %d ms → %r (set Microphone to 'CABLE Output' in Teams)",
                    delay_ms,
                    config.virtual_audio_device,
                )
            except Exception as exc:
                logger.warning(
                    "Virtual audio output unavailable (%s). "
                    "Teams will hear the real microphone directly. "
                    "Is VB-Cable installed? Check virtual_audio_device in config.yaml.",
                    exc,
                )
                self._virtual_audio = None

            # Log which audio input device is in use
            try:
                selected_input_label = getattr(self._microphone, "selected_input_label", None)
                if selected_input_label:
                    logger.info("Microphone: %s", selected_input_label)
                else:
                    import sounddevice as _sd
                    dev_info = _sd.query_devices(config.microphone_device, "input")
                    logger.info("Microphone: [%s] %s", config.microphone_device, dev_info["name"])
            except Exception:
                logger.info("Microphone: device=%s (query failed)", config.microphone_device)

            # Open preview window on the main thread if requested
            if config.preview_enabled:
                from scrumsurvivor.app.preview import PreviewWindow
                self._preview = PreviewWindow(scale=0.5)
                self._preview.open()
                logger.info("Preview window opened.")

            self._start_audio_input_thread()

            # Create the dedicated panic crossfade transition
            from scrumsurvivor.compositor.transition import CrossfadeTransition as _CT
            self._panic_crossfade = _CT(n_frames=config.crossfade_frames)
            self._register_panic_hotkey()
            self._preload_panic_webcam()

            # Stop the init frame and ungate audio — everything is ready
            self._init_cam_stop.set()
            if self._init_cam_thread is not None:
                self._init_cam_thread.join(timeout=2.0)
                self._init_cam_thread = None
            self._initializing = False
            logger.info("Pipeline initialized — entering main loop.")
            try:
                self._main_loop()
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted by user.")
            except Exception:
                logger.exception("Unhandled exception in pipeline main loop — pipeline crashed.")
                raise
            finally:
                self._running = False
                try:
                    if self._preview is not None:
                        self._preview.close()
                except Exception:
                    logger.warning("Error closing preview window.", exc_info=True)
                try:
                    if self._virtual_camera is not None:
                        self._virtual_camera.close()
                except Exception:
                    logger.warning("Error closing virtual camera.", exc_info=True)
                try:
                    self._stop_audio_input_thread()
                except Exception:
                    logger.warning("Error stopping audio input thread.", exc_info=True)
                try:
                    self._stop_audio_output_thread()
                except Exception:
                    logger.warning("Error stopping audio output thread.", exc_info=True)
                try:
                    if self._virtual_audio is not None:
                        self._virtual_audio.stop()
                except Exception:
                    logger.warning("Error stopping virtual audio.", exc_info=True)
                logger.info("Pipeline stopped.")
                # Flush all log handlers so messages reach file/console before exit
                for handler in logging.getLogger().handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass
                # Stop panic hotkey listener
                if self._panic_hotkey_listener is not None:
                    try:
                        self._panic_hotkey_listener.stop()
                    except Exception:
                        pass
                # Close panic webcam (kept open between toggles for instant re-activation)
                if self._panic_webcam is not None:
                    try:
                        self._panic_webcam.close()
                    except Exception:
                        pass
                    self._panic_webcam = None

    # ── Initialization frame ──────────────────────────────────────────────────

    def _make_init_frame(self) -> np.ndarray:
        """Generate a static 'ScrumSurvivor — Initializing...' frame."""
        w, h = self._config.output_resolution
        frame = np.full((h, w, 3), 20, dtype=np.uint8)  # near-black background

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "ScrumSurvivor"
        subtitle = "Initializing..."

        title_scale = w / 640.0
        sub_scale = title_scale * 0.45
        title_thick = max(1, round(title_scale * 2))
        sub_thick = max(1, round(sub_scale * 2))

        (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thick)
        (sw, sh), _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thick)

        gap = max(8, int(h * 0.025))
        block_h = th + gap + sh
        top_y = (h - block_h) // 2

        cv2.putText(
            frame, title,
            ((w - tw) // 2, top_y + th),
            font, title_scale, (220, 220, 220), title_thick, cv2.LINE_AA,
        )
        cv2.putText(
            frame, subtitle,
            ((w - sw) // 2, top_y + th + gap + sh),
            font, sub_scale, (130, 130, 130), sub_thick, cv2.LINE_AA,
        )
        return frame

    def _init_cam_loop(self, frame: np.ndarray) -> None:
        """Send the init frame at target_fps until the stop event is set."""
        interval = 1.0 / self._config.target_fps
        while not self._init_cam_stop.is_set():
            t = time.monotonic()
            if self._virtual_camera is not None:
                try:
                    self._virtual_camera.send(frame)
                except Exception:
                    break
            elapsed = time.monotonic() - t
            sleep_s = interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _open_audio_transport(self, stack: ExitStack):
        from scrumsurvivor.audio.duplex_transport import DuplexAudioTransport
        from scrumsurvivor.capture.microphone import MicrophoneCapture

        config = self._config
        duplex = DuplexAudioTransport(
            input_device=config.microphone_device,
            output_device_name=config.virtual_audio_device,
            sample_rate=config.sample_rate,
        )
        try:
            stack.enter_context(duplex)
        except Exception as exc:
            logger.warning(
                "Duplex audio transport unavailable (%s). Falling back to split input/output streams.",
                exc,
            )
        else:
            logger.info("Audio transport topology: duplex")
            return duplex, duplex

        mic = stack.enter_context(
            MicrophoneCapture(sample_rate=config.sample_rate, device=config.microphone_device)
        )
        logger.info("Audio transport topology: split")
        return mic, None

    def stop(self) -> None:
        self._running = False

    def _apply_idle_cooldown(self, reason: str) -> None:
        """Delay the next idle clip for a short period after speech ends."""
        cooldown_s = max(0.0, self._config.idle_after_speaking_cooldown_s)
        if cooldown_s == 0.0 or self._idle_processor is None:
            return

        suppress = getattr(self._idle_processor, "suppress_idle_clips_for", None)
        if callable(suppress):
            suppress(cooldown_s)
            logger.info(
                "Idle clip cooldown: %.2f s after %s",
                cooldown_s,
                reason,
            )

    def _queue_idle_cooldown(self, reason: str) -> None:
        with self._idle_cooldown_lock:
            self._pending_idle_cooldown_reason = reason

    def _consume_pending_idle_cooldown(self) -> str | None:
        with self._idle_cooldown_lock:
            reason = self._pending_idle_cooldown_reason
            self._pending_idle_cooldown_reason = None
            return reason

    def _start_audio_input_thread(self) -> None:
        """Start continuous microphone routing independent of video frame rate."""
        if self._microphone is None or self._speech_detector is None:
            return

        self._audio_input_stop.clear()
        self._audio_input_thread = threading.Thread(
            target=self._audio_input_loop,
            daemon=True,
            name="microphone-audio-input",
        )
        self._audio_input_thread.start()
        logger.info("Microphone audio routing started.")

    def _stop_audio_input_thread(self) -> None:
        """Stop the microphone routing thread."""
        self._audio_input_stop.set()
        if self._audio_input_thread is not None:
            self._audio_input_thread.join(timeout=2.0)
            self._audio_input_thread = None

    def _audio_input_loop(self) -> None:
        """Continuously drain microphone chunks and route them at audio cadence."""
        assert self._microphone is not None

        while not self._audio_input_stop.is_set():
            chunk = self._microphone.read(block=True, timeout=0.05)
            if chunk is None:
                continue

            try:
                self._process_microphone_chunk(chunk)
            except Exception:
                logger.warning("Microphone audio routing failed.", exc_info=True)
                return

    def _process_microphone_chunk(self, chunk: np.ndarray) -> None:
        """Route one microphone chunk without depending on the video frame loop.

        Architecture: every mic chunk is ALWAYS pushed to the scheduler
        (real audio, never silence).  Speech detection controls only the
        visual pipeline state (IDLE vs SPEAKING) and idle-clip blocking —
        it never gates audio content.  This guarantees no spoken words are
        lost due to detector hysteresis or timing.
        """
        assert self._speech_detector is not None

        self._latest_audio_chunk_size = len(chunk)
        was_speaking = bool(self._speech_detector.is_speaking)
        self._speech_detector.update(chunk)
        is_speaking = bool(self._speech_detector.is_speaking)

        # Log speech transitions at INFO level for diagnostics
        if is_speaking and not was_speaking:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            logger.info(
                "Speech ONSET confirmed (rms=%.4f, threshold=%.4f)",
                rms, self._speech_detector.threshold,
            )
        elif was_speaking and not is_speaking:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            logger.info(
                "Speech OFFSET (rms=%.4f, threshold=%.4f)",
                rms, self._speech_detector.threshold,
            )

        with self._audio_lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
            max_samples = self._config.sample_rate * 5
            if len(self._audio_buffer) > max_samples:
                self._audio_buffer = self._audio_buffer[-max_samples:]

        clip_playing = (
            self._idle_processor is not None
            and self._idle_processor.is_clip_playing
        )
        clip_allows_speaking_overlay = self._idle_bool_capability(
            "current_clip_allows_speaking_overlay"
        )
        clip_blocks_speaking = clip_playing and not clip_allows_speaking_overlay

        if self._audio_delay_buffer is None:
            return

        # Hold/release: buffer speech audio while an idle clip is playing
        if clip_blocks_speaking and is_speaking and not self._audio_delay_buffer.hold_active:
            self._start_presentation_hold()
        elif self._audio_delay_buffer.hold_active and not clip_blocks_speaking:
            reason = (
                "clip supports speaking overlay"
                if clip_playing and clip_allows_speaking_overlay
                else "clip no longer playing"
            )
            self._release_presentation_hold(reason)

        self._update_idle_clip_priority_gate()

        # Don't fill the delay buffer until the pipeline is fully initialized.
        if self._initializing:
            return

        # ALWAYS push the real audio — never silence-gate.
        self._audio_delay_buffer.push_chunk(chunk)

    def _start_audio_output_thread(self) -> None:
        """Start the fixed-rate audio pump for VB-Cable output."""
        if self._virtual_audio is None or self._audio_delay_buffer is None:
            return
        if self._virtual_audio.is_callback_driven:
            return
        self._audio_output_stop.clear()
        self._audio_output_thread = threading.Thread(
            target=self._audio_output_loop,
            daemon=True,
            name="virtual-audio-output",
        )
        self._audio_output_thread.start()
        logger.info(
            "Virtual audio pump started (block=%d samples, %.1f ms)",
            self._virtual_audio.blocksize,
            self._virtual_audio.block_duration_s * 1000,
        )

    def _stop_audio_output_thread(self) -> None:
        """Stop the VB-Cable audio pump thread."""
        self._audio_output_stop.set()
        if self._audio_output_thread is not None:
            self._audio_output_thread.join(timeout=2.0)
            self._audio_output_thread = None

    def _audio_output_loop(self) -> None:
        """Continuously pull fixed-size delayed audio blocks and write them out.

        Decouples synchronous VB-Cable writes from the video frame loop so
        frame time no longer grows with audio backlog.
        """
        assert self._virtual_audio is not None
        assert self._audio_delay_buffer is not None

        blocksize = self._virtual_audio.blocksize
        last_debug_log = time.monotonic()
        last_underflow_log = 0.0

        while not self._audio_output_stop.is_set():
            audio_chunk = self._audio_delay_buffer.pull(blocksize)
            try:
                underflowed = self._virtual_audio.write(audio_chunk)
            except Exception:
                logger.warning("Virtual audio write failed.", exc_info=True)
                return

            if underflowed and time.monotonic() - last_underflow_log >= 1.0:
                logger.warning("Virtual audio underflow detected; output stream was starved.")
                last_underflow_log = time.monotonic()

            if logger.isEnabledFor(logging.DEBUG) and time.monotonic() - last_debug_log >= 2.0:
                logger.debug(
                    "Audio pump: queued=%.2f s primed=%s",
                    self._audio_delay_buffer.available_samples / self._config.sample_rate,
                    self._audio_delay_buffer.is_primed,
                )
                last_debug_log = time.monotonic()

    def _start_presentation_hold(self) -> None:
        if self._audio_delay_buffer is None or self._audio_delay_buffer.hold_active:
            return
        self._audio_delay_buffer.begin_hold()
        logger.info("Audio presentation hold STARTED during idle clip")

    def _release_presentation_hold(self, reason: str) -> None:
        if self._audio_delay_buffer is None or not self._audio_delay_buffer.hold_active:
            return
        self._audio_delay_buffer.release_hold()
        logger.info("Audio presentation hold RELEASED (%s)", reason)

    def _has_pending_pipeline_audio(self) -> bool:
        """True while hold-release speech audio is still being consumed by the output.

        After a hold-release the buffered speech is scheduled into the future.
        Until all of that audio has been pulled through we keep idle clip starts
        blocked so the visual pipeline doesn't cut in before the speech is heard.

        Unlike the old available_samples approach this correctly resets to False
        once the held audio has played, regardless of how much ambient audio is
        still buffered in the scheduler.
        """
        if self._audio_delay_buffer is None:
            return False
        return self._audio_delay_buffer.has_scheduled_hold_audio

    def _set_idle_clip_starts_blocked(self, blocked: bool) -> None:
        if self._idle_processor is None:
            return
        setter = getattr(self._idle_processor, "set_clip_starts_blocked", None)
        if callable(setter):
            setter(blocked)

    def _idle_bool_capability(self, attribute_name: str) -> bool:
        if self._idle_processor is None:
            return False
        value = getattr(self._idle_processor, attribute_name, False)
        return bool(value) if isinstance(value, (bool, np.bool_)) else False

    def _get_speaking_base_frame(self) -> np.ndarray:
        base_frame = self._get_base_frame()
        if self._has_pending_pipeline_audio():
            return base_frame
        if self._idle_processor is None:
            return base_frame

        frame_getter = getattr(self._idle_processor, "speaking_base_frame", None)
        if not callable(frame_getter):
            return base_frame
        speaking_frame = frame_getter(base_frame)
        return speaking_frame if speaking_frame is not None else base_frame

    def _update_idle_clip_priority_gate(self) -> None:
        now = time.monotonic()
        live_speaking = bool(
            self._speech_detector is not None and self._speech_detector.is_speaking
        )
        if live_speaking:
            self._live_speech_last_seen_time = now
        # Grace window: keep clips blocked for audio_delay + 500 ms after the last
        # live speech sample.  This bridges the gap between the live detector
        # dropping during a natural word pause and the delayed audio reaching the
        # presentation detector output — preventing a stray idle clip from
        # starting during that window.
        delay_s = (
            (self._config.audio_delay_ms / 1000.0)
            if self._config.audio_delay_ms is not None
            else 0.265
        )
        recently_speaking = (now - self._live_speech_last_seen_time) < (delay_s + 0.5)
        hold_active = bool(
            self._audio_delay_buffer is not None and self._audio_delay_buffer.hold_active
        )
        presentation_speaking = bool(
            self._presentation_speech_detector is not None
            and self._presentation_speech_detector.is_speaking
        )
        self._set_idle_clip_starts_blocked(
            recently_speaking
            or hold_active
            or presentation_speaking
            or self._has_pending_pipeline_audio()
        )

    def _sync_state_to_presentation(self) -> None:
        assert self._presentation_speech_detector is not None

        with self._state_lock:
            next_state = (
                PipelineState.SPEAKING
                if self._presentation_speech_detector.is_speaking
                else PipelineState.IDLE
            )
            previous_state = self._state
            changed = previous_state != next_state
            if changed:
                self._state = next_state

        self._update_idle_clip_priority_gate()

        if not changed:
            return

        if next_state == PipelineState.SPEAKING:
            logger.info("State: %s → SPEAKING", previous_state.name)
        else:
            self._apply_idle_cooldown("presentation speech end")
            logger.info("State: %s → IDLE", previous_state.name)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        assert self._microphone is not None
        assert self._speech_detector is not None
        assert self._presentation_speech_detector is not None

        frame_interval = 1.0 / self._config.target_fps
        frame_idx = 0

        # Crossfade state — tracks previous composite for smooth transitions
        _last_composite: np.ndarray | None = None
        _last_state_for_fade: PipelineState = self.state

        # Track clip state for logging transitions
        last_virtual_audio_underflows = 0
        _last_clip_playing = False

        while self._running:
            t_start = time.monotonic()
            audio_stage_ms = 0.0
            compose_ms = 0.0
            vcam_ms = 0.0
            preview_ms = 0.0

            # Handle panic toggle requested from the hotkey thread
            if self._panic_toggle_requested.is_set():
                self._panic_toggle_requested.clear()
                self._do_panic_toggle(_last_composite)

            idle_cooldown_reason = self._consume_pending_idle_cooldown()
            if idle_cooldown_reason is not None:
                self._apply_idle_cooldown(idle_cooldown_reason)

            t_audio_stage = time.monotonic()
            chunk_size = self._latest_audio_chunk_size

            if self._virtual_audio is not None:
                underflows = self._virtual_audio.underflow_count
                if underflows != last_virtual_audio_underflows:
                    logger.warning(
                        "Virtual audio underflow count increased to %d.",
                        underflows,
                    )
                    last_virtual_audio_underflows = underflows

            # Log idle clip state transitions
            clip_now_playing = (
                self._idle_processor is not None
                and self._idle_processor.is_clip_playing
            )
            if clip_now_playing != _last_clip_playing:
                if clip_now_playing:
                    logger.info("Idle clip STARTED playing")
                else:
                    self._release_presentation_hold("idle clip ended")
                    logger.info(
                        "Idle clip ENDED (queued=%.2f s, state=%s)",
                        self._audio_delay_buffer.available_samples / self._config.sample_rate,
                        self._state.name,
                    )
                _last_clip_playing = clip_now_playing

            self._sync_state_to_presentation()
            audio_stage_ms = (time.monotonic() - t_audio_stage) * 1000

            # 3. Compose output frame
            current_state = self.state
            t_compose = time.monotonic()

            if self._panic_crossfade is not None and self._panic_crossfade.is_active:
                # Mid-crossfade (entering or exiting panic): play blended frames.
                # While entering, also advance the webcam reader so it doesn't stall.
                if self._panic_mode and self._panic_webcam is not None:
                    webcam_raw = self._panic_webcam.read()
                    if webcam_raw is not None:
                        if self._config.panic_webcam_mirror:
                            webcam_raw = cv2.flip(webcam_raw, 1)
                        self._panic_last_webcam_frame = self._resize_to_output(webcam_raw)
                output_frame = self._panic_crossfade.next_frame()

            elif self._panic_mode:
                # Fully in panic mode — show live webcam
                webcam_raw = self._panic_webcam.read() if self._panic_webcam else None
                if webcam_raw is not None:
                    if self._config.panic_webcam_mirror:
                        webcam_raw = cv2.flip(webcam_raw, 1)
                    webcam_raw = self._resize_to_output(webcam_raw)
                    self._panic_last_webcam_frame = webcam_raw
                    if self._panic_needs_enter_crossfade and self._panic_activation_frame is not None:
                        # First real webcam frame — kick off enter crossfade
                        self._panic_crossfade.start(self._panic_activation_frame, webcam_raw)
                        self._panic_needs_enter_crossfade = False
                        output_frame = self._panic_crossfade.next_frame()
                    else:
                        output_frame = webcam_raw
                else:
                    if self._panic_last_webcam_frame is not None:
                        output_frame = self._panic_last_webcam_frame
                    elif _last_composite is not None:
                        output_frame = _last_composite
                    else:
                        output_frame = self._get_base_frame()

            else:
                # Normal avatar path
                output_frame = self._compose_frame(current_state, frame_idx)

                # 3b. Crossfade on IDLE↔SPEAKING state transitions
                if (
                    self._transition is not None
                    and current_state != _last_state_for_fade
                    and _last_composite is not None
                ):
                    self._transition.start(_last_composite, output_frame)
                    logger.debug(
                        "Crossfade: %s → %s", _last_state_for_fade.name, current_state.name
                    )
                if self._transition is not None and self._transition.is_active:
                    output_frame = self._transition.next_frame()

            _last_state_for_fade = current_state
            _last_composite = output_frame
            compose_ms = (time.monotonic() - t_compose) * 1000
            frame_idx += 1

            # 4. Send to virtual camera (skip if unavailable / preview-only mode)
            if self._virtual_camera is not None:
                t_vcam = time.monotonic()
                self._virtual_camera.send(output_frame)
                vcam_ms = (time.monotonic() - t_vcam) * 1000

            # 5. Update preview window (main thread required on Windows)
            if self._preview is not None:
                t_preview = time.monotonic()
                still_open = self._preview.update(output_frame)
                preview_ms = (time.monotonic() - t_preview) * 1000
                if not still_open:
                    self._preview = None

            # 6. Maintain frame rate + timing diagnostics
            elapsed = time.monotonic() - t_start
            sleep_s = frame_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
                playback_frame_s = elapsed + sleep_s
            else:
                playback_frame_s = elapsed

            # Log frame timing every 2 seconds and warn on overruns.
            if frame_idx % 50 == 0 or elapsed > (frame_interval * 2.0):
                actual_fps = 1.0 / elapsed if elapsed > 0 else 999
                queued_s = 0.0
                if self._audio_delay_buffer is not None:
                    queued_s = self._audio_delay_buffer.available_samples / self._config.sample_rate
                live_spk = bool(self._speech_detector.is_speaking) if self._speech_detector else False
                pres_spk = bool(self._presentation_speech_detector.is_speaking) if self._presentation_speech_detector else False
                clip_active = bool(self._idle_processor.is_clip_playing) if self._idle_processor else False
                q_drops = getattr(self._microphone, "queue_drop_count", 0) if self._microphone else 0
                log_fn = logger.warning if elapsed > (frame_interval * 2.0) else logger.info
                log_fn(
                    "Frame %d: total=%.1f ms (%.1f fps) state=%s chunk=%d queued=%.2f s "
                    "live_spk=%s pres_spk=%s clip=%s q_drops=%d "
                    "stages[audio=%.1f compose=%.1f vcam=%.1f preview=%.1f]",
                    frame_idx,
                    elapsed * 1000,
                    actual_fps,
                    current_state.name,
                    chunk_size,
                    queued_s,
                    live_spk,
                    pres_spk,
                    clip_active,
                    q_drops,
                    audio_stage_ms,
                    compose_ms,
                    vcam_ms,
                    preview_ms,
                )

    # ── Panic button ──────────────────────────────────────────────────────────

    def _preload_panic_webcam(self) -> None:
        """Open the panic webcam in the background at startup so activation is instant."""
        device = self._config.panic_webcam_device
        if device is None:
            return
        from scrumsurvivor.capture.webcam import WebcamCapture
        try:
            webcam_device = int(device) if str(device).isdigit() else device
            cam = WebcamCapture(
                device=webcam_device,
                target_fps=self._config.target_fps,
                backend=self._config.panic_webcam_backend,
            )
            cam.open()
            self._panic_webcam = cam
            logger.info(
                "Panic webcam pre-opened (device=%r) — panic button will switch instantly.",
                device,
            )
        except Exception:
            logger.warning(
                "Could not pre-open panic webcam (device=%r). "
                "Panic mode will still work but the switch may take a moment.",
                device,
                exc_info=True,
            )
            self._panic_webcam = None

    def _register_panic_hotkey(self) -> None:
        """Register the global panic hotkey using pynput.GlobalHotKeys."""
        hotkey_str = self._config.panic_hotkey
        if not hotkey_str:
            return

        def _to_pynput(hotkey: str) -> str:
            """Convert 'ctrl+shift+p' to pynput '<ctrl>+<shift>+p' format."""
            _MODIFIERS = {"ctrl", "shift", "alt", "cmd", "super"}
            parts = [p.strip() for p in hotkey.lower().split("+")]
            return "+".join(f"<{p}>" if p in _MODIFIERS else p for p in parts)

        pynput_hotkey = _to_pynput(hotkey_str)
        try:
            from pynput import keyboard as _kb
            self._panic_hotkey_listener = _kb.GlobalHotKeys(
                {pynput_hotkey: self._on_panic_hotkey}
            )
            self._panic_hotkey_listener.start()
            logger.info(
                "Panic button registered: %s — press to toggle real webcam + zero-delay audio",
                hotkey_str,
            )
        except Exception:
            logger.exception(
                "Failed to register panic hotkey %r — panic button disabled.", hotkey_str
            )
            self._panic_hotkey_listener = None

    def _on_panic_hotkey(self) -> None:
        """Called by pynput on its thread — signal the main loop to toggle."""
        now = time.monotonic()
        if now < self._panic_toggle_cooldown_until:
            return
        self._panic_toggle_cooldown_until = now + 2.0
        self._panic_toggle_requested.set()

    def _do_panic_toggle(self, current_frame: np.ndarray | None) -> None:
        """Called from the main loop to activate or deactivate panic mode."""
        if not self._panic_mode:
            # ── Activate panic ────────────────────────────────────────────────
            if self._panic_webcam is None:
                # Not pre-loaded (no device configured or pre-open failed) — try now
                device = self._config.panic_webcam_device
                if device is None:
                    logger.warning(
                        "Panic button pressed but panic_webcam_device is not configured. "
                        "Run the setup wizard and select a webcam for the panic button."
                    )
                    return
                from scrumsurvivor.capture.webcam import WebcamCapture
                try:
                    webcam_device = int(device) if str(device).isdigit() else device
                    cam = WebcamCapture(
                        device=webcam_device,
                        target_fps=self._config.target_fps,
                        backend=self._config.panic_webcam_backend,
                    )
                    cam.open()
                    self._panic_webcam = cam
                except Exception:
                    logger.exception(
                        "Failed to open panic webcam device=%r — panic mode NOT activated.", device
                    )
                    return

            # Audio: bypass the delay buffer for zero-latency passthrough
            if self._audio_delay_buffer is not None:
                self._audio_delay_buffer.set_passthrough(True)
                logger.info("Panic audio: zero-delay passthrough ENABLED")

            # Capture the last avatar frame to use as the crossfade start
            self._panic_activation_frame = (
                current_frame.copy() if current_frame is not None else self._get_base_frame()
            )
            self._panic_needs_enter_crossfade = True
            self._panic_mode = True
            logger.info("PANIC MODE: activated — real webcam feed + zero-delay audio")

        else:
            # ── Deactivate panic ──────────────────────────────────────────────
            webcam_snapshot = self._panic_last_webcam_frame
            # Compose a fresh avatar frame to crossfade back to
            avatar_frame = self._compose_frame(self.state, 0)

            self._panic_mode = False

            # Restore audio delay
            if self._audio_delay_buffer is not None:
                self._audio_delay_buffer.set_passthrough(False)
                logger.info("Panic audio: zero-delay passthrough DISABLED — normal delay restored")

            # Keep the panic webcam open so the next activation is instant too.
            # Only reset the last-frame cache so the old frame isn't shown on re-entry.
            self._panic_last_webcam_frame = None

            # Start crossfade from webcam snapshot back to avatar
            if self._panic_crossfade is not None and webcam_snapshot is not None:
                self._panic_crossfade.start(webcam_snapshot, avatar_frame)

            logger.info("PANIC MODE: deactivated — avatar resumed")

    def _resize_to_output(self, frame: np.ndarray) -> np.ndarray:
        """Resize *frame* to the configured output resolution."""
        import cv2
        w, h = self._config.output_resolution
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        return frame

    def _compose_frame(self, state: PipelineState, frame_idx: int = 0) -> np.ndarray:
        """Produce the composited output frame for *state*."""
        w, h = self._config.output_resolution
        base_frame = self._get_base_frame()

        if state in (PipelineState.SPEAKING, PipelineState.SPEAKING_LIVE, PipelineState.SPEAKING_BUFFERED) and self._lipsync_engine is not None:
            overlay = self._compose_lipsync_scheduled(frame_idx)
        elif state == PipelineState.IDLE and self._idle_processor is not None:
            overlay = self._idle_processor.process(base_frame)
        else:
            overlay = base_frame

        # Composite onto background
        if self._frame_compositor is not None:
            return self._frame_compositor.compose(overlay)

        import cv2
        return cv2.resize(overlay, (w, h))

    # ── Lipsync helpers ─────────────────────────────────────────────────────

    def _get_base_frame(self) -> np.ndarray:
        """Return a copy of the static base photo (or black fallback)."""
        if self._base_photo is not None:
            return self._base_photo.copy()
        return np.zeros(
            (self._config.output_resolution[1], self._config.output_resolution[0], 3),
            dtype=np.uint8,
        )

    def _compose_lipsync_scheduled(self, frame_idx: int) -> np.ndarray:
        """Lip sync driven by the same scheduled audio timeline sent to VB-Cable."""
        base_frame = self._get_speaking_base_frame()
        if (
            self._audio_preprocessor is None
            or self._face_crop_manager is None
            or self._lipsync_engine is None
            or self._audio_delay_buffer is None
        ):
            return base_frame

        recent_samples = int(self._config.sample_rate * 1.0)
        audio_recent = self._audio_delay_buffer.recent_output_window(recent_samples)
        if len(audio_recent) < 1024 or not np.any(np.abs(audio_recent) > 1e-6):
            logger.debug(
                "Lipsync: no presentable audio in recent window (len=%d, max_abs=%.6f) — showing base frame",
                len(audio_recent),
                float(np.max(np.abs(audio_recent))) if len(audio_recent) > 0 else 0.0,
            )
            return base_frame

        return self._run_lipsync_inference(base_frame, audio_recent)

    def _compose_lipsync_live(self, frame_idx: int) -> np.ndarray:
        return self._compose_lipsync_scheduled(frame_idx)

    def _compose_lipsync_buffered(self, frame_idx: int) -> np.ndarray:
        return self._compose_lipsync_scheduled(frame_idx)

    def _run_lipsync_inference(
        self, base_frame: np.ndarray, audio_recent: np.ndarray
    ) -> np.ndarray:
        """Run mel extraction + Wav2Lip inference on *audio_recent* and paste back."""
        mel = self._audio_preprocessor.to_mel(audio_recent)
        T = mel.shape[1]
        if T < 16:
            logger.debug("_run_lipsync_inference: mel too short (T=%d)", T)
            return base_frame

        # Always take the last 16 mel frames = the most recent speech content
        mel_window = mel[:, T - 16 : T][np.newaxis]  # (1, 80, 16)

        # Get face crop from base photo
        face_crop = self._face_crop_manager.get_crop(base_frame)
        if face_crop is None:
            return base_frame

        # Run inference
        synced_face = self._lipsync_engine.process(face_crop, mel_window)

        # Paste synced face back onto base frame
        return self._face_crop_manager.paste_back(base_frame, synced_face)

