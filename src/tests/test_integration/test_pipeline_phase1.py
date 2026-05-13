"""Phase 1 pipeline integration tests."""

from __future__ import annotations

from contextlib import ExitStack
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_config(**overrides):
    from scrumsurvivor.config.settings import AppConfig
    return AppConfig(**overrides)


def test_pipeline_state_starts_idle():
    from scrumsurvivor.pipeline import Pipeline, PipelineState

    pipeline = Pipeline(_make_config())
    assert pipeline.state == PipelineState.IDLE


def test_compose_frame_idle_passthrough():
    """With no idle processor, the static base photo is used."""
    from scrumsurvivor.pipeline import Pipeline, PipelineState

    pipeline = Pipeline(_make_config())
    base = np.full((720, 1280, 3), 42, dtype=np.uint8)
    pipeline.set_base_photo(base)

    result = pipeline._compose_frame(PipelineState.IDLE)
    assert np.array_equal(result, base)


def test_compose_frame_uses_idle_processor():
    """If idle_processor is set, it should be called in IDLE state."""
    from scrumsurvivor.pipeline import Pipeline, PipelineState

    pipeline = Pipeline(_make_config())

    processed = np.full((720, 1280, 3), 99, dtype=np.uint8)
    mock_processor = MagicMock()
    mock_processor.process.return_value = processed
    pipeline.set_idle_processor(mock_processor)
    base = np.full((720, 1280, 3), 33, dtype=np.uint8)
    pipeline.set_base_photo(base)

    result = pipeline._compose_frame(PipelineState.IDLE)

    mock_processor.process.assert_called_once()
    assert np.array_equal(mock_processor.process.call_args.args[0], base)
    assert np.array_equal(result, processed)


def test_compose_frame_uses_lipsync_when_speaking():
    """In SPEAKING state, scheduled lipsync output is returned."""
    from scrumsurvivor.pipeline import Pipeline, PipelineState

    pipeline = Pipeline(_make_config())

    synced = np.full((720, 1280, 3), 77, dtype=np.uint8)
    mock_engine = MagicMock()
    pipeline.set_lipsync_engine(mock_engine)

    with patch.object(pipeline, "_compose_lipsync_scheduled", return_value=synced) as mock_lipsync:
        result = pipeline._compose_frame(PipelineState.SPEAKING)
        mock_lipsync.assert_called_once()
    assert np.array_equal(result, synced)


def test_audio_output_loop_relies_on_blocking_write_without_manual_sleep():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())

    class _VirtualAudio:
        blocksize = 512

        def __init__(self, stop_event):
            self.write_calls = 0
            self._stop_event = stop_event

        def write(self, audio_chunk):
            self.write_calls += 1
            self._stop_event.set()
            return False

    class _DelayBuffer:
        available_samples = 0
        is_primed = False

        def pull(self, num_samples):
            return np.zeros(num_samples, dtype=np.float32)

    pipeline._virtual_audio = _VirtualAudio(pipeline._audio_output_stop)
    pipeline._audio_delay_buffer = _DelayBuffer()

    with patch("scrumsurvivor.pipeline.time.sleep") as mock_sleep:
        pipeline._audio_output_loop()

    assert pipeline._virtual_audio.write_calls == 1
    mock_sleep.assert_not_called()


def test_start_audio_output_thread_skips_when_callback_output_is_active():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())

    class _VirtualAudio:
        is_callback_driven = True
        blocksize = 512

    pipeline._virtual_audio = _VirtualAudio()
    pipeline._audio_delay_buffer = object()

    with patch("scrumsurvivor.pipeline.threading.Thread") as mock_thread:
        pipeline._start_audio_output_thread()

    mock_thread.assert_not_called()


def test_open_audio_transport_prefers_duplex_when_available():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    duplex = MagicMock()
    duplex.__enter__.return_value = duplex
    duplex.__exit__.return_value = False

    with (
        patch("scrumsurvivor.audio.duplex_transport.DuplexAudioTransport", return_value=duplex),
        patch("scrumsurvivor.capture.microphone.MicrophoneCapture") as mock_mic,
        ExitStack() as stack,
    ):
        mic, output = pipeline._open_audio_transport(stack)

    assert mic is duplex
    assert output is duplex
    mock_mic.assert_not_called()


def test_open_audio_transport_falls_back_to_split_when_duplex_fails():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    duplex = MagicMock()
    duplex.__enter__.side_effect = RuntimeError("duplex unavailable")
    duplex.__exit__.return_value = False

    mic = MagicMock()
    mic.__enter__.return_value = mic
    mic.__exit__.return_value = False

    with (
        patch("scrumsurvivor.audio.duplex_transport.DuplexAudioTransport", return_value=duplex),
        patch("scrumsurvivor.capture.microphone.MicrophoneCapture", return_value=mic),
        ExitStack() as stack,
    ):
        selected_mic, output = pipeline._open_audio_transport(stack)

    assert selected_mic is mic
    assert output is None


def test_process_microphone_chunk_schedules_audio_without_direct_output_push():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.threshold = 0.5
    pipeline._speech_detector.is_speaking = True
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=False)

    chunk = np.ones(512, dtype=np.float32)
    pipeline._process_microphone_chunk(chunk)

    pipeline._audio_delay_buffer.push_chunk.assert_called_once_with(chunk)
    pipeline._audio_delay_buffer.begin_hold.assert_not_called()


def test_process_microphone_chunk_starts_hold_when_speech_begins_during_clip():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.threshold = 0.5
    pipeline._speech_detector.is_speaking = True
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=True)

    chunk = np.ones(512, dtype=np.float32)
    pipeline._process_microphone_chunk(chunk)

    pipeline._audio_delay_buffer.begin_hold.assert_called_once()
    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(True)
    pipeline._audio_delay_buffer.push_chunk.assert_called_once_with(chunk)


def test_process_microphone_chunk_starts_hold_for_idle_started_blink_clip():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.threshold = 0.5
    pipeline._speech_detector.is_speaking = True
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=True)
    type(pipeline._idle_processor).current_clip_allows_speaking_overlay = PropertyMock(return_value=False)

    chunk = np.ones(512, dtype=np.float32)
    pipeline._process_microphone_chunk(chunk)

    pipeline._audio_delay_buffer.begin_hold.assert_called_once()
    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(True)
    pipeline._audio_delay_buffer.push_chunk.assert_called_once_with(chunk)


def test_process_microphone_chunk_skips_hold_for_speaking_overlay_blink_clip():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.threshold = 0.5
    pipeline._speech_detector.is_speaking = True
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=True)
    type(pipeline._idle_processor).current_clip_allows_speaking_overlay = PropertyMock(return_value=True)

    chunk = np.ones(512, dtype=np.float32)
    pipeline._process_microphone_chunk(chunk)

    pipeline._audio_delay_buffer.begin_hold.assert_not_called()
    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(True)
    pipeline._audio_delay_buffer.push_chunk.assert_called_once_with(chunk)


def test_get_speaking_base_frame_uses_dedicated_blink_frame_reader():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    base = np.full((20, 20, 3), 10, dtype=np.uint8)
    blink_frame = np.full((20, 20, 3), 90, dtype=np.uint8)
    pipeline.set_base_photo(base)
    pipeline._idle_processor = MagicMock()
    pipeline._idle_processor.speaking_base_frame.return_value = blink_frame
    pipeline._idle_processor.process.return_value = np.full((20, 20, 3), 33, dtype=np.uint8)

    result = pipeline._get_speaking_base_frame()

    pipeline._idle_processor.speaking_base_frame.assert_called_once()
    assert np.array_equal(pipeline._idle_processor.speaking_base_frame.call_args.args[0], base)
    pipeline._idle_processor.process.assert_not_called()
    assert np.array_equal(result, blink_frame)


def test_get_speaking_base_frame_defaults_to_static_base_without_blink_frame_reader():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    base = np.full((20, 20, 3), 10, dtype=np.uint8)
    pipeline.set_base_photo(base)
    pipeline._idle_processor = MagicMock()

    del pipeline._idle_processor.speaking_base_frame
    result = pipeline._get_speaking_base_frame()

    pipeline._idle_processor.process.assert_not_called()
    assert np.array_equal(result, base)


def test_process_microphone_chunk_always_pushes_real_audio():
    """All mic chunks are pushed as-is regardless of speech state."""
    from scrumsurvivor.pipeline import Pipeline

    class _Detector:
        def __init__(self) -> None:
            self.threshold = 0.5
            self.is_speaking = False
            self._calls = 0

        def update(self, _chunk) -> None:
            self._calls += 1
            if self._calls >= 2:
                self.is_speaking = True

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = _Detector()
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    type(pipeline._audio_delay_buffer).available_samples = PropertyMock(return_value=0)
    type(pipeline._audio_delay_buffer).base_delay_samples = PropertyMock(return_value=0)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=False)
    pipeline._presentation_speech_detector = MagicMock()
    pipeline._presentation_speech_detector.is_speaking = False

    first_chunk = np.full(4, 0.8, dtype=np.float32)
    second_chunk = np.full(4, 1.0, dtype=np.float32)

    pipeline._process_microphone_chunk(first_chunk)
    pipeline._process_microphone_chunk(second_chunk)

    # Both chunks pushed as real audio (no silence gating)
    assert pipeline._audio_delay_buffer.push_chunk.call_count == 2
    pushed_first = pipeline._audio_delay_buffer.push_chunk.call_args_list[0].args[0]
    pushed_second = pipeline._audio_delay_buffer.push_chunk.call_args_list[1].args[0]
    assert np.array_equal(pushed_first, first_chunk)
    assert np.array_equal(pushed_second, second_chunk)


def test_process_microphone_chunk_pushes_real_audio_even_for_clicks():
    """Mouse clicks and transient sounds are pushed as real audio (no gating)."""
    from scrumsurvivor.pipeline import Pipeline

    class _Detector:
        def __init__(self) -> None:
            self.threshold = 0.5
            self.is_speaking = False

        def update(self, _chunk) -> None:
            return None

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = _Detector()
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    type(pipeline._audio_delay_buffer).available_samples = PropertyMock(return_value=0)
    type(pipeline._audio_delay_buffer).base_delay_samples = PropertyMock(return_value=0)
    pipeline._idle_processor = MagicMock()
    type(pipeline._idle_processor).is_clip_playing = PropertyMock(return_value=False)
    pipeline._presentation_speech_detector = MagicMock()
    pipeline._presentation_speech_detector.is_speaking = False

    click_like_chunk = np.full(4, 1.0, dtype=np.float32)
    quiet_chunk = np.zeros(4, dtype=np.float32)

    pipeline._process_microphone_chunk(click_like_chunk)
    pipeline._process_microphone_chunk(quiet_chunk)

    assert pipeline._audio_delay_buffer.push_chunk.call_count == 2
    pushed_click = pipeline._audio_delay_buffer.push_chunk.call_args_list[0].args[0]
    pushed_quiet = pipeline._audio_delay_buffer.push_chunk.call_args_list[1].args[0]
    assert np.array_equal(pushed_click, click_like_chunk)
    assert np.array_equal(pushed_quiet, quiet_chunk)


def test_sync_state_to_presentation_releases_idle_clip_start_block_after_speech():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._presentation_speech_detector = MagicMock()
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.is_speaking = False
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    type(pipeline._audio_delay_buffer).available_samples = PropertyMock(return_value=0)
    type(pipeline._audio_delay_buffer).base_delay_samples = PropertyMock(return_value=0)
    pipeline._idle_processor = MagicMock()

    pipeline._presentation_speech_detector.is_speaking = False
    pipeline._sync_state_to_presentation()

    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(False)


def test_pending_pipeline_audio_blocks_idle_starts_after_hold_release():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.is_speaking = False
    pipeline._presentation_speech_detector = MagicMock()
    pipeline._presentation_speech_detector.is_speaking = False
    pipeline._idle_processor = MagicMock()
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    # Simulate excess queued audio after hold release: available far exceeds base delay
    type(pipeline._audio_delay_buffer).base_delay_samples = PropertyMock(return_value=96000)
    type(pipeline._audio_delay_buffer).available_samples = PropertyMock(return_value=240000)

    pipeline._update_idle_clip_priority_gate()

    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(True)


def test_idle_starts_unblocked_once_pending_audio_drains():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._speech_detector = MagicMock()
    pipeline._speech_detector.is_speaking = False
    pipeline._presentation_speech_detector = MagicMock()
    pipeline._presentation_speech_detector.is_speaking = False
    pipeline._idle_processor = MagicMock()
    pipeline._audio_delay_buffer = MagicMock()
    type(pipeline._audio_delay_buffer).hold_active = PropertyMock(return_value=False)
    # available_samples within normal base delay range — no excess
    type(pipeline._audio_delay_buffer).base_delay_samples = PropertyMock(return_value=96000)
    type(pipeline._audio_delay_buffer).available_samples = PropertyMock(return_value=96000)

    pipeline._update_idle_clip_priority_gate()

    pipeline._idle_processor.set_clip_starts_blocked.assert_called_once_with(False)


def test_sync_state_to_presentation_uses_presented_audio_detector():
    from scrumsurvivor.pipeline import Pipeline

    pipeline = Pipeline(_make_config())
    pipeline._presentation_speech_detector = MagicMock()

    pipeline._presentation_speech_detector.is_speaking = True
    pipeline._sync_state_to_presentation()
    assert pipeline.state.name == "SPEAKING"

    pipeline._presentation_speech_detector.is_speaking = False
    pipeline._sync_state_to_presentation()
    assert pipeline.state.name == "IDLE"
