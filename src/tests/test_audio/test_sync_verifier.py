"""Tests for SyncVerifier."""

from __future__ import annotations

import pytest


def test_in_sync_when_delays_match():
    from scrumsurvivor.audio.sync_verifier import SyncVerifier

    verifier = SyncVerifier(audio_delay_ms=150.0, wav2lip_avg_latency_ms=150.0)
    result = verifier.run_test()
    assert result["in_sync"] is True
    assert result["offset_ms"] == pytest.approx(0.0)


def test_not_in_sync_when_large_offset():
    from scrumsurvivor.audio.sync_verifier import SyncVerifier

    verifier = SyncVerifier(audio_delay_ms=300.0, wav2lip_avg_latency_ms=150.0)
    result = verifier.run_test()
    assert result["in_sync"] is False
    assert result["offset_ms"] == pytest.approx(150.0)


def test_recommendation_provided_when_not_in_sync():
    from scrumsurvivor.audio.sync_verifier import SyncVerifier

    verifier = SyncVerifier(audio_delay_ms=300.0, wav2lip_avg_latency_ms=100.0)
    result = verifier.run_test()
    assert result["recommendation"] is not None


def test_recommendation_none_when_in_sync():
    from scrumsurvivor.audio.sync_verifier import SyncVerifier

    verifier = SyncVerifier(audio_delay_ms=150.0, wav2lip_avg_latency_ms=140.0)
    result = verifier.run_test()
    assert result["in_sync"] is True
    assert result["recommendation"] is None
