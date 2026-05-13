"""Tests for AppConfig / settings module."""

from __future__ import annotations

import pytest


def test_default_config_has_expected_values():
    from scrumsurvivor.config.settings import AppConfig

    cfg = AppConfig()
    assert cfg.target_fps == 25
    assert cfg.output_resolution == [1280, 720]
    assert cfg.log_level == "INFO"
    assert cfg.crossfade_frames == 15
    assert cfg.blink_interval_range == [4.0, 8.0]
    assert cfg.breathing_clip_interval_range == [10.0, 15.0]
    assert cfg.idle_clip_pause_min_s == 5.0
    assert cfg.idle_clip_pause_max_s == 10.0
    assert cfg.idle_after_speaking_cooldown_s == 5.0


def test_load_config_creates_file_if_missing(tmp_path):
    from scrumsurvivor.config.settings import load_config

    config_path = str(tmp_path / "config.yaml")
    cfg = load_config(config_path)
    assert (tmp_path / "config.yaml").exists()
    assert cfg.target_fps == 25


def test_save_and_load_roundtrip(tmp_path):
    from scrumsurvivor.config.settings import AppConfig, save_config, load_config

    cfg = AppConfig(target_fps=30, log_level="DEBUG")
    path = str(tmp_path / "config.yaml")
    save_config(cfg, path)

    loaded = load_config(path)
    assert loaded.target_fps == 30
    assert loaded.log_level == "DEBUG"


def test_load_ignores_unknown_keys(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"target_fps": 30, "unknown_key": "value"}))
    cfg = load_config(str(path))
    assert cfg.target_fps == 30
    assert not hasattr(cfg, "unknown_key")


def test_load_ignores_legacy_virtual_camera_backend_key(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"target_fps": 30, "virtual_camera_backend": None}))
    cfg = load_config(str(path))
    assert cfg.target_fps == 30
    assert not hasattr(cfg, "virtual_camera_backend")


def test_load_ignores_legacy_webcam_device_key(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"target_fps": 30, "webcam_device": 702}))
    cfg = load_config(str(path))
    assert cfg.target_fps == 30
    assert not hasattr(cfg, "webcam_device")


def test_validate_raises_on_invalid_fps(tmp_path):
    from scrumsurvivor.config.settings import AppConfig, save_config, load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"target_fps": -1}))
    with pytest.raises(ValueError, match="target_fps"):
        load_config(str(path))


def test_validate_raises_on_bad_log_level(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"log_level": "VERBOSE"}))
    with pytest.raises(ValueError, match="log_level"):
        load_config(str(path))


def test_validate_raises_on_negative_idle_cooldown(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"idle_after_speaking_cooldown_s": -0.1}))
    with pytest.raises(ValueError, match="idle_after_speaking_cooldown_s"):
        load_config(str(path))


def test_validate_raises_on_negative_idle_pause_min(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"idle_clip_pause_min_s": -0.1}))
    with pytest.raises(ValueError, match="idle_clip_pause_min_s"):
        load_config(str(path))


def test_validate_raises_on_bad_breathing_clip_interval_range(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"breathing_clip_interval_range": [15.0, 10.0]}))
    with pytest.raises(ValueError, match="breathing_clip_interval_range"):
        load_config(str(path))


def test_validate_raises_when_idle_pause_max_is_smaller_than_min(tmp_path):
    from scrumsurvivor.config.settings import load_config
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"idle_clip_pause_min_s": 10.0, "idle_clip_pause_max_s": 5.0}))
    with pytest.raises(ValueError, match="idle_clip_pause_max_s"):
        load_config(str(path))


def test_generate_default_config_creates_readable_yaml(tmp_path):
    from scrumsurvivor.config.settings import generate_default_config
    import yaml

    path = str(tmp_path / "config.yaml")
    generate_default_config(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    assert "virtual_camera_backend" not in data
    assert "webcam_device" not in data
    assert "target_fps" in data
    assert data["target_fps"] == 25
