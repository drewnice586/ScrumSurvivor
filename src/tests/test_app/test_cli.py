"""Tests for the CLI commands (non-hardware)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from unittest.mock import patch

from scrumsurvivor.audio.device_selector import format_output_device_selector


def _write_theme_assets(root: Path, *theme_names: str) -> None:
    for theme_name in theme_names:
        theme_dir = root / "assets" / "themes" / theme_name
        theme_dir.mkdir(parents=True, exist_ok=True)
        (theme_dir / "base_photo.png").write_bytes(b"\xff")


def test_generate_config_creates_file(tmp_path):
    from scrumsurvivor.cli import generate_config

    config_path = str(tmp_path / "config.yaml")
    runner = CliRunner()
    result = runner.invoke(generate_config, ["--config", config_path])
    assert result.exit_code == 0
    assert (tmp_path / "config.yaml").exists()


def test_generate_config_abort_if_exists(tmp_path):
    from scrumsurvivor.cli import generate_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text("existing")
    runner = CliRunner()
    # Answer 'n' to overwrite prompt
    result = runner.invoke(generate_config, ["--config", str(config_path)], input="n\n")
    assert result.exit_code != 0
    # Original contents should still be there
    assert config_path.read_text() == "existing"


def test_check_gpu_exits_1_when_no_cuda():
    from scrumsurvivor.cli import check_gpu
    from unittest.mock import patch
    from scrumsurvivor.lipsync.gpu_check import GPUReport

    insufficient_report = GPUReport(
        available=False, device_name="", vram_total_gb=0,
        vram_free_gb=0, cuda_version="N/A", sufficient=False, warning="No CUDA."
    )
    runner = CliRunner()
    with patch("scrumsurvivor.lipsync.gpu_check.check_gpu", return_value=insufficient_report):
        result = runner.invoke(check_gpu, [])
    assert result.exit_code == 1


def test_validate_assets_exits_1_when_missing(tmp_path):
    from scrumsurvivor.cli import validate_assets
    from scrumsurvivor.config.settings import AppConfig, save_config

    # Point to paths that genuinely don't exist
    cfg = AppConfig(
        base_photo_path=str(tmp_path / "missing_base.png"),
    )
    config_path = str(tmp_path / "config.yaml")
    save_config(cfg, config_path)
    runner = CliRunner()
    result = runner.invoke(validate_assets, ["--config", config_path])
    # Assets don't exist → should exit 1
    assert result.exit_code == 1


def test_validate_assets_ok_when_present(tmp_path):
    from scrumsurvivor.cli import validate_assets
    from scrumsurvivor.config.settings import AppConfig, save_config

    # Create fake asset files
    (tmp_path / "base.png").write_bytes(b"\xff")
    cfg = AppConfig(
        base_photo_path=str(tmp_path / "base.png"),
    )
    config_path = str(tmp_path / "config.yaml")
    save_config(cfg, config_path)

    runner = CliRunner()
    result = runner.invoke(validate_assets, ["--config", config_path])
    assert result.exit_code == 0


def test_startup_theme_prompt_uses_configured_theme_as_default(tmp_path, monkeypatch):
    from scrumsurvivor.cli import _maybe_select_theme_for_startup
    from scrumsurvivor.config.settings import AppConfig

    monkeypatch.chdir(tmp_path)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "base_photo.png").write_bytes(b"\xff")
    _write_theme_assets(tmp_path, "business", "casual")

    prompted: dict[str, object] = {}

    def fake_prompt(options: list[str | None], default_theme: str | None = None) -> str | None:
        prompted["options"] = options
        prompted["default_theme"] = default_theme
        return "business"

    monkeypatch.setattr("scrumsurvivor.cli._prompt_theme_selection", fake_prompt)

    cfg = AppConfig(active_theme="casual")
    _maybe_select_theme_for_startup(cfg, prompt_theme=True)

    assert cfg.active_theme == "business"
    assert prompted["options"] == [None, "business", "casual"]
    assert prompted["default_theme"] == "casual"


def test_startup_theme_prompt_handles_missing_default_assets(tmp_path, monkeypatch):
    from scrumsurvivor.cli import _maybe_select_theme_for_startup
    from scrumsurvivor.config.settings import AppConfig

    monkeypatch.chdir(tmp_path)
    _write_theme_assets(tmp_path, "business", "casual")

    prompted: dict[str, object] = {}

    def fake_prompt(options: list[str | None], default_theme: str | None = None) -> str | None:
        prompted["options"] = options
        prompted["default_theme"] = default_theme
        return "casual"

    monkeypatch.setattr("scrumsurvivor.cli._prompt_theme_selection", fake_prompt)

    cfg = AppConfig()
    _maybe_select_theme_for_startup(cfg, prompt_theme=False)

    assert cfg.active_theme == "casual"
    assert prompted["options"] == ["business", "casual"]
    assert prompted["default_theme"] is None


def test_startup_theme_prompt_skips_prompt_when_only_one_option(tmp_path, monkeypatch):
    from scrumsurvivor.cli import _maybe_select_theme_for_startup
    from scrumsurvivor.config.settings import AppConfig

    monkeypatch.chdir(tmp_path)
    _write_theme_assets(tmp_path, "casual")

    def fail_prompt(*args, **kwargs):
        raise AssertionError("prompt should not be called when only one asset option exists")

    monkeypatch.setattr("scrumsurvivor.cli._prompt_theme_selection", fail_prompt)

    cfg = AppConfig(active_theme="business")
    _maybe_select_theme_for_startup(cfg, prompt_theme=True)

    assert cfg.active_theme == "casual"


def test_setup_writes_selected_hardware_and_threshold(tmp_path):
    from scrumsurvivor.cli import setup
    from scrumsurvivor.config.settings import load_config
    from scrumsurvivor.setup_wizard import DeviceOption

    config_path = str(tmp_path / "config.yaml")
    runner = CliRunner()

    with (
        patch(
            "scrumsurvivor.setup_wizard.list_microphone_options",
            return_value=[
                DeviceOption(value=None, label="System default", device_index=1, default_sample_rate=48000),
                DeviceOption(value=4, label="Microphone (Logitech BRIO)", device_index=4, default_sample_rate=48000),
            ],
        ),
        patch(
            "scrumsurvivor.setup_wizard.list_virtual_audio_output_options",
            return_value=[
                DeviceOption(
                    value=format_output_device_selector(
                        "CABLE Input (VB-Audio Virtual Cable)",
                        hostapi_name="Windows WASAPI",
                        device_id=25,
                    ),
                    label="CABLE Input (VB-Audio Virtual Cable) [Windows WASAPI]",
                    device_index=25,
                    default_sample_rate=48000,
                ),
            ],
        ),
        patch("scrumsurvivor.setup_wizard.resolve_sample_rate", return_value=48000),
        patch("scrumsurvivor.setup_wizard.tune_speech_threshold", return_value=0.0375),
    ):
        result = runner.invoke(
            setup,
            ["--config", config_path],
            input="2\n1\ny\ny\n",
        )

    assert result.exit_code == 0
    cfg = load_config(config_path)
    assert cfg.microphone_device == 4
    assert cfg.virtual_audio_device == format_output_device_selector(
        "CABLE Input (VB-Audio Virtual Cable)",
        hostapi_name="Windows WASAPI",
        device_id=25,
    )
    assert cfg.sample_rate == 48000
    assert cfg.preview_enabled is True
    assert cfg.speech_threshold == pytest.approx(0.0375)
