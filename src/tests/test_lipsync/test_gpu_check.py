"""Tests for GPU detection (non-GPU — mocked torch module)."""

from __future__ import annotations

import sys
import types
import pytest
from unittest.mock import MagicMock, patch


def _build_torch_mock(available: bool = True, total_gb: float = 8.0, free_gb: float = 6.0):
    """Build a minimal mock of the torch module with cuda sub-module."""
    torch_mock = MagicMock(name="torch")

    props = MagicMock()
    props.total_memory = int(total_gb * 1024**3)

    torch_mock.cuda.is_available.return_value = available
    torch_mock.cuda.get_device_name.return_value = "NVIDIA Test GPU"
    torch_mock.cuda.get_device_properties.return_value = props
    torch_mock.cuda.mem_get_info.return_value = (
        int(free_gb * 1024**3),
        int(total_gb * 1024**3),
    )
    torch_mock.version.cuda = "11.8"
    return torch_mock


def test_no_cuda_returns_insufficient():
    from scrumsurvivor.lipsync.gpu_check import check_gpu

    torch_mock = _build_torch_mock(available=False)
    with patch.dict(sys.modules, {"torch": torch_mock}):
        report = check_gpu()

    assert report.available is False
    assert report.sufficient is False
    assert report.warning is not None


@pytest.mark.parametrize("total_gb,expected_sufficient", [
    (2.0, False),
    (3.0, False),
    (4.0, True),
    (8.0, True),
])
def test_vram_sufficiency(total_gb, expected_sufficient):
    from scrumsurvivor.lipsync.gpu_check import check_gpu

    torch_mock = _build_torch_mock(available=True, total_gb=total_gb, free_gb=total_gb / 2)
    with patch.dict(sys.modules, {"torch": torch_mock}):
        report = check_gpu()

    assert report.sufficient == expected_sufficient


def test_marginal_vram_has_warning():
    from scrumsurvivor.lipsync.gpu_check import check_gpu

    torch_mock = _build_torch_mock(available=True, total_gb=4.0, free_gb=2.0)
    with patch.dict(sys.modules, {"torch": torch_mock}):
        report = check_gpu()

    assert report.sufficient is True
    assert report.warning is not None


def test_require_gpu_raises_when_insufficient():
    from scrumsurvivor.lipsync.gpu_check import GPUReport, require_gpu

    report = GPUReport(
        available=False,
        device_name="",
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        cuda_version="N/A",
        sufficient=False,
        warning="No CUDA.",
    )
    with pytest.raises(RuntimeError, match="requires a CUDA"):
        require_gpu(report)


def test_require_gpu_does_not_raise_when_sufficient():
    from scrumsurvivor.lipsync.gpu_check import GPUReport, require_gpu

    report = GPUReport(
        available=True,
        device_name="RTX 3050",
        vram_total_gb=8.0,
        vram_free_gb=6.0,
        cuda_version="11.8",
        sufficient=True,
        warning=None,
    )
    require_gpu(report)  # should not raise
