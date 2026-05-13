"""GPU detection and capability check for Wav2Lip lipsync mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_VRAM_GB = 3.5   # refuse below this
_WARN_VRAM_GB = 5.0  # warn if below this but still attempt


@dataclass
class GPUReport:
    available: bool
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str
    sufficient: bool          # True if VRAM >= _MIN_VRAM_GB and CUDA available
    warning: str | None       # Warning message if marginal


def check_gpu() -> GPUReport:
    """Detect NVIDIA GPU via PyTorch CUDA and return a capability report."""
    try:
        import torch
    except ImportError:
        return GPUReport(
            available=False,
            device_name="",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="N/A",
            sufficient=False,
            warning="PyTorch is not installed.",
        )

    if not torch.cuda.is_available():
        return GPUReport(
            available=False,
            device_name="",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="N/A",
            sufficient=False,
            warning="No CUDA-capable GPU found. CPU inference is too slow for real-time lipsync.",
        )

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    free_bytes = torch.cuda.mem_get_info(0)[0]
    free_gb = free_bytes / 1024**3
    device_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda or "unknown"

    sufficient = total_gb >= _MIN_VRAM_GB
    warning: str | None = None
    if sufficient and total_gb < _WARN_VRAM_GB:
        warning = (
            f"GPU has {total_gb:.1f} GB VRAM — marginal for Wav2Lip. "
            "Reduce resolution or switch to standard model if you experience stuttering."
        )

    return GPUReport(
        available=True,
        device_name=device_name,
        vram_total_gb=total_gb,
        vram_free_gb=free_gb,
        cuda_version=cuda_ver,
        sufficient=sufficient,
        warning=warning,
    )


def require_gpu(report: GPUReport) -> None:
    """Raise ``RuntimeError`` if *report* shows the GPU is insufficient."""
    if not report.sufficient:
        msg = (
            "Lipsync mode requires a CUDA-capable GPU with at least "
            f"{_MIN_VRAM_GB:.0f} GB VRAM.\n"
            f"  available={report.available}, "
            f"device={report.device_name!r}, "
            f"vram={report.vram_total_gb:.1f} GB\n"
            f"  Hint: {report.warning or 'No CUDA GPU detected.'}"
        )
        raise RuntimeError(msg)
    if report.warning:
        logger.warning("GPU warning: %s", report.warning)


def print_gpu_report(report: GPUReport) -> None:
    """Pretty-print *report* to stdout."""
    print("── GPU Report ──────────────────────────────────")
    print(f"  Available    : {report.available}")
    print(f"  Device       : {report.device_name or 'N/A'}")
    print(f"  VRAM total   : {report.vram_total_gb:.2f} GB")
    print(f"  VRAM free    : {report.vram_free_gb:.2f} GB")
    print(f"  CUDA version : {report.cuda_version}")
    print(f"  Sufficient   : {report.sufficient}")
    if report.warning:
        print(f"  ⚠  {report.warning}")
    print("─────────────────────────────────────────────────")
