<#
.SYNOPSIS
    Sets up or activates the ScrumSurvivor Python virtual environment and installs dependencies.

.DESCRIPTION
    - Creates .venv if it doesn't exist
    - Activates the virtual environment
    - Checks if all requirements.txt packages are satisfied
    - Installs missing packages via pip
#>

$ErrorActionPreference = "Stop"
$VenvDir = Join-Path $PSScriptRoot ".venv"
$RequirementsFile = Join-Path $PSScriptRoot "requirements.txt"

# --- Create venv if it doesn't exist ---
if (-not (Test-Path $VenvDir)) {
    Write-Host "[setup] Creating virtual environment in .venv ..." -ForegroundColor Cyan
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: Failed to create virtual environment. Is Python 3.10+ installed?" -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[setup] Virtual environment already exists." -ForegroundColor Green
}

# --- Activate the venv ---
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "[setup] ERROR: Activate script not found at $ActivateScript" -ForegroundColor Red
    exit 1
}

Write-Host "[setup] Activating virtual environment ..." -ForegroundColor Cyan
& $ActivateScript

# --- Verify Python version ---
$PythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$Major, $Minor = $PythonVersion -split '\.'
if ([int]$Major -lt 3 -or ([int]$Major -eq 3 -and [int]$Minor -lt 10)) {
    Write-Host "[setup] ERROR: Python 3.10+ required, found $PythonVersion" -ForegroundColor Red
    exit 1
}
Write-Host "[setup] Python version: $PythonVersion" -ForegroundColor Green

# --- Check and install requirements ---
if (-not (Test-Path $RequirementsFile)) {
    Write-Host "[setup] ERROR: requirements.txt not found at $RequirementsFile" -ForegroundColor Red
    exit 1
}

Write-Host "[setup] Checking installed packages against requirements.txt ..." -ForegroundColor Cyan

# Get currently installed packages
$InstalledRaw = pip freeze 2>$null
$InstalledPackages = @{}
foreach ($line in $InstalledRaw) {
    if ($line -match '^([^=]+)==(.+)$') {
        $InstalledPackages[$Matches[1].ToLower()] = $Matches[2]
    }
}

# Parse requirements and check for missing packages
$Missing = @()
foreach ($line in Get-Content $RequirementsFile) {
    $trimmed = $line.Trim()
    # Skip empty lines and comments
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    # Extract package name (before any version specifier)
    if ($trimmed -match '^([a-zA-Z0-9_-]+)') {
        $pkgName = $Matches[1].ToLower()
        if (-not $InstalledPackages.ContainsKey($pkgName)) {
            $Missing += $trimmed
        }
    }
}

if ($Missing.Count -gt 0) {
    Write-Host "[setup] Installing $($Missing.Count) missing package(s) ..." -ForegroundColor Yellow
    pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: pip install failed. Check output above." -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] All packages installed." -ForegroundColor Green
} else {
    Write-Host "[setup] All requirements already satisfied." -ForegroundColor Green
}

# --- Install PyTorch with CUDA (never use the PyPI CPU-only build) ---
Write-Host "[setup] Checking PyTorch CUDA support ..." -ForegroundColor Cyan

$torchCudaOk = $false
try {
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaAvailable -eq "True") {
        $torchVer = python -c "import torch; print(torch.__version__)" 2>$null
        Write-Host "[setup] PyTorch $torchVer already installed with CUDA support." -ForegroundColor Green
        $torchCudaOk = $true
    }
} catch {}

if (-not $torchCudaOk) {
    # Detect CUDA version from nvidia-smi
    $cudaTag = $null
    try {
        $smiLines = & nvidia-smi 2>$null
        foreach ($line in $smiLines) {
            if ($line -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $cudaMaj = [int]$Matches[1]
                if     ($cudaMaj -ge 12) { $cudaTag = "cu124" }
                elseif ($cudaMaj -eq 11) { $cudaTag = "cu118" }
                else                     { $cudaTag = "cu118" }
                Write-Host "[setup] NVIDIA GPU detected, CUDA $cudaMaj.x -> using PyTorch wheel: $cudaTag" -ForegroundColor Cyan
                break
            }
        }
    } catch {}

    if ($null -eq $cudaTag) {
        Write-Host "[setup] No NVIDIA GPU detected or nvidia-smi not found — installing CPU-only PyTorch." -ForegroundColor Yellow
        Write-Host "[setup] NOTE: lipsync will be disabled without a CUDA GPU." -ForegroundColor Yellow
        $wheelArgs = @("torch", "torchvision", "torchaudio")
    } else {
        $wheelUrl  = "https://download.pytorch.org/whl/$cudaTag"
        $wheelArgs = @("torch", "torchvision", "torchaudio", "--index-url", $wheelUrl)
        Write-Host "[setup] Installing PyTorch with CUDA from $wheelUrl ..." -ForegroundColor Cyan
        Write-Host "[setup] (This downloads ~2.5 GB on first install — please wait)" -ForegroundColor DarkGray
    }

    pip install @wheelArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: PyTorch install failed. Check output above." -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] PyTorch installed." -ForegroundColor Green
}

# --- Install the project package in editable mode ---
Write-Host "[setup] Installing scrumsurvivor package in editable mode ..." -ForegroundColor Cyan
pip install -e $PSScriptRoot --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] ERROR: Failed to install scrumsurvivor package." -ForegroundColor Red
    exit 1
}
Write-Host "[setup] scrumsurvivor package installed." -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ScrumSurvivor environment is ready!" -ForegroundColor Green
Write-Host " Python: $PythonVersion"
Write-Host " Venv:   $VenvDir"
Write-Host " Run:    python -m scrumsurvivor run"
Write-Host "========================================" -ForegroundColor Cyan
