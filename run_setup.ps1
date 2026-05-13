# ScrumSurvivor — Full Prerequisites & Installation Wizard
# Run this ONCE on a new machine before anything else.
# Usage:  .\run_setup.ps1
#
# What it does (in order, skipping steps that are already done):
#   1.  Set PowerShell execution policy for this session
#   2.  Create Python .venv and install pip packages
#   3.  Install PyTorch with CUDA support (auto-detects CUDA version)
#   4.  Install VB-Cable audio driver (download from https://vb-audio.com/Cable/)
#   5.  Check / guide OBS Virtual Camera installation
#   6.  Check / place ffmpeg binary (ffmpeg\ffmpeg.exe)
#   7.  Check / place Wav2Lip model weights (models\)
#   8.  Remind user to run .\run_config.ps1 next

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function Write-Step {
    param([int]$Number, [string]$Title)
    Write-Host ""
    Write-Host "  [$Number/7] $Title" -ForegroundColor Cyan
    Write-Host "  $('─' * 60)" -ForegroundColor DarkGray
}

function Write-Ok   { param([string]$Msg) Write-Host "  ✓ $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "  ! $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "  ✗ $Msg" -ForegroundColor Red }
function Write-Info { param([string]$Msg) Write-Host "    $Msg" -ForegroundColor DarkGray }

function Pause-ForUser {
    param([string]$Prompt = "Press ENTER to continue...")
    Write-Host ""
    Write-Host "  $Prompt" -ForegroundColor White -NoNewline
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""
}

function Ask-YesNo {
    param([string]$Question, [bool]$Default = $true)
    $hint = if ($Default) { "[Y/n]" } else { "[y/N]" }
    Write-Host ""
    Write-Host "  $Question $hint " -ForegroundColor White -NoNewline
    $answer = Read-Host
    if ($answer -eq "") { return $Default }
    return $answer -match '^[Yy]'
}

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║       ScrumSurvivor — Installation & Setup Wizard       ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  This wizard installs all prerequisites for ScrumSurvivor." -ForegroundColor White
Write-Host "  Steps that are already complete are skipped automatically." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Prerequisites (ensure these are met before continuing):" -ForegroundColor Yellow
Write-Host "   • Python 3.10+ must be installed system-wide." -ForegroundColor White
Write-Host "   • Administrator rights are required to install drivers and software." -ForegroundColor White
Write-Host ""
Pause-ForUser "Press ENTER to start..."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Execution policy
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 1 "PowerShell Execution Policy"

$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -in @("RemoteSigned", "Unrestricted", "Bypass")) {
    Write-Ok "Execution policy already set to '$policy' for CurrentUser."
} else {
    Write-Warn "Current policy: $policy — setting to RemoteSigned for CurrentUser..."
    try {
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
        Write-Ok "Execution policy set to RemoteSigned."
    } catch {
        Write-Err "Could not set execution policy: $_"
        Write-Info "You can set it manually:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
    }
}

# Also set for this process (needed for the rest of this session)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Python venv + pip packages
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 2 "Python Virtual Environment & Packages"

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Ok "Virtual environment already exists."
} else {
    Write-Info "Creating virtual environment..."
}

# setup_venv.ps1 handles creation + requirements install; it is idempotent
& (Join-Path $PSScriptRoot "setup_venv.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Err "setup_venv.ps1 failed. Fix the error above and re-run this wizard."
    exit 1
}

# Activate for the rest of this session
. $venvActivate

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PyTorch with CUDA
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 3 "PyTorch with CUDA GPU Support"

# setup_venv.ps1 (Step 2) already installed PyTorch with the correct CUDA wheel.
# Just verify and report here.
try {
    $torchVer      = python -c "import torch; print(torch.__version__)" 2>$null
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaAvailable -eq "True") {
        Write-Ok "PyTorch $torchVer installed with CUDA support."
    } else {
        Write-Warn "PyTorch $torchVer installed but CUDA is NOT available."
        Write-Info "If you have an NVIDIA GPU, re-run setup_venv.ps1 after installing NVIDIA drivers."
    }
} catch {
    Write-Err "PyTorch does not appear to be installed. Re-run this wizard."
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VB-Cable audio driver
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 4 "VB-Cable Virtual Audio Driver"

# Check if VB-Cable is already present as an audio device
$vbCablePresent    = $false
$vbCableNeedsReboot = $false
try {
    $audioDevices = python -c "import sounddevice as sd; devs = sd.query_devices(); print([d['name'] for d in devs])" 2>$null
    if ($audioDevices -match "CABLE") {
        $vbCablePresent = $true
    }
} catch {}

if ($vbCablePresent) {
    Write-Ok "VB-Cable is already installed (detected in audio devices)."
} else {
    Write-Warn "VB-Cable was NOT detected in your audio devices."
    Write-Info "VB-Cable is a free virtual audio driver by VB-Audio (donationware)."
    Write-Info "Administrator rights are required to install the driver."
    Write-Info ""

    if (Ask-YesNo "Download and install VB-Cable automatically now? (requires admin rights)") {
        $vbZipUrl  = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
        $vbTmpDir  = Join-Path $env:TEMP "ScrumSurvivor_VBCable"
        $vbZipPath = Join-Path $vbTmpDir "VBCABLE_Driver.zip"

        try {
            if (-not (Test-Path $vbTmpDir)) { New-Item -ItemType Directory -Path $vbTmpDir | Out-Null }
            Write-Info "Downloading VB-Cable installer (~1.3 MB)..."
            Invoke-WebRequest -Uri $vbZipUrl -OutFile $vbZipPath -UseBasicParsing
            Write-Info "Extracting..."
            Expand-Archive -Path $vbZipPath -DestinationPath $vbTmpDir -Force

            $setupExe = Get-ChildItem -Path $vbTmpDir -Filter "VBCABLE_Setup_x64.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $setupExe) {
                $setupExe = Get-ChildItem -Path $vbTmpDir -Filter "VBCABLE_Setup.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            }

            if ($setupExe) {
                Write-Info "Launching VB-Cable installer — please approve the UAC prompt when it appears..."
                Start-Process -FilePath $setupExe.FullName -Verb RunAs
                Pause-ForUser "Complete the VB-Cable installation, then press ENTER to continue..."
                Write-Ok "VB-Cable installer finished."
                $vbCableNeedsReboot = $true
            } else {
                Write-Warn "Could not locate VBCABLE_Setup_x64.exe in the downloaded package."
                Write-Info "Please open $vbTmpDir and run the setup executable manually."
            }
        } catch {
            Write-Warn "Automatic download failed: $_"
            Write-Info "Please download VB-Cable manually from https://vb-audio.com/Cable/"
            if (Ask-YesNo "Open the VB-Cable download page in your browser?") {
                Start-Process "https://vb-audio.com/Cable/"
            }
        }
    } else {
        Write-Info "Skipping VB-Cable installation."
        Write-Info "ScrumSurvivor requires VB-Cable. Install it later from https://vb-audio.com/Cable/"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — OBS Virtual Camera
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 5 "OBS Virtual Camera"

# Check whether the OBS Virtual Camera driver is installed.
# We look for the OBS Studio installation on disk and the virtual camera DLL
# rather than trying to open a live camera session (which requires OBS to be
# actively running with the virtual camera started).
$obsOk = $false
$obsPaths = @(
    "$env:ProgramFiles\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll",
    "$env:ProgramFiles\obs-studio\obs-plugins\64bit\obs-virtualcam-module64.dll",
    "$env:ProgramFiles (x86)\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll"
)
foreach ($p in $obsPaths) {
    if (Test-Path $p) { $obsOk = $true; break }
}
# Fallback: check the registry for the DirectShow filter registered by OBS
if (-not $obsOk) {
    $obsRegKey = "HKLM:\SOFTWARE\Classes\CLSID\{A3FCE0F5-3493-419F-958A-ABA1220EC37A}"
    if (Test-Path $obsRegKey) { $obsOk = $true }
}
# Second fallback: check if OBS Studio itself is present
if (-not $obsOk) {
    $obsExe = "$env:ProgramFiles\obs-studio\bin\64bit\obs64.exe"
    if (Test-Path $obsExe) { $obsOk = $true }
}

if ($obsOk) {
    Write-Ok "OBS Virtual Camera is available."
} else {
    Write-Warn "OBS Virtual Camera was NOT detected."
    Write-Info "ScrumSurvivor requires the OBS Studio Virtual Camera plugin."
    Write-Info ""
    Write-Info "To install:"
    Write-Info "  1. Download and install OBS Studio from https://obsproject.com/"
    Write-Info "  2. You do NOT need to configure OBS — just installing it is enough."
    Write-Info ""

    if (Ask-YesNo "Open the OBS Studio download page in your browser?") {
        Start-Process "https://obsproject.com/download"
    }

    Pause-ForUser "After installing OBS Studio, press ENTER to continue..."

    # Re-check using file/registry detection (same logic as above)
    foreach ($p in $obsPaths) {
        if (Test-Path $p) { $obsOk = $true; break }
    }
    if (-not $obsOk -and (Test-Path "HKLM:\SOFTWARE\Classes\CLSID\{A3FCE0F5-3493-419F-958A-ABA1220EC37A}")) { $obsOk = $true }
    if (-not $obsOk -and (Test-Path "$env:ProgramFiles\obs-studio\bin\64bit\obs64.exe")) { $obsOk = $true }

    if ($obsOk) {
        Write-Ok "OBS Virtual Camera is now available."
    } else {
        Write-Warn "OBS Virtual Camera still not detected."
        Write-Info "Make sure OBS Studio finished installing, then re-run this wizard."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 6 "ffmpeg (for Illusion Verifier)"

$ffmpegLocal  = Join-Path $PSScriptRoot "ffmpeg\ffmpeg.exe"
$ffmpegInPath = $null
try { $ffmpegInPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source } catch {}

if (Test-Path $ffmpegLocal) {
    Write-Ok "ffmpeg found at ffmpeg\ffmpeg.exe  (local copy — preferred)."
} elseif ($ffmpegInPath) {
    Write-Ok "ffmpeg found in PATH: $ffmpegInPath"
    Write-Info "The Illusion Verifier will use the PATH version."
    Write-Info "To use a local copy instead, place ffmpeg.exe in the ffmpeg\ folder."
} else {
    Write-Warn "ffmpeg was NOT found."
    Write-Info "ffmpeg is required by the Illusion Verifier tool."
    Write-Info ""
    Write-Info "To install:"
    Write-Info "  1. Download a Windows build from https://www.gyan.dev/ffmpeg/builds/"
    Write-Info "     (Recommended: 'ffmpeg-release-essentials.zip')"
    Write-Info "  2. Extract the zip and copy:"
    Write-Info "       bin\ffmpeg.exe  →  $(Join-Path $PSScriptRoot 'ffmpeg\ffmpeg.exe')"
    Write-Info "  3. You do NOT need ffprobe or ffplay (but they can go in ffmpeg\ too)."
    Write-Info ""

    $ffmpegDir = Join-Path $PSScriptRoot "ffmpeg"
    if (-not (Test-Path $ffmpegDir)) {
        New-Item -ItemType Directory -Path $ffmpegDir | Out-Null
    }

    if (Ask-YesNo "Open the ffmpeg download page in your browser?") {
        Start-Process "https://www.gyan.dev/ffmpeg/builds/"
    }

    Pause-ForUser "After placing ffmpeg.exe in the ffmpeg\ folder, press ENTER to continue..."

    if (Test-Path $ffmpegLocal) {
        Write-Ok "ffmpeg found."
    } else {
        Write-Warn "ffmpeg still not found. The Illusion Verifier will not work until it is placed."
        Write-Info "Re-run this wizard after placing ffmpeg.exe."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Wav2Lip model weights
# ─────────────────────────────────────────────────────────────────────────────
Write-Step 7 "Wav2Lip Model Weights"

$modelsDir       = Join-Path $PSScriptRoot "models"
$noganModel      = Join-Path $modelsDir "Wav2Lip-SD-NOGAN.pt"
$ganModel        = Join-Path $modelsDir "Wav2Lip-SD-GAN.pt"

$noganOk = Test-Path $noganModel
$ganOk   = Test-Path $ganModel

if ($noganOk -and $ganOk) {
    Write-Ok "Both Wav2Lip model files found:"
    Write-Info "  models\Wav2Lip-SD-NOGAN.pt"
    Write-Info "  models\Wav2Lip-SD-GAN.pt"
} else {
    if ($noganOk) {
        Write-Ok "Wav2Lip-SD-NOGAN.pt  ✓ (standard model — used by default)"
    } else {
        Write-Warn "Wav2Lip-SD-NOGAN.pt  MISSING  ← required"
    }
    if ($ganOk) {
        Write-Ok "Wav2Lip-SD-GAN.pt    ✓ (GAN variant — optional)"
    } else {
        Write-Info "Wav2Lip-SD-GAN.pt    not found  (optional)"
    }

    if (-not $noganOk) {
        Write-Host ""
        Write-Info "The Wav2Lip model weights must be downloaded manually from Google Drive."
        Write-Info "These are large files (~360 MB each) and are not included in the repository."
        Write-Info ""
        Write-Info "Download links (from the official Wav2Lip README):"
        Write-Info "  Wav2Lip (NOGAN, required):"
        Write-Info "    https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link"
        Write-Info "  Wav2Lip + GAN (optional — higher quality, slower):"
        Write-Info "    https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link"
        Write-Info ""
        Write-Info "After downloading, rename the files to EXACTLY:"
        Write-Info "  Wav2Lip-SD-NOGAN.pt   (the NOGAN / standard model)"
        Write-Info "  Wav2Lip-SD-GAN.pt     (the GAN model, optional)"
        Write-Info ""
        Write-Info "Place the renamed file(s) in:"
        Write-Info "  $(Join-Path $PSScriptRoot 'models\')"

        if (-not (Test-Path $modelsDir)) {
            New-Item -ItemType Directory -Path $modelsDir | Out-Null
        }

        # Open the models folder so the user can drop files in easily
        Start-Process "explorer.exe" $modelsDir

        if (Ask-YesNo "Open the Wav2Lip (NOGAN) Google Drive download page in your browser?") {
            Start-Process "https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link"
        }

        Pause-ForUser "After placing the model file(s) in models\, press ENTER to continue..."

        $noganOk = Test-Path $noganModel
        if ($noganOk) {
            Write-Ok "Wav2Lip-SD-NOGAN.pt found."
        } else {
            Write-Warn "Wav2Lip-SD-NOGAN.pt still not found. ScrumSurvivor will not start without it."
            Write-Info "Re-run this wizard after placing the file, or copy it in manually."
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║                  Setup Complete!                        ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Quick status recap
$checks = @(
    @{ Label = "Python venv";         OK = (Test-Path $venvActivate);                                  Reboot = $false },
    @{ Label = "VB-Cable driver";     OK = $vbCablePresent;                                            Reboot = $vbCableNeedsReboot },
    @{ Label = "OBS Virtual Camera";  OK = $obsOk;                                                     Reboot = $false },
    @{ Label = "ffmpeg";              OK = ((Test-Path $ffmpegLocal) -or ($null -ne $ffmpegInPath));   Reboot = $false },
    @{ Label = "Wav2Lip model";       OK = (Test-Path $noganModel);                                    Reboot = $false }
)
foreach ($c in $checks) {
    if ($c.OK) {
        Write-Host "  ✓ $($c.Label)" -ForegroundColor Green
    } elseif ($c.Reboot) {
        Write-Host "  ~ $($c.Label)  ← installed, reboot required" -ForegroundColor Yellow
    } else {
        Write-Host "  ✗ $($c.Label)  ← not complete" -ForegroundColor Red
    }
}

Write-Host ""
if ($vbCableNeedsReboot) {
    Write-Host "  Next steps (after reboot):" -ForegroundColor White
} else {
    Write-Host "  Next steps:" -ForegroundColor White
}
Write-Host "   1. Capture your avatar photo + idle clips:   .\run_create_assets.ps1" -ForegroundColor DarkGray
Write-Host "   2. Configure microphone & audio settings:    .\run_config.ps1" -ForegroundColor DarkGray
Write-Host "   3. Start ScrumSurvivor before a meeting:     .\run.ps1" -ForegroundColor DarkGray
Write-Host ""

if ($vbCableNeedsReboot) {
    Write-Host "  ╔══════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "  ║              !!  REBOOT REQUIRED  !!                    ║" -ForegroundColor Yellow
    Write-Host "  ╚══════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  VB-Cable was installed and needs a system reboot to activate." -ForegroundColor Yellow
    Write-Host "  Please save your work and restart your computer now." -ForegroundColor Yellow
    Write-Host "  After rebooting, continue with the next steps listed above." -ForegroundColor White
    Write-Host ""
}
