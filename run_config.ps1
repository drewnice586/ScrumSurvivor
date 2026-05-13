# ScrumSurvivor — launch the interactive setup wizard
# Usage:  .\run_setup.ps1
#         .\run_setup.ps1 -ConfigPath custom-config.yaml

param(
    [string]$ConfigPath = "config.yaml"
)

Set-Location $PSScriptRoot

# ── Activate virtual environment ──────────────────────────────────────────────
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run .\setup_venv.ps1 first."
    exit 1
}
. $venvActivate

# ── Launch setup wizard ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ScrumSurvivor setup starting..." -ForegroundColor Cyan
Write-Host "  Follow the prompts to select microphone, virtual audio, and speech threshold." -ForegroundColor DarkGray
Write-Host ""

python -m scrumsurvivor setup --config $ConfigPath