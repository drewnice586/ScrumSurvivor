# ScrumSurvivor — launch the interactive asset creation wizard
# Usage:  .\run_create_assets.ps1

Set-Location $PSScriptRoot

# ── Activate virtual environment ──────────────────────────────────────────────
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run .\setup_venv.ps1 first."
    exit 1
}
. $venvActivate

# ── Launch asset creator ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ScrumSurvivor asset creator starting..." -ForegroundColor Cyan
Write-Host "  Follow the on-screen instructions to capture base photo + idle clips." -ForegroundColor DarkGray
Write-Host ""

python -m scrumsurvivor.create_assets
