# ScrumSurvivor — Start pipeline
# Usage:  .\run.ps1
#         .\run.ps1 --preview        (force local preview window on)
#         .\run.ps1 --no-preview     (force local preview window off)

param(
    [switch]$Preview,
    [switch]$NoPreview
)

Set-Location $PSScriptRoot

# ── Kill leftover ScrumSurvivor Python processes only ─────────────────────────
function Stop-ProjectPythonProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    $modulePattern = "(^|\s)-m\s+$([regex]::Escape($ModuleName))(\s|$)"

    try {
        $existingProcesses = Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -match '^python(?:w)?(?:\.exe)?$' -and
            $_.CommandLine -and
            $_.CommandLine -match $modulePattern
        }
    } catch {
        $existingProcesses = @()
    }

    if ($existingProcesses) {
        Write-Host "  Stopping $($existingProcesses.Count) leftover Python process(es) for $ModuleName..." -ForegroundColor Yellow
        $existingProcesses | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 600
    }
}

Stop-ProjectPythonProcesses -ModuleName "scrumsurvivor"

# ── Activate virtual environment ──────────────────────────────────────────────
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run .\setup_venv.ps1 first."
    exit 1
}
. $venvActivate

# ── Launch pipeline ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ScrumSurvivor starting..." -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

if ($Preview) {
    python -m scrumsurvivor run --preview --prompt-theme
} elseif ($NoPreview) {
    python -m scrumsurvivor run --no-preview --prompt-theme
} else {
    python -m scrumsurvivor run --prompt-theme
}
