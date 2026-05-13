# Illusion Verifier — Start recorder from workspace root
# Usage:  .\run_verifier.ps1
#         .\run_verifier.ps1 --help

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$VerifierArgs
)

Set-Location $PSScriptRoot

# ── Kill leftover Illusion Verifier Python processes only ─────────────────────
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

Stop-ProjectPythonProcesses -ModuleName "illusion_verifier"

# ── Activate virtual environment ──────────────────────────────────────────────
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run .\setup_venv.ps1 first."
    exit 1
}
. $venvActivate

Write-Host ""
Write-Host "  Illusion verifier starting..." -ForegroundColor Cyan
Write-Host "  Tip: record a clap or say 'one two three' to review sync." -ForegroundColor DarkGray
Write-Host "  Stop: by default press Ctrl+Shift+F10, or override with --stop-hotkey." -ForegroundColor DarkGray
Write-Host ""

python -m illusion_verifier @VerifierArgs