param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$VerifierArgs
)

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run .\setup.ps1 first."
    exit 1
}

. $venvActivate

Write-Host "" 
Write-Host "  Illusion verifier starting..." -ForegroundColor Cyan
Write-Host "  Tip: record a clap or say 'one two three' to review sync." -ForegroundColor DarkGray
Write-Host "  Stop: by default press Ctrl+Shift+F10, or override with --stop-hotkey." -ForegroundColor DarkGray
Write-Host "" 

python -m illusion_verifier @VerifierArgs