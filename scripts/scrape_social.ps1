# Banking Social Media Analysis - Facebook + LinkedIn browser scrape (no API tokens)
# Run from project root in PowerShell after: .\.venv\Scripts\Activate.ps1
#
# What you may need to do:
# 1. A Chromium window opens - if Facebook/LinkedIn ask you to log in or solve a captcha, do it there.
# 2. Leave the window alone while it scrolls (can take 5-15 min per brand).
# 3. If the script stops with a HITL gate, follow the printed message, then re-run this script.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:BROWSER_HEADED = "1"
$env:SOCIAL_BROWSER_ONLY = "1"
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) ".playwright-browsers"

Write-Host ""
Write-Host "=== 1/3 FACEBOOK (browser) ===" -ForegroundColor Cyan
python -m listening collect --source facebook --force-facebook-browser
if ($LASTEXITCODE -ne 0) {
  Write-Host "Facebook collect exited with code $LASTEXITCODE. Check: python -m listening hitl status" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 2/3 LINKEDIN (browser) ===" -ForegroundColor Cyan
python -m listening collect --source linkedin
if ($LASTEXITCODE -ne 0) {
  Write-Host "LinkedIn collect exited with code $LASTEXITCODE. Check: python -m listening hitl status" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 3/3 REBUILD DATASET ===" -ForegroundColor Cyan
python -m listening process --skip-nlp-model

Write-Host ""
Write-Host "Done. Open reports\summary.md and data\processed\unified_mentions_*.csv" -ForegroundColor Green
Write-Host "HITL status: python -m listening hitl status"
