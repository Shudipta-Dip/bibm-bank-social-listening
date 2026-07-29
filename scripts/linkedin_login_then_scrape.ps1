# Interactive LinkedIn login + scrape
# 1) Chromium opens on LinkedIn login
# 2) YOU log in (2FA ok)
# 3) After login is detected, scrapes both company pages

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:BROWSER_HEADED = "1"
$env:SOCIAL_BROWSER_ONLY = "1"
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) ".playwright-browsers"

Write-Host ""
Write-Host "A Chromium window will open to LinkedIn login." -ForegroundColor Yellow
Write-Host "Log in there (including 2FA). Do not close the window." -ForegroundColor Yellow
Write-Host ""

python -c @"
from pathlib import Path
import time, os
from playwright.sync_api import sync_playwright

root = Path('.').resolve()
profile = root / 'browser_profiles' / 'linkedin'
profile.mkdir(parents=True, exist_ok=True)
storage = root / 'browser_profiles' / 'linkedin_storage.json'

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        user_data_dir=str(profile),
        headless=False,
        viewport={'width': 1365, 'height': 900},
        locale='en-US',
        args=['--disable-blink-features=AutomationControlled'],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto('https://www.linkedin.com/login', wait_until='domcontentloaded', timeout=120000)
    print('[HITL] Log into LinkedIn now. Waiting up to 10 minutes...')
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(4)
        u = page.url.lower()
        if 'feed' in u or '/in/' in u or '/company/' in u or ('login' not in u and 'checkpoint' not in u and 'uas/' not in u):
            # still on login?
            body = ''
            try:
                body = page.inner_text('body')[:800].lower()
            except Exception:
                pass
            if 'sign in' in body and 'email' in body:
                continue
            print('[HITL] Login looks complete:', page.url)
            break
    else:
        print('[HITL] Timed out waiting for login')
        ctx.close()
        raise SystemExit(2)
    try:
        ctx.storage_state(path=str(storage))
    except Exception:
        pass
    ctx.close()
print('Session saved. Starting collectors...')
"@

if ($LASTEXITCODE -ne 0) {
  Write-Host "Login helper failed/timed out. Re-run this script." -ForegroundColor Red
  exit $LASTEXITCODE
}

python -m listening collect --source linkedin
python -m listening process --skip-nlp-model
Write-Host "Done. Check reports\summary.md" -ForegroundColor Green
