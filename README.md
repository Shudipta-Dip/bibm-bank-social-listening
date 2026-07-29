# BIBM Bank Social Listening

Comparative social-listening study for **BRAC Bank** vs **Standard Chartered Bangladesh** (BIBM brand management).

Interactive dashboard + free collection/NLP pipeline (Facebook BCUB group searches, Google Play, App Store, Reddit).

## Live dashboard

- **Local:** `streamlit run dashboard/app.py` → http://localhost:8501  
- **Hosted:** [Streamlit Community Cloud](https://share.streamlit.io) (deploys from this GitHub repo — GitHub Pages cannot run Streamlit)

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public repo required on the free Cloud plan).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (`Shudipta-Dip`).
3. **Create app** → select this repository.
4. Set:
   - **Main file path:** `dashboard/app.py`
   - **Python version:** 3.11+ (or 3.12)
5. Deploy. The app loads `data/processed/unified_mentions_clean.csv` from the repo.

After each dashboard/data change, push to `main`/`master`; Cloud will redeploy automatically if that is enabled.

## Quick start (dashboard only)

```powershell
cd $env:USERPROFILE\Projects\bibm-bank-social-listening
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

## Full pipeline setup

```powershell
pip install -r requirements-pipeline.txt
playwright install chromium
copy .env.example .env
```

Edit `.env` with Meta Page tokens (preferred for Facebook) and optional LinkedIn credentials. Confirm brands in `config/brands.yaml`.

### Run collection / NLP

```powershell
python -m listening run --skip-nlp-model
python -m listening collect --source google_play --source app_store
python -m listening process --skip-nlp-model
python -m listening clean
```

Outputs:

- `data/raw/<source>/<brand>/*.jsonl` — raw payloads (gitignored)
- `data/processed/unified_mentions_clean.csv` — dashboard evidence base (tracked)
- `reports/summary.md` — coverage summary

## Project layout

| Path | Role |
|------|------|
| `dashboard/` | Streamlit app + analytics |
| `src/listening/` | Collectors, normalize, NLP, HITL |
| `config/` | Brands and themes |
| `data/processed/unified_mentions_clean.csv` | Cleaned mentions for the dashboard |

## Ethics

Client-authorized collection for owned brand surfaces and public app reviews. Authors are stored as hashes in exports where applicable. Browser automation may conflict with platform ToS — prefer official APIs where available. Never commit `.env`.
