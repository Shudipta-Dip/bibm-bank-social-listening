# Banking Social Media Analysis

Comparative social-listening study for **BRAC Bank** vs **Standard Chartered Bangladesh**.

Interactive dashboard + free collection/NLP pipeline (Facebook BCUB group searches, Google Play, App Store, Reddit).

## Live dashboard

- **GitHub Pages:** https://shudipta-dip.github.io/bibm-bank-social-listening/
- **Local (full filters):** `streamlit run dashboard/app.py` → http://localhost:8501

GitHub Pages hosts a **static** Plotly snapshot from `docs/` (GitHub cannot run Streamlit servers). The Streamlit app is unchanged for local interactive filtering.

Rebuild the Pages site after data or chart logic changes:

```powershell
python scripts/build_github_pages.py
```

## Quick start (dashboard only)

```powershell
cd <project-root>
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
