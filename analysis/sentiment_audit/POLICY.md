# Sentiment reclassification policy (applied 2026-07-30)

## Parallel metrics (user decision)
- **Text sentiment** → `sentiment_text` (= `sentiment_final` for dashboard charts)
- **Star / rating sentiment** → `rating_sentiment` from `star_rating` (1–2 neg, 3 neu, 4–5 pos)
- **Never merge** stars into text labels

## Text path
1. Base model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
2. Policy overrides (`src/listening/nlp/policy.py`):
   - Customer gift (“a gift from …”) → **positive**
   - Congrats / executive PR → **neutral**
   - Query / advice-seeking that model marked positive → **neutral**
   - Short “? + ভালো/good/best” asks → **neutral**

## Reproduce
```powershell
.\.venv\Scripts\python.exe analysis/sentiment_audit/reclassify_corpus.py
```

Backup of pre-reclassify clean CSV:
`data/processed/unified_mentions_clean.csv.bak_pre_reclassify`
