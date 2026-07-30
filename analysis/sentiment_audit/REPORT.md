# Adversarial Sentiment Classification Audit — BIBM

**Date:** 2026-07-30  
**Corpus:** `data/processed/unified_mentions_clean.csv` (n = 3,642)  
**Scope:** Analysis only — production `sentiment_final` and the Streamlit dashboard were **not** modified.

---

## 1. Executive finding (high stakes)

Production labels in the clean CSV that feed the dashboard are **not** from the configured transformer
`cardiffnlp/twitter-xlm-roberta-base-sentiment`.

Evidence:

| Signal | Observation |
|--------|-------------|
| `sentiment_score` mass | Only `{0.4, 0.6, 0.7, 0.8}` — exact outputs of `_lexicon_sentiment` |
| No-star rows vs lexicon | **100% agreement** with the tiny keyword lexicon |
| Social vs reviews | Non-rated Facebook/Reddit text → lexicon; store rows with stars → star map overwrites text |

So brand conclusions that depend on `sentiment_final` (Meaningfulness, PODs, phygital net, App/UX positivity) currently rest on:

1. A **~10-token English/Bangla substring lexicon**, and  
2. **Star ratings** (1–2 neg / 3 neu / 4–5 pos) when present.

That is **not** government-grade NLP. It systematically over-labels questions and advice that contain words like `good`, `best`, `ভালো` as **positive**.

---

## 2. Methods

### 2.1 Vendor baselines (cloned under `third_party/sentiment/`)

| Clone | Upstream | Use |
|-------|----------|-----|
| `vaderSentiment` | cjhutto/vaderSentiment | English social lexicon (compound ≥ 0.05 / ≤ −0.05) |
| `xlm-t` | cardiffnlp/xlm-t | Reference for multilingual Twitter XLM-R family |
| `banglabert` | csebuetnlp/banglabert | Bangla NLU / BLUB authority |
| `bangla-bert` | sagorbrur/bangla-bert | Published Bangla sentiment numbers |

### 2.2 Judges applied to the corpus

| Judge | Model / rule | Coverage |
|-------|----------------|----------|
| Lexicon reproduce | Exact copy of `src/listening/nlp/sentiment.py::_lexicon_sentiment` | Full |
| VADER | Local clone | Full (English-leaning) |
| XLM-R | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (project config) | Full |
| BanglaBERT-Senti | `ahs95/banglabert-sentiment-analysis` (5-class → 3-class) | `bn` + `mixed` (n = 1,238) |

### 2.3 Intent / failure taxonomy

Rule flags in `error_taxonomy.py`: query/advice-seeking, recommendations, comparisons, switch intent, negation near praise, lexicon hits inside questions, short lexicon positives.

### 2.4 Limitations

- Judges disagree with each other; none is ground truth. Human gold is still required (`data/hitl/quality_samples.jsonl` has **0** human labels).
- VADER is weak on Bangla script; BN rows rely on BanglaBERT-Senti + XLM-R.
- BanglaBERT-Senti is 5-class (very neg → very pos); mapping compresses nuance.
- Star-vs-text “conflict” is expected when users give 5★ with a complaint about fees — both signals can be “right” for different questions.
- First XLM-R run failed (missing `tiktoken`/`sentencepiece`); re-run after install — metrics below are from the **successful** run.

---

## 3. Quantitative results

### 3.1 Agreement with production `sentiment_final`

| Judge | n | Agreement % | Cohen’s κ |
|-------|---|-------------|-----------|
| Lexicon | 3642 | 81.60 | 0.6618 |
| XLM-R (config model) | 3642 | 71.83 | 0.5416 |
| VADER | 3642 | 65.87 | 0.4316 |
| BanglaBERT-Senti (bn/mixed) | 1238 | 49.19 | 0.2358 |

Lexicon “wins” agreement only because production **is** the lexicon on non-star rows (circular).

XLM-R κ = 0.54 on full corpus — moderate; **Facebook κ = 0.24** (weak) where most social discourse lives.

BanglaBERT-Senti κ = 0.24 on bn/mixed — production Bangla labels do **not** track a Bangla-specialized judge.

### 3.2 Production vs XLM-R confusion

| production \\ xlmr | negative | neutral | positive |
|--------------------|----------|---------|----------|
| negative | 515 | 137 | 27 |
| neutral | 423 | 1457 | 115 |
| positive | 85 | **239** | 644 |

**239 production-positives** are XLM-R **neutral**; **85** are XLM-R **negative**. These are primary FP candidates when stars are absent or when stars disagree with text.

### 3.3 Production vs BanglaBERT-Senti (bn/mixed)

| production \\ bn | negative | neutral | positive |
|------------------|----------|---------|----------|
| negative | 130 | 6 | 27 |
| neutral | 304 | 366 | 213 |
| positive | 32 | 47 | 113 |

Large mass of production-**neutral** called **negative** by BanglaBERT (304) — likely under-calling complaints in Bangla Facebook talk when no lexicon neg-token fires.

### 3.4 Candidate volumes

| Set | n |
|-----|---|
| False-positive candidates | 674 |
| False-negative candidates | 99 |
| Star vs text conflicts | 763 |
| Query-like production positives | 86 |

Top FP failure modes:

| Mode | Count |
|------|------:|
| short_lexicon_positive | 421 |
| substring_false_positive_in_query | 118 |
| judge_disagreement_fp | 84 |
| negation_or_sarcasm | 9 |
| advice_marked_positive | 9 |
| comparative_not_affective | 8 |

---

## 4. Failure modes (with examples)

### A. Substring false positive in queries (your observation)

Lexicon sees `ভালো` / `good` inside an informational question → `positive`.

- “কোন ব্যাংকে সব মিলিয়ে **ভালো** হবে ?” → production **positive**; XLM-R **neutral**; BN **neutral**
- “student account এর জন্য কোনটা **ভালো**?? (১)brac (2)prime” → same pattern
- “কোনটা **ভালো** হবে। … কোন ব্যাংক **ভালো** হবে জানাবেন।” → same pattern

**Impact:** Inflates positive share and Meaningfulness for banks mentioned in advice threads.

### B. Advice / recommendation marked positive

- “এরচেয়ে দেশি **ভালো** কিছু ব্যাংকের ডিজিটাল একাউন্ট…” (suggesting alternatives) → production positive
- Mixed praise + complaint: “**ভালো** কিন্তু ইউজারনেম প্রত্যেকবার…” → production positive; XLM-R neutral

### C. Negation near praise

- “User interface is **not** so **good**.” → production **positive** (hit on `good`); XLM-R **negative**; VADER **negative**
- “apps is **good**, but not working my phone.” → production positive; XLM-R negative

### D. Star override vs text

`clean_for_analysis` forces star map into `sentiment_final`:

- 4★ “User interface is not so good…” → production **positive**
- 5★ “Very good quality app, but the charges are high.” → production positive; XLM-R negative

Stars are a valid **rating** signal but unsafe as the sole brand-sentiment label for mixed reviews.

### E. Neutral mass / Bangla under-calling

1,995 neutrals at score 0.4 (lexicon default). BanglaBERT often labels the same posts negative — production may miss complaint volume on Facebook BN.

---

## 5. Human verification pack

File: [`out/edge_cases_for_human.csv`](out/edge_cases_for_human.csv) (**55 rows**)

Columns: `record_id`, `source`, `language`, `text`, `production_label`, `lexicon_label`, `xlmr_label`, `vader_or_bn_label`, `failure_mode`, `why_suspect`, **`your_label`** (blank).

Please fill `your_label` with `positive` | `neutral` | `negative`. Priority order:

1. `substring_false_positive_in_query`
2. `advice_marked_positive`
3. `negation_or_sarcasm`
4. `comparative_not_affective`
5. Star conflicts / FN samples

Supporting files: `false_positive_candidates.csv`, `false_negative_candidates.csv`, `star_vs_text_conflicts.csv`, `example_snippets.txt`, confusion matrices.

---

## 6. Recommended remediation (after your labels — do not apply yet)

1. **Stop silent lexicon fallback** in production runs; fail loudly or gate HITL if the HF model cannot load.
2. **Re-process** the corpus with XLM-R (and BanglaBERT-Senti for `bn`/`mixed`) into parallel columns: `sentiment_text_model`, keep `rating_sentiment` separate; do not drop text label when stars exist.
3. **Intent rule:** if query/advice flag and no clear affective clause → force `neutral` for brand dashboards.
4. **Negation guard** for lexicon path if lexicon is retained as backup.
5. **Activate gold QA:** label ≥200 stratified rows; enforce existing gates (macro-F1 ≥ 0.65, κ ≥ 0.5) before freezing metrics for BIBM.
6. Dashboard: expose `sentiment_source` and dual charts (text vs stars) so stakeholders see the method.

---

## 7. How to reproduce

```powershell
cd C:\Users\USER\Projects\bibm-bank-social-listening
.\.venv\Scripts\pip.exe install torch transformers accelerate sentencepiece tiktoken
# clones already under third_party/sentiment/ (see third_party/README.md)
.\.venv\Scripts\python.exe analysis/sentiment_audit/run_adversarial_audit.py
.\.venv\Scripts\python.exe analysis/sentiment_audit/curate_edge_cases.py
```

Caches: `analysis/sentiment_audit/cache/` (gitignored).  
Report + review CSVs: `analysis/sentiment_audit/out/`.
