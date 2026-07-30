"""Adversarial sentiment audit against lexicon, XLM-R, VADER, BanglaBERT-Senti.

Usage (repo root):
  .\\.venv\\Scripts\\python.exe analysis/sentiment_audit/run_adversarial_audit.py

Does not modify production clean CSV or dashboard.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "sentiment" / "vaderSentiment"))

from analysis.sentiment_audit.error_taxonomy import (  # noqa: E402
    STAR_MAP,
    failure_mode,
    intent_flags,
    lexicon_hits,
    reproduce_lexicon,
)

OUT = Path(__file__).resolve().parent / "out"
CACHE = Path(__file__).resolve().parent / "cache"
CLEAN = ROOT / "data" / "processed" / "unified_mentions_clean.csv"

XLMR_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BN_ID = "ahs95/banglabert-sentiment-analysis"

BN_5_TO_3 = {
    "very negative": "negative",
    "very_negative": "negative",
    "negative": "negative",
    "label_0": "negative",
    "0": "negative",
    "neutral": "neutral",
    "label_1": "neutral",
    "1": "neutral",
    "positive": "positive",
    "label_2": "positive",
    "2": "positive",
    "very positive": "positive",
    "very_positive": "positive",
    "label_3": "positive",
    "3": "positive",
    "label_4": "positive",
    "4": "positive",
}


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, b in zip(y_true, y_pred):
        matrix[idx[a]][idx[b]] += 1
    po = sum(matrix[i][i] for i in range(len(labels))) / n
    row = [sum(matrix[i][j] for j in range(len(labels))) for i in range(len(labels))]
    col = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement(a: list[str], b: list[str]) -> float:
    if not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] VADER unavailable: {exc}")
        return None


def vader_label(analyzer, text: str) -> tuple[str, float]:
    if analyzer is None or not text or not str(text).strip():
        return "neutral", 0.0
    s = analyzer.polarity_scores(str(text)[:2000])
    c = float(s.get("compound", 0.0))
    if c >= 0.05:
        return "positive", abs(c)
    if c <= -0.05:
        return "negative", abs(c)
    return "neutral", abs(c)


def load_hf_pipe(model_id: str):
    cache_meta = CACHE / f"pipe_{model_id.replace('/', '_')}.ok"
    try:
        from transformers import pipeline

        print(f"[info] Loading HF pipeline: {model_id}")
        pipe = pipeline(
            "sentiment-analysis",
            model=model_id,
            truncation=True,
            max_length=512,
        )
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_meta.write_text("ok", encoding="utf-8")
        return pipe
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not load {model_id}: {exc}")
        return None


def map_bn_label(raw: str) -> str:
    key = (raw or "").strip().lower().replace("-", "_")
    if key in BN_5_TO_3:
        return BN_5_TO_3[key]
    # id2label sometimes "LABEL_0" style
    k2 = key.replace(" ", "_")
    return BN_5_TO_3.get(k2, "neutral")


def hf_predict(pipe, text: str, bangla: bool = False) -> tuple[str, float]:
    if pipe is None or not text or not str(text).strip():
        return "neutral", 0.0
    try:
        out = pipe(str(text)[:2000])[0]
        raw = str(out.get("label", "neutral"))
        score = float(out.get("score") or 0.0)
        if bangla:
            return map_bn_label(raw), score
        # cardiffnlp / generic
        lab = raw.strip().lower()
        mapping = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        return mapping.get(lab, "neutral"), score
    except Exception:  # noqa: BLE001
        return "neutral", 0.0


def batch_score(texts: list[str], fn, cache_path: Path, batch_note: str) -> list[tuple[str, float]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(data) == len(texts):
            print(f"[info] cache hit {cache_path.name} ({len(data)})")
            return [(d["label"], d["score"]) for d in data]
    print(f"[info] scoring {batch_note}: n={len(texts)}")
    out = []
    t0 = time.time()
    for i, text in enumerate(texts):
        out.append(fn(text))
        if (i + 1) % 200 == 0:
            print(f"  … {i+1}/{len(texts)} ({time.time()-t0:.1f}s)")
    payload = [{"label": a, "score": b} for a, b in out]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[info] wrote cache {cache_path.name} in {time.time()-t0:.1f}s")
    return out


def star_sentiment(row) -> str | None:
    if pd.isna(row.get("star_rating")):
        return None
    try:
        return STAR_MAP.get(int(row["star_rating"]))
    except (TypeError, ValueError):
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN, encoding="utf-8", low_memory=False)
    df["text"] = df["text"].fillna("")
    df["language"] = df["language"].fillna("und").astype(str)
    df["production"] = df["sentiment_final"].astype(str).str.lower()
    texts = df["text"].tolist()

    # --- Lexicon reproduce ---
    lex = [reproduce_lexicon(t) for t in texts]
    df["lexicon_label"] = [a for a, _ in lex]
    df["lexicon_score"] = [b for _, b in lex]
    df["lex_pos_hits"] = [",".join(lexicon_hits(t)[0]) for t in texts]
    df["lex_neg_hits"] = [",".join(lexicon_hits(t)[1]) for t in texts]
    df["intent_flags"] = [",".join(intent_flags(t)) for t in texts]

    # Production text path for non-star rows should match lexicon if pipeline used --skip-nlp-model
    no_star = df["star_rating"].isna()
    lex_agree = (df.loc[no_star, "production"] == df.loc[no_star, "lexicon_label"]).mean()
    print(f"[info] lexicon vs production (no-star rows): {100*lex_agree:.1f}%")

    # --- VADER ---
    vader = load_vader()
    vader_out = batch_score(
        texts,
        lambda t: vader_label(vader, t),
        CACHE / "vader_full.json",
        "VADER",
    )
    df["vader_label"] = [a for a, _ in vader_out]
    df["vader_score"] = [b for _, b in vader_out]

    # --- XLM-R (project config) ---
    xlmr = load_hf_pipe(XLMR_ID)
    xlmr_out = batch_score(
        texts,
        lambda t: hf_predict(xlmr, t, bangla=False),
        CACHE / "xlmr_full.json",
        "XLM-R twitter-xlm-roberta",
    )
    df["xlmr_label"] = [a for a, _ in xlmr_out]
    df["xlmr_score"] = [b for _, b in xlmr_out]

    # --- BanglaBERT sentiment (bn + mixed) ---
    bn_pipe = load_hf_pipe(BN_ID)
    bn_mask = df["language"].str.lower().isin(["bn", "mixed", "bangla", "bengali"])
    bn_indices = df.index[bn_mask].tolist()
    bn_texts = df.loc[bn_indices, "text"].tolist()
    bn_scored = batch_score(
        bn_texts,
        lambda t: hf_predict(bn_pipe, t, bangla=True),
        CACHE / "bn_senti_bn_mixed.json",
        "BanglaBERT-Senti (bn/mixed)",
    )
    df["bn_label"] = "skip"
    df["bn_score"] = 0.0
    for idx, (lab, sc) in zip(bn_indices, bn_scored):
        df.at[idx, "bn_label"] = lab
        df.at[idx, "bn_score"] = sc

    # External judge for row: prefer bn model for bn/mixed else vader; always keep xlmr
    def external_pack(row) -> dict[str, str]:
        judges = {"xlmr": row["xlmr_label"]}
        if row["bn_label"] != "skip":
            judges["bn"] = row["bn_label"]
        else:
            judges["vader"] = row["vader_label"]
        return judges

    df["star_from_rating"] = df.apply(star_sentiment, axis=1)
    # Text-only production proxy: lexicon for no-star, else star
    df["text_path_would_be"] = df["lexicon_label"]

    # --- Metrics ---
    metrics_rows = []
    for name, col in [
        ("lexicon", "lexicon_label"),
        ("vader", "vader_label"),
        ("xlmr", "xlmr_label"),
    ]:
        y_t = df["production"].tolist()
        y_p = df[col].tolist()
        metrics_rows.append(
            {
                "judge": name,
                "n": len(df),
                "agreement_pct": round(100 * agreement(y_t, y_p), 2),
                "cohen_kappa": round(cohen_kappa(y_t, y_p), 4),
            }
        )
    # bn subset
    sub = df[df["bn_label"] != "skip"]
    if len(sub):
        metrics_rows.append(
            {
                "judge": "banglabert_senti_bn_mixed",
                "n": len(sub),
                "agreement_pct": round(
                    100 * agreement(sub["production"].tolist(), sub["bn_label"].tolist()), 2
                ),
                "cohen_kappa": round(
                    cohen_kappa(sub["production"].tolist(), sub["bn_label"].tolist()), 4
                ),
            }
        )

    # per source
    for src, g in df.groupby("source"):
        metrics_rows.append(
            {
                "judge": f"xlmr|source={src}",
                "n": len(g),
                "agreement_pct": round(
                    100 * agreement(g["production"].tolist(), g["xlmr_label"].tolist()), 2
                ),
                "cohen_kappa": round(
                    cohen_kappa(g["production"].tolist(), g["xlmr_label"].tolist()), 4
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "metrics_summary.csv", index=False)

    # disagreement matrices production vs xlmr
    ct = pd.crosstab(df["production"], df["xlmr_label"], margins=True)
    ct.to_csv(OUT / "disagreement_matrix_production_vs_xlmr.csv")
    ct2 = pd.crosstab(df["production"], df["vader_label"], margins=True)
    ct2.to_csv(OUT / "disagreement_matrix_production_vs_vader.csv")
    ct3 = pd.crosstab(df["production"], df["lexicon_label"], margins=True)
    ct3.to_csv(OUT / "disagreement_matrix_production_vs_lexicon.csv")
    if len(sub):
        pd.crosstab(sub["production"], sub["bn_label"], margins=True).to_csv(
            OUT / "disagreement_matrix_production_vs_bn.csv"
        )

    # --- FP candidates ---
    fp_rows = []
    for _, row in df.iterrows():
        if row["production"] != "positive":
            continue
        judges = external_pack(row)
        non_pos = sum(1 for v in judges.values() if v != "positive")
        flags = [f for f in str(row["intent_flags"]).split(",") if f]
        suspect = non_pos >= 2 or bool(flags) or (
            row["xlmr_label"] != "positive" and pd.isna(row["star_rating"])
        )
        if not suspect:
            continue
        mode = failure_mode(flags, row["production"], judges)
        fp_rows.append(
            {
                "record_id": row.get("record_id"),
                "source": row.get("source"),
                "language": row.get("language"),
                "brand": row.get("brand"),
                "star_rating": row.get("star_rating"),
                "text": row["text"][:800],
                "production_label": row["production"],
                "lexicon_label": row["lexicon_label"],
                "xlmr_label": row["xlmr_label"],
                "vader_label": row["vader_label"],
                "bn_label": row["bn_label"],
                "intent_flags": row["intent_flags"],
                "lex_pos_hits": row["lex_pos_hits"],
                "failure_mode": mode,
                "n_judges_non_positive": non_pos,
            }
        )
    fp_df = pd.DataFrame(fp_rows)
    fp_df.to_csv(OUT / "false_positive_candidates.csv", index=False, encoding="utf-8-sig")

    # --- FN candidates ---
    fn_rows = []
    for _, row in df.iterrows():
        if row["production"] == "positive":
            continue
        judges = external_pack(row)
        n_pos = sum(1 for v in judges.values() if v == "positive")
        if n_pos < 2 and not (row["xlmr_label"] == "positive" and row["vader_label"] == "positive"):
            continue
        if row["xlmr_label"] != "positive":
            continue
        fn_rows.append(
            {
                "record_id": row.get("record_id"),
                "source": row.get("source"),
                "language": row.get("language"),
                "brand": row.get("brand"),
                "star_rating": row.get("star_rating"),
                "text": row["text"][:800],
                "production_label": row["production"],
                "lexicon_label": row["lexicon_label"],
                "xlmr_label": row["xlmr_label"],
                "vader_label": row["vader_label"],
                "bn_label": row["bn_label"],
                "intent_flags": row["intent_flags"],
                "failure_mode": "possible_false_negative",
            }
        )
    fn_df = pd.DataFrame(fn_rows)
    fn_df.to_csv(OUT / "false_negative_candidates.csv", index=False, encoding="utf-8-sig")

    # --- Star vs text conflicts ---
    star_rows = []
    rated = df[df["star_rating"].notna()].copy()
    for _, row in rated.iterrows():
        star_lab = STAR_MAP.get(int(row["star_rating"]))
        if star_lab != row["xlmr_label"] or star_lab != row["lexicon_label"]:
            star_rows.append(
                {
                    "record_id": row.get("record_id"),
                    "source": row.get("source"),
                    "star_rating": int(row["star_rating"]),
                    "production_label": row["production"],
                    "star_implied": star_lab,
                    "lexicon_label": row["lexicon_label"],
                    "xlmr_label": row["xlmr_label"],
                    "text": row["text"][:800],
                }
            )
    star_df = pd.DataFrame(star_rows)
    star_df.to_csv(OUT / "star_vs_text_conflicts.csv", index=False, encoding="utf-8-sig")

    # Query-like positive rate
    qpos = df[(df["production"] == "positive") & df["intent_flags"].str.contains("query", na=False)]
    stats = {
        "n_total": len(df),
        "n_production_positive": int((df["production"] == "positive").sum()),
        "n_production_neutral": int((df["production"] == "neutral").sum()),
        "n_production_negative": int((df["production"] == "negative").sum()),
        "lexicon_agree_no_star_pct": round(100 * float(lex_agree), 2),
        "n_fp_candidates": len(fp_df),
        "n_fn_candidates": len(fn_df),
        "n_star_text_conflicts": len(star_df),
        "n_query_like_positives": len(qpos),
        "fp_mode_counts": Counter(fp_df["failure_mode"].tolist() if len(fp_df) else []).most_common(),
        "score_value_counts": Counter(df["sentiment_score"].tolist()).most_common(10),
    }
    (OUT / "audit_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # Save scored frame (trimmed) for edge curation
    keep = [
        "record_id",
        "brand",
        "source",
        "language",
        "platform_type",
        "is_review",
        "star_rating",
        "text",
        "production",
        "sentiment_score",
        "lexicon_label",
        "lexicon_score",
        "xlmr_label",
        "xlmr_score",
        "vader_label",
        "vader_score",
        "bn_label",
        "bn_score",
        "intent_flags",
        "lex_pos_hits",
        "lex_neg_hits",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(OUT / "scored_corpus.csv", index=False, encoding="utf-8-sig")

    print("[done] metrics:")
    print(metrics_df.to_string(index=False))
    print(f"FP candidates: {len(fp_df)} | FN: {len(fn_df)} | star conflicts: {len(star_df)}")
    print(f"Wrote outputs under {OUT}")


if __name__ == "__main__":
    main()
