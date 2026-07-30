"""Multilingual sentiment analysis (HF model + policy + parallel rating signal)."""

from __future__ import annotations

from typing import Any, Optional

from listening.nlp.policy import apply_text_sentiment_policy, rating_sentiment_from_stars
from listening.normalize.schema import rating_to_sentiment

_PIPELINE = None
_MODEL_NAME: Optional[str] = None

LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}


def _get_pipeline(model_name: str):
    global _PIPELINE, _MODEL_NAME
    if _PIPELINE is not None and _MODEL_NAME == model_name:
        return _PIPELINE
    from transformers import pipeline

    _PIPELINE = pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )
    _MODEL_NAME = model_name
    return _PIPELINE


def _normalize_label(label: str) -> str:
    key = (label or "").strip().lower()
    return LABEL_MAP.get(key, "neutral")


def predict_sentiment(text: str, model_name: str) -> tuple[str, float]:
    """Raw model (or lexicon fallback) — policy applied in enrich_sentiment."""
    if not text or not text.strip():
        return "neutral", 0.0
    try:
        pipe = _get_pipeline(model_name)
        out = pipe(text[:2000])[0]
        label = _normalize_label(str(out.get("label", "neutral")))
        score = float(out.get("score") or 0.0)
        return label, score
    except Exception:
        return _lexicon_sentiment(text)


def _lexicon_sentiment(text: str) -> tuple[str, float]:
    t = text.lower()
    pos = ["good", "great", "excellent", "love", "best", "বালো", "ভালো", "দারুণ", "excellent", "helpful", "fast"]
    neg = ["bad", "worst", "hate", "scam", "fraud", "slow", "crash", "খারাপ", "বাজে", "সমস্যা", "problem", "error"]
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    if p > n:
        return "positive", min(1.0, 0.5 + 0.1 * (p - n))
    if n > p:
        return "negative", min(1.0, 0.5 + 0.1 * (n - p))
    return "neutral", 0.4


def enrich_sentiment(
    records: list[dict[str, Any]],
    model_name: str,
    use_rating_for_reviews: bool = True,
) -> list[dict[str, Any]]:
    """Attach parallel text + rating sentiment. Never overwrite text with stars."""
    for r in records:
        rating = r.get("rating")
        if rating is not None:
            r["rating_sentiment"] = rating_to_sentiment(rating)
        else:
            r["rating_sentiment"] = rating_sentiment_from_stars(r.get("star_rating"))

        text = r.get("text") or ""
        raw_label, raw_score = predict_sentiment(text, model_name)
        label, score, policy_tag = apply_text_sentiment_policy(text, raw_label, raw_score)
        r["sentiment_label"] = label
        r["sentiment_score"] = round(score, 4)
        r["sentiment_model_raw"] = raw_label
        r["sentiment_policy"] = policy_tag

        if use_rating_for_reviews and r.get("rating_sentiment"):
            r["sentiment_source"] = f"text+rating|{policy_tag}"
        else:
            r["sentiment_source"] = f"text|{policy_tag}"
    return records
