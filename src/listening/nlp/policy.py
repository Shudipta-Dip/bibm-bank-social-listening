"""Post-model sentiment policy (adversarial audit + human gold).

Rules (2026-07-30 human edge-case pack):
- Keep text sentiment and star/rating sentiment as *parallel* metrics (never merge).
- Prefer transformer text label; apply intent overrides below.
- Query / advice-seeking that looks positive → neutral (not praise).
- Congratulatory / bank-PR social posts → neutral, except clear customer-gift praise → positive.
"""

from __future__ import annotations

import re
from typing import Optional

QUERY_OR_ADVICE_RE = re.compile(
    r"(?is)"
    r"("
    r"\bhow\s+(do|can|to)\b|\bwhat\s+(is|are|should)\b|\bwhere\s+(can|do|to)\b|"
    r"\banyone\b|\bsuggest(ion|ed)?\b|\badvice\b|\brecommend(ation)?\b|"
    r"\bcan\s+i\b|\bcould\s+you\b|\bwhich\s+bank\b|\bshould\s+i\b|"
    r"\bwhich\s+(one|is)\b|\bkonta\b|"
    r"পরামর্শ|কেউ\s*(বল|জান)|কিভাবে|কোন\s*ব্যাংক|কোনটা|সাজেস্ট|"
    r"বলতে\s*পারেন|জানতে\s*চাই|উপায়\s*কি|ভালো\s*হবে\s*\?"
    r")"
)

# Bank PR / congratulations — usually not customer affect.
PR_NEUTRAL_RE = re.compile(
    r"(?is)"
    r"("
    r"\bcongratulations?\b|"
    r"\bcongratulation\b|"
    r"\bchief\s+technology\s+officer\b|\b\(?\s*cto\s*\)?\b|"
    r"has\s+been\s+part\s+of\b.{0,40}\bjourney\b|"
    r"helping\s+build\s+the\s+technology\s+foundation\b|"
    r"see\s+more\s*$"
    r")"
)

# Customer benefit / gift — keep or force positive.
CUSTOMER_GIFT_RE = re.compile(
    r"(?is)"
    r"("
    r"\ba\s+gift\s+from\b|"
    r"\bgift\s+from\s+(standard\s+chartered|scb|brac)\b|"
    r"অফার|ক্যাশব্যাক|cash\s*back|welcome\s+bonus"
    r")"
)

STAR_TO_RATING_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def rating_sentiment_from_stars(star: Optional[float | int]) -> Optional[str]:
    if star is None:
        return None
    try:
        if star != star:  # NaN
            return None
        return STAR_TO_RATING_SENTIMENT.get(int(star))
    except (TypeError, ValueError):
        return None


def apply_text_sentiment_policy(
    text: str,
    model_label: str,
    model_score: float = 0.0,
) -> tuple[str, float, str]:
    """Return (label, score, policy_tag) after human-aligned overrides.

    Does not use star ratings — those stay in ``rating_sentiment``.
    """
    label = (model_label or "neutral").strip().lower()
    if label not in {"positive", "negative", "neutral"}:
        label = "neutral"
    score = float(model_score or 0.0)
    t = text or ""
    tag = "model"

    # Customer gift / clear benefit → positive (even if short / PR-adjacent).
    if CUSTOMER_GIFT_RE.search(t):
        return "positive", max(score, 0.7), "policy_customer_gift"

    # Congrats / executive PR → neutral (not customer sentiment).
    if PR_NEUTRAL_RE.search(t):
        return "neutral", min(score, 0.55) if score else 0.5, "policy_pr_neutral"

    # Questions / advice-seeking marked positive by the model → neutral.
    # Keep model negative (complaint inside a question still counts).
    if label == "positive" and QUERY_OR_ADVICE_RE.search(t):
        return "neutral", min(score, 0.55) if score else 0.5, "policy_query_to_neutral"

    # Bare "?" + positive with no strong affect length — soft guard for BN questions
    # that missed the regex but are short asks.
    if (
        label == "positive"
        and "?" in t
        and len(t.strip()) < 120
        and re.search(r"(?i)ভালো|good|best|better", t)
    ):
        return "neutral", min(score, 0.55) if score else 0.5, "policy_short_question_to_neutral"

    return label, score, tag
