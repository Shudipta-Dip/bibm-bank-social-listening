"""Rule-based intent / failure-mode flags for adversarial sentiment audit."""

from __future__ import annotations

import re
from typing import Iterable

# Posts that are usually informational / seeking help, not affective praise.
QUERY_RE = re.compile(
    r"(?is)"
    r"("
    r"\bhow\s+(do|can|to)\b|\bwhat\s+(is|are|should)\b|\bwhere\s+(can|do|to)\b|"
    r"\banyone\b|\bsuggest(ion|ed)?\b|\badvice\b|\brecommend(ation)?\b|"
    r"\bcan\s+i\b|\bcould\s+you\b|\bwhich\s+bank\b|\bshould\s+i\b|"
    r"পরামর্শ|কেউ\s*(বল|জান)|কিভাবে|কোন\s*ব্যাংক|সাজেস্ট|"
    r"বলতে\s*পারেন|জানতে\s*চাই|উপায়\s*কি"
    r")"
)

ADVICE_RE = re.compile(
    r"(?is)"
    r"("
    r"\btry\s+(ebl|brac|scb|mtb|city|dbbl)\b|"
    r"\bgo\s+for\b|\bmy\s+suggestion\b|\bdefinitely\s+a\s+good\s+choice\b|"
    r"ভালো\s*হবে|ট্রাই\s*কর|যান"
    r")"
)

COMPARISON_RE = re.compile(
    r"(?is)"
    r"("
    r"\bbetter\s+than\b|\bworse\s+than\b|\bvs\.?\b|\bcompared\s+to\b|"
    r"\blearn\s+(something\s+)?from\b|\bcopy\s+their\b|"
    r"থেকে\s*ভালো|চেয়ে\s*ভালো|বনাম"
    r")"
)

SWITCH_RE = re.compile(
    r"(?is)"
    r"("
    r"\bswitch\b|\bquit\s+scb\b|\bmove\s+to\b|\bleave\s+scb\b|"
    r"সুইচ|ট্রান্সফার\s*হতে|ছেড়ে"
    r")"
)

# Lexicon tokens that fire production positives (substring match, same as pipeline).
LEX_POS = [
    "good",
    "great",
    "excellent",
    "love",
    "best",
    "বালো",
    "ভালো",
    "দারুণ",
    "helpful",
    "fast",
]
LEX_NEG = [
    "bad",
    "worst",
    "hate",
    "scam",
    "fraud",
    "slow",
    "crash",
    "খারাপ",
    "বাজে",
    "সমস্যা",
    "problem",
    "error",
]

NEGATION_RE = re.compile(
    r"(?is)\b(not|never|no|isn't|aren't|wasn't|weren't|don't|doesn't|didn't)\b.{0,40}\b(good|great|best|excellent|love)\b"
    r"|\b(good|great|best|excellent)\b.{0,20}\b(not|never)\b"
)

STAR_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def lexicon_hits(text: str) -> tuple[list[str], list[str]]:
    t = (text or "").lower()
    pos = [w for w in LEX_POS if w in t]
    neg = [w for w in LEX_NEG if w in t]
    return pos, neg


def reproduce_lexicon(text: str) -> tuple[str, float]:
    """Mirror src/listening/nlp/sentiment.py::_lexicon_sentiment exactly."""
    if not text or not str(text).strip():
        return "neutral", 0.0
    t = str(text).lower()
    p = sum(1 for w in LEX_POS if w in t)
    n = sum(1 for w in LEX_NEG if w in t)
    # production list has duplicate "excellent"; count membership once per unique word
    # but production uses the list with duplicate — match production literally:
    pos_list = [
        "good",
        "great",
        "excellent",
        "love",
        "best",
        "বালো",
        "ভালো",
        "দারুণ",
        "excellent",
        "helpful",
        "fast",
    ]
    neg_list = [
        "bad",
        "worst",
        "hate",
        "scam",
        "fraud",
        "slow",
        "crash",
        "খারাপ",
        "বাজে",
        "সমস্যা",
        "problem",
        "error",
    ]
    p = sum(1 for w in pos_list if w in t)
    n = sum(1 for w in neg_list if w in t)
    if p > n:
        return "positive", min(1.0, 0.5 + 0.1 * (p - n))
    if n > p:
        return "negative", min(1.0, 0.5 + 0.1 * (n - p))
    return "neutral", 0.4


def intent_flags(text: str) -> list[str]:
    t = text or ""
    flags: list[str] = []
    if QUERY_RE.search(t):
        flags.append("query_or_advice_seeking")
    if ADVICE_RE.search(t):
        flags.append("advice_recommendation")
    if COMPARISON_RE.search(t):
        flags.append("comparison")
    if SWITCH_RE.search(t):
        flags.append("switch_intent")
    if NEGATION_RE.search(t):
        flags.append("negation_near_praise")
    pos_h, _ = lexicon_hits(t)
    if pos_h and ("query_or_advice_seeking" in flags or "?" in t):
        flags.append("lexicon_pos_in_question")
    if len(t.strip()) < 80 and pos_h:
        flags.append("short_lexicon_positive")
    return flags


def failure_mode(flags: Iterable[str], production: str, judges: dict[str, str]) -> str:
    flags = list(flags)
    non_pos_judges = sum(1 for v in judges.values() if v and v != "positive")
    if production == "positive" and "lexicon_pos_in_question" in flags:
        return "substring_false_positive_in_query"
    if production == "positive" and "advice_recommendation" in flags and non_pos_judges >= 1:
        return "advice_marked_positive"
    if production == "positive" and "query_or_advice_seeking" in flags:
        return "query_marked_positive"
    if production == "positive" and "negation_near_praise" in flags:
        return "negation_or_sarcasm"
    if production == "positive" and non_pos_judges >= 2:
        return "judge_disagreement_fp"
    if production != "positive" and sum(1 for v in judges.values() if v == "positive") >= 2:
        return "possible_false_negative"
    if "comparison" in flags and production == "positive":
        return "comparative_not_affective"
    if "switch_intent" in flags:
        return "switch_discourse"
    if flags:
        return "+".join(flags[:2])
    return "other_disagreement"
