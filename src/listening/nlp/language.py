"""Language detection (Bangla / English / Banglish / und)."""

from __future__ import annotations

import re
from typing import Optional

_BN_RE = re.compile(r"[\u0980-\u09FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str | None) -> str:
    if not text or not str(text).strip():
        return "und"
    text = str(text)
    has_bn = bool(_BN_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))

    # Banglish heuristic: Latin script but common Bangla-romanized tokens or heavy mix
    banglish_tokens = {
        "kore",
        "korlam",
        "ache",
        "acche",
        "nai",
        "hoy",
        "hobe",
        "bhalo",
        "khub",
        "onek",
        "taka",
        "banker",
        "app ta",
        "otp",
        "lagche",
        "lagtese",
        "problem",
    }
    lower = text.lower()
    banglish_hits = sum(1 for t in banglish_tokens if t in lower)

    try:
        from langdetect import detect, DetectorFactory

        DetectorFactory.seed = 0
        # langdetect on short text is noisy; use as hint
        lid = detect(text) if len(text) >= 20 else None
    except Exception:
        lid = None

    if has_bn and has_latin:
        return "mixed"
    if has_bn and not has_latin:
        return "bn"
    if has_latin and banglish_hits >= 2:
        return "mixed"  # banglish treated as mixed for strata
    if lid == "bn":
        return "bn"
    if lid == "en":
        return "en"
    if has_latin:
        return "en"
    return "und"


def enrich_languages(records: list[dict]) -> list[dict]:
    for r in records:
        if not r.get("language"):
            r["language"] = detect_language(r.get("text"))
    return records
