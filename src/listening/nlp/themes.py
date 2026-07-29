"""Keyword/theme tagging for banking topics (EN + BN)."""

from __future__ import annotations

from typing import Any


def tag_themes(text: str | None, theme_lexicon: dict[str, list[str]]) -> list[str]:
    if not text:
        return []
    lower = text.lower()
    hits = []
    for theme, keywords in (theme_lexicon or {}).items():
        for kw in keywords:
            if kw.lower() in lower:
                hits.append(theme)
                break
    return hits


def enrich_themes(records: list[dict[str, Any]], theme_lexicon: dict[str, list[str]]) -> list[dict[str, Any]]:
    for r in records:
        r["themes"] = tag_themes(r.get("text"), theme_lexicon)
    return records
