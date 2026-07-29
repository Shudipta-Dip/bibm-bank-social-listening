"""UnifiedMention schema and helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from listening.utils import author_hash, make_record_id, normalize_text, utc_now


BRANDS = ("brac_bank", "scb_bangladesh")
SOURCES = ("facebook", "linkedin", "google_play", "app_store", "reddit")
CONTENT_TYPES = ("post", "comment", "review", "review_reply")
AUTHOR_TYPES = ("user", "page", "unknown")
SENTIMENTS = ("positive", "neutral", "negative")
METHODS = ("api", "http_lib", "browser", "manual_export")


@dataclass
class Engagement:
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    rating: Optional[int] = None
    helpful: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedMention:
    record_id: str
    brand: str
    source: str
    content_type: str
    native_id: str
    text: str
    collected_at: str
    parent_native_id: Optional[str] = None
    thread_id: Optional[str] = None
    author_hash: Optional[str] = None
    author_type: str = "unknown"
    text_original: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[str] = None
    url: Optional[str] = None
    engagement: dict[str, Any] = field(default_factory=dict)
    rating: Optional[int] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    rating_sentiment: Optional[str] = None
    themes: list[str] = field(default_factory=list)
    in_scope: bool = True
    collection_method: str = "http_lib"
    coverage_note: Optional[str] = None
    raw_json: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rating_to_sentiment(rating: int | None) -> Optional[str]:
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    if rating <= 2:
        return "negative"
    return None


def build_mention(
    *,
    brand: str,
    source: str,
    content_type: str,
    native_id: str,
    text: str,
    created_at: datetime | str | None = None,
    parent_native_id: str | None = None,
    thread_id: str | None = None,
    author_key: str | None = None,
    author_type: str = "unknown",
    url: str | None = None,
    engagement: Engagement | dict | None = None,
    rating: int | None = None,
    collection_method: str = "http_lib",
    coverage_note: str | None = None,
    raw: Any = None,
    in_scope: bool = True,
    text_original: str | None = None,
) -> UnifiedMention:
    text_n = normalize_text(text)
    eng = engagement.to_dict() if isinstance(engagement, Engagement) else (engagement or {})
    created = None
    if created_at is not None:
        if isinstance(created_at, datetime):
            created = created_at.isoformat()
        else:
            created = str(created_at)

    return UnifiedMention(
        record_id=make_record_id(source, brand, native_id),
        brand=brand,
        source=source,
        content_type=content_type,
        native_id=str(native_id),
        parent_native_id=str(parent_native_id) if parent_native_id else None,
        thread_id=str(thread_id) if thread_id else None,
        author_hash=author_hash(author_key),
        author_type=author_type if author_type in AUTHOR_TYPES else "unknown",
        text=text_n,
        text_original=text_original,
        created_at=created,
        collected_at=utc_now().isoformat(),
        url=url,
        engagement=eng,
        rating=rating,
        rating_sentiment=rating_to_sentiment(rating),
        in_scope=in_scope,
        collection_method=collection_method,
        coverage_note=coverage_note,
        raw_json=json.dumps(raw if raw is not None else {}, ensure_ascii=False, default=str),
    )
