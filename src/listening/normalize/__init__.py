"""Normalize raw collector payloads into UnifiedMention records."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from listening.normalize.schema import Engagement, UnifiedMention, build_mention
from listening.utils import DATA_RAW, cutoff_since, iter_jsonl, parse_iso


def _within_window(created_at: Optional[datetime], since: datetime) -> bool:
    if created_at is None:
        return True  # keep undated; flag later if needed
    return created_at >= since


def _repair_future_facebook_time(
    created: Optional[datetime], label: str | None
) -> Optional[datetime]:
    """Repair dateutil misreads of labels such as 'Name 41 weeks ago'."""
    now = datetime.now(timezone.utc)
    if not created or created <= now + timedelta(days=2):
        return created
    match = re.search(
        r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago$",
        label or "",
        re.I,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "minute":
        return now - timedelta(minutes=amount)
    if unit == "hour":
        return now - timedelta(hours=amount)
    days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return now - timedelta(days=amount * days)


def from_google_play(raw: dict[str, Any], brand: str, since: datetime) -> list[UnifiedMention]:
    out: list[UnifiedMention] = []
    review_id = str(raw.get("reviewId") or raw.get("id") or "")
    if not review_id:
        return out
    created = parse_iso(raw.get("at") or raw.get("date"))
    in_scope = _within_window(created, since)
    score = raw.get("score")
    text = raw.get("content") or raw.get("text") or ""
    title = raw.get("title") or ""
    body = f"{title}\n{text}".strip() if title else text
    eng = Engagement(helpful=raw.get("thumbsUpCount"), rating=score)
    out.append(
        build_mention(
            brand=brand,
            source="google_play",
            content_type="review",
            native_id=review_id,
            text=body,
            created_at=created,
            author_key=raw.get("userName") or raw.get("userImage"),
            author_type="user",
            url=None,
            engagement=eng,
            rating=int(score) if score is not None else None,
            collection_method=raw.get("_collection_method", "http_lib"),
            raw=raw,
            in_scope=in_scope,
            thread_id=review_id,
        )
    )
    reply = raw.get("replyContent")
    if reply:
        reply_at = parse_iso(raw.get("repliedAt"))
        out.append(
            build_mention(
                brand=brand,
                source="google_play",
                content_type="review_reply",
                native_id=f"{review_id}::reply",
                parent_native_id=review_id,
                thread_id=review_id,
                text=reply,
                created_at=reply_at,
                author_type="page",
                author_key=f"{brand}_play_reply",
                collection_method=raw.get("_collection_method", "http_lib"),
                raw={"replyContent": reply, "repliedAt": raw.get("repliedAt"), "parent": review_id},
                in_scope=_within_window(reply_at or created, since),
            )
        )
    return out


def from_app_store(raw: dict[str, Any], brand: str, since: datetime) -> list[UnifiedMention]:
    out: list[UnifiedMention] = []
    review_id = str(raw.get("id") or raw.get("reviewId") or "")
    if not review_id:
        return out
    created = parse_iso(raw.get("date") or raw.get("updated"))
    in_scope = _within_window(created, since)
    score = raw.get("rating") or raw.get("score")
    title = raw.get("title") or ""
    text = raw.get("review") or raw.get("content") or raw.get("text") or ""
    body = f"{title}\n{text}".strip() if title else text
    eng = Engagement(helpful=raw.get("voteCount") or raw.get("voteSum"), rating=score)
    out.append(
        build_mention(
            brand=brand,
            source="app_store",
            content_type="review",
            native_id=review_id,
            text=body,
            created_at=created,
            author_key=raw.get("userName") or raw.get("author"),
            author_type="user",
            url=raw.get("url"),
            engagement=eng,
            rating=int(score) if score is not None else None,
            collection_method=raw.get("_collection_method", "http_lib"),
            coverage_note=raw.get("_coverage_note"),
            raw=raw,
            in_scope=in_scope,
            thread_id=review_id,
        )
    )
    developer_response = raw.get("developerResponse") or raw.get("response")
    if isinstance(developer_response, dict):
        reply_text = developer_response.get("body") or developer_response.get("text")
        reply_at = parse_iso(developer_response.get("modified") or developer_response.get("date"))
    else:
        reply_text = developer_response if isinstance(developer_response, str) else None
        reply_at = None
    if reply_text:
        out.append(
            build_mention(
                brand=brand,
                source="app_store",
                content_type="review_reply",
                native_id=f"{review_id}::reply",
                parent_native_id=review_id,
                thread_id=review_id,
                text=reply_text,
                created_at=reply_at,
                author_type="page",
                author_key=f"{brand}_appstore_reply",
                collection_method=raw.get("_collection_method", "http_lib"),
                raw={"developerResponse": developer_response, "parent": review_id},
                in_scope=_within_window(reply_at or created, since),
            )
        )
    return out


def from_facebook(raw: dict[str, Any], brand: str, since: datetime) -> list[UnifiedMention]:
    out: list[UnifiedMention] = []
    content_type = raw.get("_content_type") or ("comment" if raw.get("parent_id") or raw.get("_parent_id") else "post")
    native_id = str(raw.get("id") or "")
    if not native_id:
        return out
    text = raw.get("message") or raw.get("text") or raw.get("story") or ""
    created = parse_iso(raw.get("created_time") or raw.get("created_at"))
    created = _repair_future_facebook_time(created, raw.get("created_label"))
    from_obj = raw.get("from") or {}
    author_key = (
        from_obj.get("id") if isinstance(from_obj, dict) else None
    ) or raw.get("author_id")
    author_type = "page" if raw.get("_is_page") else ("user" if author_key else "unknown")
    parent_id = raw.get("_parent_id") or raw.get("parent_id")
    thread_id = raw.get("_thread_id") or parent_id or native_id
    likes = None
    if isinstance(raw.get("reactions"), dict):
        likes = raw["reactions"].get("summary", {}).get("total_count")
    elif raw.get("like_count") is not None:
        likes = raw.get("like_count")
    comments_count = None
    if isinstance(raw.get("comments"), dict):
        comments_count = raw["comments"].get("summary", {}).get("total_count")
    elif raw.get("comment_count") is not None:
        comments_count = raw.get("comment_count")
    shares = None
    if isinstance(raw.get("shares"), dict):
        shares = raw["shares"].get("count")
    elif raw.get("share_count") is not None:
        shares = raw.get("share_count")
    eng = Engagement(likes=likes, comments=comments_count, shares=shares)
    out.append(
        build_mention(
            brand=brand,
            source="facebook",
            content_type=content_type if content_type in ("post", "comment") else "post",
            native_id=native_id,
            parent_native_id=parent_id,
            thread_id=thread_id,
            text=text,
            created_at=created,
            author_key=str(author_key) if author_key else None,
            author_type=author_type,
            url=raw.get("permalink_url") or raw.get("url"),
            engagement=eng,
            collection_method=raw.get("_collection_method", "api"),
            raw=raw,
            in_scope=_within_window(created, since),
        )
    )
    return out


def from_linkedin(raw: dict[str, Any], brand: str, since: datetime) -> list[UnifiedMention]:
    out: list[UnifiedMention] = []
    content_type = raw.get("_content_type") or "post"
    native_id = str(raw.get("id") or raw.get("urn") or "")
    if not native_id:
        return out
    text = raw.get("commentary") or raw.get("text") or raw.get("message") or ""
    created = parse_iso(raw.get("created_at") or raw.get("published_at") or raw.get("date"))
    parent_id = raw.get("_parent_id")
    eng = Engagement(
        likes=raw.get("num_likes") or raw.get("likes"),
        comments=raw.get("num_comments") or raw.get("comments_count"),
        shares=raw.get("num_shares") or raw.get("shares"),
    )
    out.append(
        build_mention(
            brand=brand,
            source="linkedin",
            content_type=content_type if content_type in ("post", "comment") else "post",
            native_id=native_id,
            parent_native_id=parent_id,
            thread_id=parent_id or native_id,
            text=text,
            created_at=created,
            author_key=raw.get("author_id") or raw.get("author"),
            author_type=raw.get("author_type") or "unknown",
            url=raw.get("url"),
            engagement=eng,
            collection_method=raw.get("_collection_method", "browser"),
            raw=raw,
            in_scope=_within_window(created, since),
        )
    )
    return out


def from_reddit(raw: dict[str, Any], brand: str, since: datetime) -> list[UnifiedMention]:
    out: list[UnifiedMention] = []
    content_type = raw.get("_content_type") or "post"
    native_id = str(raw.get("id") or "")
    if not native_id:
        return out
    if content_type == "comment":
        text = raw.get("body") or raw.get("text") or ""
        parent_id = raw.get("_parent_id") or raw.get("parent_id")
        thread_id = raw.get("_thread_id") or (str(raw.get("link_id") or "").replace("t3_", "") or None)
    else:
        title = (raw.get("title") or "").strip()
        body = (raw.get("selftext") or raw.get("text") or "").strip()
        text = f"{title}\n{body}".strip() if title and body else (title or body)
        parent_id = None
        thread_id = native_id
    created = parse_iso(raw.get("created_at") or raw.get("created_utc"))
    eng = Engagement(
        likes=raw.get("score"),
        comments=raw.get("num_comments"),
    )
    out.append(
        build_mention(
            brand=brand,
            source="reddit",
            content_type=content_type if content_type in ("post", "comment") else "post",
            native_id=native_id,
            parent_native_id=str(parent_id) if parent_id else None,
            thread_id=str(thread_id) if thread_id else native_id,
            text=text,
            created_at=created,
            author_key=raw.get("author"),
            author_type="user",
            url=raw.get("permalink") or raw.get("url"),
            engagement=eng,
            collection_method=raw.get("_collection_method", "http_lib"),
            raw=raw,
            in_scope=_within_window(created, since),
        )
    )
    return out


ADAPTERS = {
    "google_play": from_google_play,
    "app_store": from_app_store,
    "facebook": from_facebook,
    "linkedin": from_linkedin,
    "reddit": from_reddit,
}


def normalize_raw_row(raw: dict[str, Any], source: str, brand: str, since: datetime) -> list[UnifiedMention]:
    adapter = ADAPTERS.get(source)
    if not adapter:
        return []
    return adapter(raw, brand, since)


def load_all_raw(source: str | None = None, brand: str | None = None) -> list[tuple[str, str, dict]]:
    """Yield (source, brand, raw_dict) from data/raw/**/*.jsonl."""
    rows: list[tuple[str, str, dict]] = []
    if not DATA_RAW.exists():
        return rows
    for path in sorted(DATA_RAW.rglob("*.jsonl")):
        parts = path.relative_to(DATA_RAW).parts
        if len(parts) < 2:
            continue
        src, br = parts[0], parts[1]
        if source and src != source:
            continue
        if brand and br != brand:
            continue
        for raw in iter_jsonl(path):
            rows.append((src, br, raw))
    return rows


def normalize_corpus(
    since: datetime,
    source: str | None = None,
    brand: str | None = None,
    drop_empty: bool = True,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for src, br, raw in load_all_raw(source=source, brand=brand):
        for mention in normalize_raw_row(raw, src, br, since):
            if drop_empty and not (mention.text or "").strip():
                # keep posts that are image-only? plan says drop empty text
                if mention.content_type in ("post", "comment", "review"):
                    continue
            if mention.record_id in seen:
                continue
            seen.add(mention.record_id)
            out.append(mention.to_dict())
    return out
