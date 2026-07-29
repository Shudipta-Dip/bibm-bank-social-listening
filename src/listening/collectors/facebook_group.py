"""Collect posts and comments from configured Facebook group searches via CDP.

The user launches and authenticates a dedicated Chrome profile. This collector
attaches to that existing browser on localhost; it never submits credentials,
solves challenges, reacts, comments, or otherwise writes to Facebook.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from listening.collectors import (
    CollectorResult,
    format_exc,
    log_run,
    new_run_id,
    write_raw_items,
)
from listening.hitl import GATE_H3_BLOCKER, HitlBlockedError, raise_gate
from listening.utils import (
    ROOT,
    load_checkpoint,
    save_checkpoint,
    utc_now,
)

DEFAULT_CDP_URL = "http://127.0.0.1:9222"
POST_ID_RE = re.compile(r"/posts/(\d+)")
COMMENT_ID_RE = re.compile(r"[?&]comment_id=(\d+)")
COUNT_RE = re.compile(r"([\d,.]+)\s*(comments?|reactions?|likes?)", re.I)


def _safe_cdp_url() -> str:
    endpoint = os.getenv("FACEBOOK_CDP_URL", DEFAULT_CDP_URL).strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError("FACEBOOK_CDP_URL must be an HTTP(S) or WS(S) URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("For session safety, FACEBOOK_CDP_URL must be localhost")
    return endpoint


def _author_hash(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _canonical_post_url(href: str, post_id: str) -> str:
    parsed = urlparse(href)
    match = re.search(r"(/groups/[^/]+/posts/\d+/)", parsed.path)
    path = match.group(1) if match else f"/groups/bcub.bd/posts/{post_id}/"
    return f"https://www.facebook.com{path}"


def _parse_count(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"([\d,.]+)", value)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    try:
        return int(float(number))
    except ValueError:
        return None


def _parse_facebook_time(label: str | None, now: datetime | None = None) -> datetime | None:
    """Parse common Facebook labels (1h, 2d, July 20 at 3:12 PM)."""
    if not label:
        return None
    now = now or utc_now()
    text = label.strip()
    short = re.fullmatch(r"(\d+)\s*([mhdw])", text, re.I)
    if short:
        amount = int(short.group(1))
        unit = short.group(2).lower()
        delta = {
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }[unit]
        return now - delta
    if text.lower() in {"just now", "now"}:
        return now
    long_relative = re.search(
        r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago$",
        text,
        re.I,
    )
    if long_relative:
        amount = int(long_relative.group(1))
        unit = long_relative.group(2).lower()
        days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit)
        if days:
            return now - timedelta(days=amount * days)
        delta = (
            timedelta(minutes=amount)
            if unit == "minute"
            else timedelta(hours=amount)
        )
        return now - delta

    cleaned = re.sub(r"\s+at\s+", " ", text, flags=re.I)
    cleaned = re.sub(r"\s+·\s+.*$", "", cleaned)
    candidates = [cleaned, f"{cleaned} {now.year}"]
    from dateutil import parser as date_parser

    for candidate in candidates:
        try:
            parsed = date_parser.parse(candidate, fuzzy=True)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
            # Month/day labels omit year. If they land in the future, use last year.
            if parsed > now + timedelta(days=2):
                parsed = parsed.replace(year=parsed.year - 1)
            return parsed
        except (ValueError, OverflowError):
            continue
    return None


def _challenge_reason(page) -> str | None:
    url = page.url.lower()
    if "checkpoint" in url:
        return "Facebook checkpoint detected"
    if "captcha" in url:
        return "Facebook CAPTCHA detected"
    try:
        body = page.inner_text("body", timeout=3000)[:6000].lower()
    except Exception:
        body = ""
    markers = {
        "captcha": "Facebook CAPTCHA detected",
        "security check": "Facebook security check detected",
        "confirm your identity": "Facebook identity confirmation detected",
        "account temporarily locked": "Facebook account lock detected",
        "we noticed a new login": "Facebook new-login review notice detected",
    }
    for marker, reason in markers.items():
        if marker in body:
            return reason
    return None


def _pause_on_challenge(page, brand: str, run_id: str) -> None:
    reason = _challenge_reason(page)
    if not reason:
        return
    screenshot = ROOT / "data" / "hitl" / f"facebook_group_{brand}_{run_id}.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(screenshot), full_page=False)
    except Exception:
        pass
    raise_gate(
        GATE_H3_BLOCKER,
        f"{reason}. Collection stopped; review it manually in Chrome before resuming.",
        brand=brand,
        source="facebook",
        payload={"screenshot": str(screenshot), "url": page.url},
    )


def _click_matching_buttons(container, patterns: tuple[str, ...], limit: int) -> int:
    clicked = 0
    try:
        buttons = container.locator('[role="button"]')
        labels = buttons.evaluate_all(
            """els => els.slice(0, 100).map((el, index) => ({
                index,
                label: (
                    el.getAttribute('aria-label') ||
                    el.innerText ||
                    ''
                ).trim(),
                visible: !!(
                    el.offsetWidth ||
                    el.offsetHeight ||
                    el.getClientRects().length
                )
            }))"""
        )
    except Exception:
        return 0
    for item in labels:
        if clicked >= limit:
            break
        label = item["label"]
        if not item["visible"] or not any(
            re.search(pattern, label, re.I) for pattern in patterns
        ):
            continue
        button = buttons.nth(item["index"])
        try:
            button.click(timeout=1500)
            clicked += 1
            time.sleep(random.uniform(0.8, 1.5))
        except Exception:
            continue
    return clicked


def _extract_search_cards(page, brand: str, query: str, group_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    articles = page.locator('[role="article"]')
    for index in range(articles.count()):
        article = articles.nth(index)
        try:
            post_link = article.locator(
                'a[href*="/groups/"][href*="/posts/"]'
            ).first
            if post_link.count() == 0:
                continue
            href = post_link.get_attribute("href") or ""
            post_match = POST_ID_RE.search(href)
            if not post_match:
                continue
            post_id = post_match.group(1)

            _click_matching_buttons(article, (r"^see more$",), limit=2)

            message = ""
            for selector in (
                '[data-ad-preview="message"]',
                '[data-ad-comet-preview="message"]',
            ):
                candidate = article.locator(selector)
                if candidate.count():
                    message = (candidate.first.inner_text(timeout=1200) or "").strip()
                    if message:
                        break
            if not message:
                candidates = article.locator('div[dir="auto"]')
                texts: list[str] = []
                for candidate_index in range(min(candidates.count(), 30)):
                    try:
                        text = (candidates.nth(candidate_index).inner_text(timeout=500) or "").strip()
                        if 20 <= len(text) <= 12000:
                            texts.append(text)
                    except Exception:
                        continue
                message = max(texts, key=len, default="")
            if not message:
                continue

            time_label = post_link.get_attribute("aria-label") or post_link.inner_text(timeout=500)
            created = _parse_facebook_time(time_label)

            action = article.locator('[aria-label^="Actions for this post by "]')
            action_label = action.first.get_attribute("aria-label") if action.count() else None
            author_display = None
            if action_label:
                author_display = action_label.removeprefix("Actions for this post by ").strip()

            article_text = article.inner_text(timeout=1500) or ""
            comment_count = None
            reaction_count = None
            for count_text, kind in COUNT_RE.findall(article_text):
                count = _parse_count(count_text)
                if kind.lower().startswith("comment"):
                    comment_count = count
                elif kind.lower().startswith(("reaction", "like")):
                    reaction_count = count

            results.append(
                {
                    "id": post_id,
                    "message": message,
                    "created_time": created.isoformat() if created else None,
                    "created_label": time_label,
                    "permalink_url": _canonical_post_url(href, post_id),
                    "author_id": _author_hash(author_display),
                    "comment_count": comment_count,
                    "like_count": reaction_count,
                    "_content_type": "post",
                    "_thread_id": post_id,
                    "_collection_method": "browser_cdp",
                    "_collection_context": "facebook_group_search",
                    "_group_id": group_id,
                    "_group_name": "Bank Card Users of Bangladesh (BCUB)",
                    "_query": query,
                    "_collected_at": utc_now().isoformat(),
                }
            )
        except Exception:
            continue
    return results


def _expand_comments(page, rounds: int) -> None:
    patterns = (
        r"view more comments",
        r"view previous comments",
        r"view \d+ more comments",
        r"more comments",
    )
    for _ in range(rounds):
        clicked = _click_matching_buttons(page, patterns, limit=3)
        if clicked == 0:
            break
        time.sleep(random.uniform(1.5, 2.5))


def _extract_comments(page, post_id: str, brand: str, max_comments: int) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    article_data = page.evaluate(
        """() => [...document.querySelectorAll('[role="article"]')].map(article => ({
            hrefs: [...article.querySelectorAll('a[href*="comment_id="]')]
                .map(link => link.href),
            linkTexts: [...article.querySelectorAll('a[href*="comment_id="]')]
                .map(link => (link.innerText || '').trim()),
            texts: [...article.querySelectorAll('div[dir="auto"]')]
                .map(node => (node.innerText || '').trim())
                .filter(text => text.length >= 2 && text.length <= 8000),
            labels: [...article.querySelectorAll('[aria-label^="Comment by "]')]
                .map(node => node.getAttribute('aria-label'))
                .filter(Boolean)
        }))"""
    )
    for article in article_data:
        if len(results) >= max_comments:
            break
        try:
            if not article["hrefs"]:
                continue
            article_comment_ids = {
                match.group(1)
                for href_value in article["hrefs"]
                if (match := COMMENT_ID_RE.search(href_value))
            }
            # A wrapping post article contains every loaded comment. Keep only
            # the inner article representing one comment to avoid post-text leaks.
            if len(article_comment_ids) != 1:
                continue
            href = article["hrefs"][0]
            match = COMMENT_ID_RE.search(href)
            if not match:
                continue
            comment_id = match.group(1)
            if comment_id in results:
                continue

            body = max(article["texts"], key=len, default="")
            if not body:
                continue

            label = article["labels"][0] if article["labels"] else None
            author_display = None
            created_label = None
            if label:
                match_label = re.match(r"Comment by (.+?)\s+(.+? ago)$", label)
                if match_label:
                    author_display, created_label = match_label.groups()
            if not created_label:
                created_label = (
                    article["linkTexts"][0] if article["linkTexts"] else None
                )
            created = _parse_facebook_time(created_label)

            results[comment_id] = {
                "id": comment_id,
                "message": body,
                "created_time": created.isoformat() if created else None,
                "created_label": created_label,
                "permalink_url": href.split("&__cft__")[0],
                "author_id": _author_hash(author_display),
                "like_count": None,
                "_content_type": "comment",
                "_parent_id": post_id,
                "_thread_id": post_id,
                "_collection_method": "browser_cdp",
                "_collection_context": "facebook_group_search",
                "_group_name": "Bank Card Users of Bangladesh (BCUB)",
                "_collected_at": utc_now().isoformat(),
            }
        except Exception:
            continue
    return list(results.values())


def _crawl_search(
    page,
    *,
    brand: str,
    search_url: str,
    query: str,
    group_id: str,
    since: datetime,
    run_id: str,
    max_posts: int,
    scroll_rounds: int,
    max_comments_per_post: int,
) -> list[dict[str, Any]]:
    page.goto(search_url, wait_until="domcontentloaded", timeout=120000)
    time.sleep(random.uniform(4.0, 6.0))
    _pause_on_challenge(page, brand, run_id)

    posts: dict[str, dict[str, Any]] = {}
    unchanged_rounds = 0
    old_rounds = 0
    for _ in range(scroll_rounds):
        before = len(posts)
        cards = _extract_search_cards(page, brand, query, group_id)
        visible_dates = []
        for card in cards:
            created_text = card.get("created_time")
            created = datetime.fromisoformat(created_text) if created_text else None
            if created:
                visible_dates.append(created)
            # Explicitly retain only the configured 12-month window where known.
            if created and created < since:
                continue
            posts[card["id"]] = card
            if len(posts) >= max_posts:
                break
        if len(posts) >= max_posts:
            break
        if visible_dates and max(visible_dates) < since:
            old_rounds += 1
        else:
            old_rounds = 0
        if old_rounds >= 2:
            break

        unchanged_rounds = unchanged_rounds + 1 if len(posts) == before else 0
        if unchanged_rounds >= 4:
            break
        page.mouse.wheel(0, random.randint(3500, 5200))
        time.sleep(random.uniform(2.8, 5.2))
        _pause_on_challenge(page, brand, run_id)

    rows: list[dict[str, Any]] = list(posts.values())
    for position, post in enumerate(posts.values(), start=1):
        if max_comments_per_post <= 0:
            break
        page.goto(post["permalink_url"], wait_until="domcontentloaded", timeout=120000)
        time.sleep(random.uniform(3.5, 5.5))
        _pause_on_challenge(page, brand, run_id)
        _click_matching_buttons(page, (r"^see more$",), limit=3)
        _expand_comments(page, rounds=6)
        comments = _extract_comments(
            page,
            post["id"],
            brand,
            max_comments=max_comments_per_post,
        )
        rows.extend(comments)
        print(
            f"[facebook-group] {brand} {position}/{len(posts)} "
            f"post={post['id']} comments={len(comments)}",
            flush=True,
        )
        time.sleep(random.uniform(2.5, 4.5))
    return rows


def collect_facebook_group(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    config: dict[str, Any] | None = None,
) -> CollectorResult:
    """Collect configured BCUB search results for one bank."""
    run_id = new_run_id()
    result = CollectorResult(source="facebook_group", brand=brand, run_id=run_id)
    cfg = config or {}
    global_cfg = cfg.get("facebook_group") or {}
    brand_cfg_group = brand_cfg.get("facebook_group") or {}
    searches = brand_cfg_group.get("searches") or []
    group_id = str(global_cfg.get("group_id") or "2248656405437853")
    max_posts = int(global_cfg.get("max_posts_per_brand") or 100)
    scroll_rounds = int(global_cfg.get("scroll_rounds") or 30)
    max_comments = int(global_cfg.get("max_comments_per_post") or 50)

    if not searches:
        result.error_summary = f"No facebook_group.searches configured for {brand}"
        log_run(result.finish("error"))
        return result

    all_rows: dict[tuple[str, str], dict[str, Any]] = {}
    page = None
    browser = None
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(_safe_cdp_url())
            if not browser.contexts:
                raise RuntimeError("Attached Chrome has no browser context")
            context = browser.contexts[0]
            page = context.new_page()

            for search in searches:
                search_url = str(search.get("url") or "")
                query = str(search.get("query") or "")
                if not search_url:
                    continue
                print(f"[facebook-group] {brand}: query={query!r}", flush=True)
                rows = _crawl_search(
                    page,
                    brand=brand,
                    search_url=search_url,
                    query=query,
                    group_id=group_id,
                    since=since,
                    run_id=run_id,
                    max_posts=max_posts,
                    scroll_rounds=scroll_rounds,
                    max_comments_per_post=max_comments,
                )
                for row in rows:
                    all_rows[(row["_content_type"], row["id"])] = row
                save_checkpoint(
                    "facebook_group",
                    brand,
                    {
                        "complete": False,
                        "run_id": run_id,
                        "last_query": query,
                        "items_so_far": len(all_rows),
                        "updated_at": utc_now().isoformat(),
                    },
                )
            page.close()

        rows = list(all_rows.values())
        # Store under source=facebook so the existing normalizer applies.
        result.item_count = write_raw_items("facebook", brand, run_id, rows)
        result.meta = {
            "group_id": group_id,
            "group_name": "Bank Card Users of Bangladesh (BCUB)",
            "queries": [s.get("query") for s in searches],
            "posts": sum(1 for r in rows if r["_content_type"] == "post"),
            "comments": sum(1 for r in rows if r["_content_type"] == "comment"),
            "method": "browser_cdp",
        }
        save_checkpoint(
            "facebook_group",
            brand,
            {
                "complete": True,
                "last_run_id": run_id,
                "last_count": result.item_count,
                "updated_at": utc_now().isoformat(),
            },
        )
    except HitlBlockedError as exc:
        result.hitl_flags.append(exc.gate_id)
        result.error_summary = str(exc)
        result.status = "hitl_blocked"
        if page:
            try:
                page.close()
            except Exception:
                pass
        if all_rows:
            result.item_count = write_raw_items(
                "facebook", brand, run_id, list(all_rows.values())
            )
            result.status = "partial"
    except Exception as exc:
        result.error_summary = format_exc(exc)
        result.status = "error"
        if page:
            try:
                page.close()
            except Exception:
                pass
        if all_rows:
            result.item_count = write_raw_items(
                "facebook", brand, run_id, list(all_rows.values())
            )
            result.status = "partial"

    log_run(result.finish())
    return result
