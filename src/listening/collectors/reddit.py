"""Reddit collector via free public JSON search (no API token required)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from listening.collectors import CollectorResult, format_exc, log_run, new_run_id, write_raw_items
from listening.utils import parse_iso, save_checkpoint, utc_now

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Try mirrors: www is often blocked for datacenter/scripted clients
API_BASES = (
    "https://old.reddit.com",
    "https://www.reddit.com",
    "https://api.reddit.com",
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    resp = requests.get(
        url,
        params=params,
        timeout=45,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if resp.status_code == 429:
        time.sleep(15)
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def _get_json_multi(path: str, params: dict[str, Any] | None = None) -> Any:
    """Try several Reddit hosts until one succeeds. Fail fast on 403."""
    last_exc: Exception | None = None
    for base in API_BASES:
        url = f"{base}{path}"
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=30,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            if resp.status_code == 403:
                last_exc = requests.HTTPError(f"403 Blocked for url: {resp.url}")
                continue
            if resp.status_code == 429:
                time.sleep(10)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
            continue
    assert last_exc is not None
    raise last_exc


def _permalink(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http"):
        return path
    return "https://www.reddit.com" + path


def _submission_row(child: dict[str, Any], subreddit: str, query: str) -> dict[str, Any] | None:
    data = child.get("data") or {}
    pid = data.get("id") or data.get("name")
    if not pid:
        return None
    created = data.get("created_utc")
    text_body = (data.get("selftext") or "").strip()
    title = (data.get("title") or "").strip()
    created_at = None
    if created:
        created_at = datetime.fromtimestamp(float(created), tz=timezone.utc).isoformat()
    return {
        "id": str(pid).replace("t3_", ""),
        "name": data.get("name"),
        "title": title,
        "selftext": text_body,
        "text": f"{title}\n{text_body}".strip() if title else text_body,
        "author": data.get("author"),
        "subreddit": data.get("subreddit") or subreddit,
        "score": data.get("score"),
        "num_comments": data.get("num_comments"),
        "upvote_ratio": data.get("upvote_ratio"),
        "permalink": _permalink(data.get("permalink")),
        "url": data.get("url"),
        "created_utc": created,
        "created_at": created_at,
        "link_flair_text": data.get("link_flair_text"),
        "_content_type": "post",
        "_collection_method": "http_lib",
        "_query": query,
        "_collected_at": utc_now().isoformat(),
    }


def _comment_rows(post_id: str, subreddit: str, since: datetime) -> list[dict[str, Any]]:
    """Fetch comment tree for a submission; flatten top-level + replies lightly."""
    out: list[dict[str, Any]] = []
    try:
        payload = _get_json_multi(f"/comments/{post_id}.json", {"limit": 100, "raw_json": 1})
    except Exception:
        return out
    if not isinstance(payload, list) or len(payload) < 2:
        return out

    def walk(node: Any, parent_id: str | None = None) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("kind")
        data = node.get("data") or {}
        if kind == "Listing":
            for ch in data.get("children") or []:
                walk(ch, parent_id)
            return
        if kind == "more":
            return
        if kind != "t1":
            return
        cid = data.get("id")
        body = (data.get("body") or "").strip()
        created = data.get("created_utc")
        created_dt = parse_iso(created)
        if created_dt and created_dt < since:
            replies = data.get("replies")
            if isinstance(replies, dict):
                walk(replies, parent_id=str(cid) if cid else parent_id)
            return
        if cid and body and body not in ("[deleted]", "[removed]"):
            out.append(
                {
                    "id": str(cid),
                    "parent_id": parent_id or post_id,
                    "link_id": data.get("link_id") or f"t3_{post_id}",
                    "body": body,
                    "text": body,
                    "author": data.get("author"),
                    "subreddit": data.get("subreddit") or subreddit,
                    "score": data.get("score"),
                    "permalink": _permalink(data.get("permalink")),
                    "created_utc": created,
                    "created_at": created_dt.isoformat() if created_dt else None,
                    "_content_type": "comment",
                    "_thread_id": post_id,
                    "_parent_id": parent_id or post_id,
                    "_collection_method": "http_lib",
                    "_collected_at": utc_now().isoformat(),
                }
            )
        replies = data.get("replies")
        if isinstance(replies, dict):
            walk(replies, parent_id=str(cid) if cid else parent_id)

    walk(payload[1], parent_id=post_id)
    return out


def _search_subreddit(subreddit: str, query: str, since: datetime) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    after = None
    pages = 0
    while pages < 8:
        params: dict[str, Any] = {
            "q": query,
            "restrict_sr": "1",
            "sort": "new",
            "t": "year",
            "limit": 100,
            "raw_json": 1,
            "type": "link",
        }
        if after:
            params["after"] = after
        data = _get_json_multi(f"/r/{subreddit}/search.json", params)
        listing = (data or {}).get("data") or {}
        children = listing.get("children") or []
        if not children:
            break
        reached_cutoff = False
        for ch in children:
            row = _submission_row(ch, subreddit, query)
            if not row:
                continue
            created = parse_iso(row.get("created_at"))
            if created and created < since:
                reached_cutoff = True
                continue
            items.append(row)
        after = listing.get("after")
        pages += 1
        time.sleep(1.5)
        if not after or reached_cutoff:
            break
    return items


def _pullpush_search(subreddit: str, query: str, since: datetime) -> list[dict[str, Any]]:
    """Fallback free search via PullPush (Pushshift-compatible public archive)."""
    items: list[dict[str, Any]] = []
    after_ts = int(since.timestamp())
    payload = None
    for attempt in range(2):
        try:
            resp = requests.get(
                "https://api.pullpush.io/reddit/search/submission",
                params={
                    "q": query,
                    "subreddit": subreddit,
                    "after": after_ts,
                    "size": 100,
                    "sort": "desc",
                    "sort_type": "created_utc",
                },
                timeout=45,
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"[reddit] pullpush rate-limited; sleeping {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:
            print(f"[reddit] pullpush failed r/{subreddit} {query!r}: {exc}")
            return items
    if payload is None:
        print(f"[reddit] pullpush unavailable for r/{subreddit} {query!r}; will try browser")
        return items

    for data_row in payload.get("data") or []:
        created = data_row.get("created_utc")
        created_at = None
        if created:
            created_at = datetime.fromtimestamp(float(created), tz=timezone.utc).isoformat()
        title = (data_row.get("title") or "").strip()
        body = (data_row.get("selftext") or "").strip()
        pid = str(data_row.get("id") or "")
        if not pid:
            continue
        created_dt = parse_iso(created_at)
        if created_dt and created_dt < since:
            continue
        items.append(
            {
                "id": pid,
                "title": title,
                "selftext": body,
                "text": f"{title}\n{body}".strip() if title else body,
                "author": data_row.get("author"),
                "subreddit": data_row.get("subreddit") or subreddit,
                "score": data_row.get("score"),
                "num_comments": data_row.get("num_comments"),
                "permalink": _permalink(data_row.get("permalink")),
                "url": data_row.get("url"),
                "created_utc": created,
                "created_at": created_at,
                "_content_type": "post",
                "_collection_method": "http_lib",
                "_query": query,
                "_via": "pullpush",
                "_collected_at": utc_now().isoformat(),
            }
        )
    return items


def _browser_search(subreddit: str, query: str, since: datetime) -> list[dict[str, Any]]:
    """Scrape old.reddit.com search HTML when JSON APIs are blocked."""
    from urllib.parse import quote_plus

    from playwright.sync_api import sync_playwright

    items: list[dict[str, Any]] = []
    url = (
        f"https://old.reddit.com/r/{subreddit}/search"
        f"?q={quote_plus(query)}&restrict_sr=on&sort=new&t=year"
    )
    # Headless is more reliable for Reddit public search (no login needed).
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=USER_AGENT,
                viewport={"width": 1280, "height": 900},
            )
            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            time.sleep(3)
            for _ in range(3):
                results = page.query_selector_all("div.search-result, div.search-result-link")
                for el in results:
                    try:
                        title_el = el.query_selector("a.search-title")
                        title = (title_el.inner_text() if title_el else "").strip()
                        href = title_el.get_attribute("href") if title_el else None
                        author_el = el.query_selector("a.author")
                        author = author_el.inner_text().strip() if author_el else None
                        time_el = el.query_selector("time")
                        created_at = time_el.get_attribute("datetime") if time_el else None
                        exp = el.query_selector("div.search-result-body")
                        body = (exp.inner_text() if exp else "").strip()
                        score_el = el.query_selector("span.search-score")
                        score_txt = (score_el.inner_text() if score_el else "").strip()
                        score = None
                        if score_txt:
                            digits = "".join(ch for ch in score_txt if ch.isdigit() or ch == "-")
                            score = int(digits) if digits else None
                        pid = None
                        if href and "/comments/" in href:
                            parts = href.strip("/").split("/")
                            if "comments" in parts:
                                i = parts.index("comments")
                                if i + 1 < len(parts):
                                    pid = parts[i + 1]
                        if not pid:
                            pid = str(abs(hash((title, created_at or "", query))))
                        created_dt = parse_iso(created_at)
                        if created_dt and created_dt < since:
                            continue
                        text = f"{title}\n{body}".strip() if title and body else (title or body)
                        if not text:
                            continue
                        if href and not href.startswith("http"):
                            href = "https://old.reddit.com" + href
                        items.append(
                            {
                                "id": pid,
                                "title": title,
                                "selftext": body,
                                "text": text,
                                "author": author,
                                "subreddit": subreddit,
                                "score": score,
                                "permalink": href,
                                "created_at": created_dt.isoformat() if created_dt else created_at,
                                "_content_type": "post",
                                "_collection_method": "browser",
                                "_query": query,
                                "_via": "old_reddit_browser",
                                "_collected_at": utc_now().isoformat(),
                            }
                        )
                    except Exception:
                        continue
                nxt = page.query_selector("span.nextprev a[rel='nofollow next']")
                if not nxt:
                    break
                nxt.click()
                time.sleep(2.5)
            browser.close()
    except Exception as exc:
        print(f"[reddit] browser search failed r/{subreddit} {query!r}: {exc}")
        return items
    seen = set()
    uniq = []
    for row in items:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        uniq.append(row)
    print(f"[reddit] browser raw={len(items)} uniq={len(uniq)} for r/{subreddit} {query!r}")
    return uniq


def collect_reddit(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    config: dict[str, Any] | None = None,
) -> CollectorResult:
    run_id = new_run_id()
    result = CollectorResult(source="reddit", brand=brand, run_id=run_id)
    cfg = config or {}
    reddit_global = cfg.get("reddit") or {}
    brand_reddit = brand_cfg.get("reddit") or {}

    subreddits = list(brand_reddit.get("subreddits") or reddit_global.get("subreddits") or ["Dhaka", "bangladesh"])
    queries = list(brand_reddit.get("queries") or [])
    if not queries:
        display = brand_cfg.get("display_name") or brand
        queries = [display]

    fetch_comments = bool(reddit_global.get("fetch_comments", True))
    seen_ids: set[str] = set()
    items: list[dict[str, Any]] = []

    try:
        for sub in subreddits:
            for query in queries:
                print(f"[reddit] {brand}: r/{sub} q={query!r}")
                # Browser-first: reddit.com JSON is blocked; PullPush is often rate-limited.
                found = _browser_search(sub, query, since)
                if not found:
                    found = _pullpush_search(sub, query, since)
                    if found:
                        print(f"[reddit] pullpush: {len(found)} posts for r/{sub} {query!r}")
                for row in found:
                    rid = row["id"]
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    items.append(row)
                    if fetch_comments and not row.get("_via"):
                        time.sleep(1.0)
                        try:
                            comments = _comment_rows(rid, sub, since)
                        except Exception:
                            comments = []
                        for c in comments:
                            cid = c["id"]
                            if cid in seen_ids:
                                continue
                            seen_ids.add(cid)
                            items.append(c)
                time.sleep(3.0)

        result.item_count = write_raw_items("reddit", brand, run_id, items)
        result.meta = {
            "subreddits": subreddits,
            "queries": queries,
            "posts": sum(1 for i in items if i.get("_content_type") == "post"),
            "comments": sum(1 for i in items if i.get("_content_type") == "comment"),
        }
        save_checkpoint(
            "reddit",
            brand,
            {
                "complete": True,
                "last_run_id": run_id,
                "last_count": result.item_count,
                "updated_at": utc_now().isoformat(),
            },
        )
        print(
            f"[reddit] {brand}: {result.item_count} items "
            f"({result.meta['posts']} posts, {result.meta['comments']} comments)"
        )
    except Exception as exc:
        result.error_summary = format_exc(exc)
        result.status = "error"
        if items:
            result.item_count = write_raw_items("reddit", brand, run_id, items)
            result.status = "partial"

    log_run(result.finish())
    return result
