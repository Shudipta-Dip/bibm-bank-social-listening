"""Facebook collector: Graph API first, Playwright fallback."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from listening.collectors import CollectorResult, format_exc, log_run, new_run_id, write_raw_items
from listening.hitl import (
    GATE_H2_TARGET,
    GATE_H3_BLOCKER,
    GATE_H4_COMPLETENESS,
    HitlBlockedError,
    gate_resolution,
    raise_gate,
)
from listening.utils import (
    ROOT,
    env_flag,
    load_checkpoint,
    parse_iso,
    save_checkpoint,
    utc_now,
)

GRAPH_BASE = "https://graph.facebook.com"


def _graph_version() -> str:
    return os.getenv("META_GRAPH_VERSION", "v21.0")


def _get_token(brand_cfg: dict[str, Any]) -> Optional[str]:
    env_name = (brand_cfg.get("facebook") or {}).get("token_env")
    if not env_name:
        return None
    tok = os.getenv(env_name, "").strip()
    return tok or None


def _graph_get(path: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    params["access_token"] = token
    url = f"{GRAPH_BASE}/{_graph_version()}/{path.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=60)
    data = resp.json()
    if resp.status_code >= 400 or "error" in data:
        err = data.get("error") or {}
        raise RuntimeError(f"Graph API error: {err.get('message') or data}")
    return data


def _resolve_page_id(brand_cfg: dict[str, Any], token: str) -> str:
    fb = brand_cfg.get("facebook") or {}
    if fb.get("page_id"):
        return str(fb["page_id"])
    page_url = fb.get("page_url") or ""
    slug = page_url.rstrip("/").split("/")[-1]
    if not slug:
        raise_gate(
            GATE_H2_TARGET,
            "Facebook page_id/page_url missing or ambiguous",
            source="facebook",
            payload={"facebook": fb},
        )
        raise RuntimeError("unreachable")
    data = _graph_get(slug, token, {"fields": "id,name,link"})
    return str(data["id"])


def _collect_via_graph(
    brand: str,
    brand_cfg: dict[str, Any],
    token: str,
    since: datetime,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    hitl_flags: list[str] = []
    page_id = _resolve_page_id(brand_cfg, token)
    items: list[dict[str, Any]] = []
    ckpt = load_checkpoint("facebook", brand)
    after = ckpt.get("graph_after") if ckpt.get("method") == "api" and not ckpt.get("complete") else None

    fields = (
        "id,message,created_time,permalink_url,shares,from,"
        "reactions.summary(true),comments.summary(true)"
    )
    params: dict[str, Any] = {
        "fields": fields,
        "limit": 50,
        "since": int(since.timestamp()),
    }
    if after:
        params["after"] = after

    while True:
        data = _graph_get(f"{page_id}/posts", token, params)
        posts = data.get("data") or []
        if not posts:
            break
        for post in posts:
            created = parse_iso(post.get("created_time"))
            if created and created < since:
                continue
            row = dict(post)
            row["_content_type"] = "post"
            row["_thread_id"] = post.get("id")
            row["_collection_method"] = "api"
            row["_is_page"] = True
            row["_collected_at"] = utc_now().isoformat()
            items.append(row)

            deep = gate_resolution(GATE_H4_COMPLETENESS, brand=brand, source="facebook") != "accept_partial"
            c_after = None
            comment_pages = 0
            max_comment_pages = 20 if deep else 3
            while comment_pages < max_comment_pages:
                c_params: dict[str, Any] = {
                    "fields": "id,message,created_time,from,like_count,parent",
                    "limit": 50,
                    "filter": "toplevel" if not deep else "stream",
                }
                if c_after:
                    c_params["after"] = c_after
                try:
                    cdata = _graph_get(f"{post['id']}/comments", token, c_params)
                except Exception:
                    break
                comments = cdata.get("data") or []
                if not comments:
                    break
                for c in comments:
                    crow = dict(c)
                    crow["_content_type"] = "comment"
                    crow["_parent_id"] = post["id"]
                    crow["_thread_id"] = post["id"]
                    crow["_collection_method"] = "api"
                    crow["_collected_at"] = utc_now().isoformat()
                    items.append(crow)
                paging = (cdata.get("paging") or {}).get("cursors") or {}
                c_after = paging.get("after")
                comment_pages += 1
                if not c_after:
                    break
            if comment_pages >= max_comment_pages:
                hitl_flags.append(GATE_H4_COMPLETENESS)

        paging = (data.get("paging") or {}).get("cursors") or {}
        after = paging.get("after")
        save_checkpoint(
            "facebook",
            brand,
            {
                "method": "api",
                "graph_after": after,
                "page_id": page_id,
                "complete": False,
                "updated_at": utc_now().isoformat(),
            },
        )
        if not after:
            break
        params = {
            "fields": fields,
            "limit": 50,
            "since": int(since.timestamp()),
            "after": after,
        }

    if GATE_H4_COMPLETENESS in hitl_flags and gate_resolution(GATE_H4_COMPLETENESS, brand=brand, source="facebook") is None:
        try:
            raise_gate(
                GATE_H4_COMPLETENESS,
                "Facebook comment threads may be truncated. Resolve with resume (deeper) or accept_partial.",
                brand=brand,
                source="facebook",
                blocking=False,
            )
        except HitlBlockedError:
            pass

    save_checkpoint(
        "facebook",
        brand,
        {
            "method": "api",
            "graph_after": None,
            "page_id": page_id,
            "complete": True,
            "last_run_id": run_id,
            "updated_at": utc_now().isoformat(),
        },
    )
    return items, hitl_flags


def _fb_dismiss_overlays(page) -> None:
    try:
        page.keyboard.press("Escape")
        time.sleep(0.4)
        page.keyboard.press("Escape")
    except Exception:
        pass
    for label in (
        "Allow all cookies",
        "Accept all",
        "Accept All",
        "Only allow essential cookies",
        "Decline optional cookies",
        "Close",
        "Not Now",
        "Not now",
    ):
        try:
            btn = page.get_by_role("button", name=label)
            if btn.count() and btn.first.is_visible(timeout=700):
                btn.first.click(timeout=1200)
                time.sleep(0.8)
        except Exception:
            pass


def _fb_needs_login(page) -> bool:
    url = page.url.lower()
    if any(x in url for x in ("/login.php", "/login/", "checkpoint/", "recover/")):
        return True
    try:
        if page.query_selector_all('[role="article"]'):
            return False
    except Exception:
        pass
    try:
        body = page.inner_text("body", timeout=3000)[:3000].lower()
    except Exception:
        body = ""
    return "you must log in to continue" in body or "this content isn't available right now" in body


def _fb_is_logged_in(page) -> bool:
    """True when session looks authenticated (no top login form / soft banner)."""
    url = page.url.lower()
    if any(x in url for x in ("/login.php", "/login/", "checkpoint/", "recover/")):
        return False
    try:
        # Logged-out pages show email+password fields in the header
        if page.locator('input[name="email"]').count() and page.locator('input[name="pass"]').count():
            if page.locator('input[name="email"]').first.is_visible(timeout=800):
                return False
    except Exception:
        pass
    try:
        body = page.inner_text("body", timeout=3000)[:2000].lower()
    except Exception:
        body = ""
    if "log in or sign up for facebook" in body:
        return False
    return True


def _fb_ensure_interactive_login(page, storage_path: Path, minutes: int = 10) -> bool:
    """Open Facebook login and wait until the user completes it."""
    if _fb_is_logged_in(page):
        # Confirm on home
        try:
            page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=90000)
            time.sleep(2)
            _fb_dismiss_overlays(page)
        except Exception:
            pass
        if _fb_is_logged_in(page):
            print("[HITL] Facebook session already logged in.")
            return True

    print(
        f"[HITL] Chromium is open on Facebook login. "
        f"Please log in now (2FA/captcha OK). Waiting up to {minutes} minutes..."
    )
    try:
        page.goto("https://www.facebook.com/login", wait_until="domcontentloaded", timeout=120000)
    except Exception:
        page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=120000)
    time.sleep(2)

    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        time.sleep(5)
        _fb_dismiss_overlays(page)
        if _fb_is_logged_in(page):
            try:
                page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=90000)
                time.sleep(2)
            except Exception:
                pass
            if _fb_is_logged_in(page):
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                print("[HITL] Facebook login detected. Continuing scrape...")
                return True
    return False


def _fb_wait_for_login(page, page_url: str, brand: str, minutes: int = 5) -> bool:
    if not _fb_needs_login(page):
        return True
    print(f"[HITL] Facebook hard-block for {brand}. Log in in Chromium (up to {minutes} min)...")
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        time.sleep(5)
        if not _fb_needs_login(page) and _fb_is_logged_in(page):
            try:
                page.goto(page_url, wait_until="domcontentloaded", timeout=90000)
                time.sleep(3)
            except Exception:
                pass
            return True
    return False


_FB_EXTRACT_JS = r"""
() => {
  const out = [];
  const push = (t) => {
    t = (t || '').trim();
    if (t.length >= 40) out.push(t.slice(0, 8000));
  };
  document.querySelectorAll('[role="article"]').forEach(el => push(el.innerText));
  document.querySelectorAll('div[data-ad-preview="message"], div[data-ad-comet-preview="message"]').forEach(el => push(el.innerText));
  document.querySelectorAll('div[dir="auto"]').forEach(el => {
    const t = (el.innerText || '').trim();
    if (t.length >= 80 && t.length < 4000) out.push(t);
  });
  return out;
}
"""


def _fb_extract_posts(page, brand: str, page_url: str) -> list[dict[str, Any]]:
    try:
        blobs = page.evaluate(_FB_EXTRACT_JS)
    except Exception:
        blobs = []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    skip = (
        "log in",
        "log into facebook",
        "create new account",
        "email or phone",
        "forgot account",
    )
    for text in blobs or []:
        text = (text or "").strip()
        if len(text) < 40:
            continue
        low = text.lower()
        if any(low.startswith(p) for p in skip) or "log in or sign up for facebook" in low:
            continue
        key = text[:280]
        if key in seen:
            continue
        seen.add(key)
        native_id = f"fb_browser_{brand}_{abs(hash(key))}"
        items.append(
            {
                "id": native_id,
                "message": text[:8000],
                "created_time": None,
                "permalink_url": page_url,
                "_content_type": "post",
                "_thread_id": native_id,
                "_collection_method": "browser",
                "_is_page": True,
                "_collected_at": utc_now().isoformat(),
                "_since_filter_unverified": True,
            }
        )
    return items


def _collect_via_playwright(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    from playwright.sync_api import sync_playwright

    fb = brand_cfg.get("facebook") or {}
    page_url = fb.get("page_url")
    if not page_url:
        raise_gate(
            GATE_H2_TARGET,
            f"Facebook page_url required ({brand})",
            brand=brand,
            source="facebook",
            payload={"facebook": fb},
        )

    storage_path = Path(os.getenv("FACEBOOK_STORAGE_STATE", "browser_profiles/facebook_storage.json"))
    if not storage_path.is_absolute():
        storage_path = ROOT / storage_path

    headed = env_flag("BROWSER_HEADED", True)
    require_login = env_flag("FACEBOOK_REQUIRE_LOGIN", False)
    scroll_rounds = 40 if require_login else 20
    items: list[dict[str, Any]] = []
    hitl_flags: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx_kwargs: dict[str, Any] = {
            "viewport": {"width": 1365, "height": 900},
            "locale": "en-US",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        if storage_path.exists():
            ctx_kwargs["storage_state"] = str(storage_path)
        context = browser.new_context(**ctx_kwargs)
        page = context.new_page()

        if require_login:
            page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=120000)
            time.sleep(2)
            _fb_dismiss_overlays(page)
            if not _fb_ensure_interactive_login(page, storage_path, minutes=10):
                shot = ROOT / "data" / "hitl" / f"facebook_blocker_{brand}_{run_id}.png"
                shot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot))
                try:
                    context.storage_state(path=str(storage_path))
                except Exception:
                    pass
                browser.close()
                raise_gate(
                    GATE_H3_BLOCKER,
                    f"Facebook login timed out for {brand}. Log in in Chromium, resolve resume, re-run.",
                    brand=brand,
                    source="facebook",
                    payload={"screenshot": str(shot)},
                )
            try:
                context.storage_state(path=str(storage_path))
            except Exception:
                pass

        page.goto(page_url, wait_until="domcontentloaded", timeout=120000)
        time.sleep(4)
        _fb_dismiss_overlays(page)

        if (not require_login) and _fb_needs_login(page) and not _fb_extract_posts(page, brand, page_url):
            if not _fb_wait_for_login(page, page_url, brand, minutes=5):
                shot = ROOT / "data" / "hitl" / f"facebook_blocker_{brand}_{run_id}.png"
                shot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot))
                browser.close()
                raise_gate(
                    GATE_H3_BLOCKER,
                    f"Facebook login required for {brand}",
                    brand=brand,
                    source="facebook",
                    payload={"screenshot": str(shot)},
                )
            page.goto(page_url, wait_until="domcontentloaded", timeout=120000)
            time.sleep(3)
            _fb_dismiss_overlays(page)

        # Prefer posts tab when logged in
        try:
            page.goto(page_url.rstrip("/") + "/posts", wait_until="domcontentloaded", timeout=90000)
            time.sleep(3)
            _fb_dismiss_overlays(page)
        except Exception:
            pass

        merged: dict[str, dict[str, Any]] = {}
        for _ in range(scroll_rounds):
            for row in _fb_extract_posts(page, brand, page_url):
                merged[row["id"]] = row
            page.mouse.wheel(0, 4500)
            time.sleep(random.uniform(1.2, 2.8))
            _fb_dismiss_overlays(page)

        items = list(merged.values())
        print(f"[facebook] {brand}: captured {len(items)} text items (login={require_login})")

        if not items:
            shot = ROOT / "data" / "hitl" / f"facebook_empty_{brand}_{run_id}.png"
            shot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shot))
            hitl_flags.append(GATE_H3_BLOCKER)
            try:
                raise_gate(
                    GATE_H3_BLOCKER,
                    f"Facebook returned 0 posts for {brand}. See {shot}",
                    brand=brand,
                    source="facebook",
                    blocking=False,
                    payload={"screenshot": str(shot)},
                )
            except HitlBlockedError:
                pass

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            context.storage_state(path=str(storage_path))
        except Exception:
            pass
        browser.close()

    save_checkpoint(
        "facebook",
        brand,
        {
            "method": "browser",
            "complete": True,
            "last_run_id": run_id,
            "count": len(items),
            "updated_at": utc_now().isoformat(),
        },
    )
    return items, hitl_flags


def collect_facebook(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    force_browser: bool = False,
) -> CollectorResult:
    run_id = new_run_id()
    result = CollectorResult(source="facebook", brand=brand, run_id=run_id)
    token = _get_token(brand_cfg)
    browser_only = force_browser or env_flag("SOCIAL_BROWSER_ONLY", True) or not token

    try:
        if token and not browser_only:
            items, flags = _collect_via_graph(brand, brand_cfg, token, since, run_id)
            result.hitl_flags.extend(flags)
            result.meta["method"] = "api"
        else:
            items, flags = _collect_via_playwright(brand, brand_cfg, since, run_id)
            result.hitl_flags.extend(flags)
            result.meta["method"] = "browser"

        result.item_count = write_raw_items("facebook", brand, run_id, items)
    except HitlBlockedError as e:
        result.hitl_flags.append(e.gate_id)
        result.error_summary = str(e)
        result.status = "hitl_blocked"
    except Exception as exc:
        msg = str(exc)
        if token and not force_browser and "Graph API" in msg:
            try:
                items, flags = _collect_via_playwright(brand, brand_cfg, since, run_id)
                result.hitl_flags.extend(flags)
                result.meta["method"] = "browser"
                result.meta["graph_error"] = msg
                result.item_count = write_raw_items("facebook", brand, run_id, items)
                result.status = "partial"
            except HitlBlockedError as e:
                result.hitl_flags.append(e.gate_id)
                result.error_summary = str(e)
                result.status = "hitl_blocked"
            except Exception as exc2:
                result.error_summary = format_exc(exc2)
                result.status = "error"
        else:
            result.error_summary = format_exc(exc)
            result.status = "error"

    log_run(result.finish())
    return result
