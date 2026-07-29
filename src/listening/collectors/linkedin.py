"""LinkedIn company-page collector via Playwright (free browser automation)."""

from __future__ import annotations

import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from listening.collectors import CollectorResult, format_exc, log_run, new_run_id, write_raw_items
from listening.hitl import (
    GATE_H2_TARGET,
    GATE_H5_LOGIN,
    GATE_H6_SOFT_BAN,
    GATE_H7_COVERAGE,
    HitlBlockedError,
    gate_resolution,
    raise_gate,
)
from listening.utils import ROOT, env_flag, load_checkpoint, save_checkpoint, utc_now


def _storage_path() -> Path:
    storage = os.getenv("LINKEDIN_STORAGE_STATE", "browser_profiles/linkedin_storage.json")
    path = Path(storage)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _maybe_login(page, context, storage_path: Path) -> None:
    email = os.getenv("LINKEDIN_EMAIL", "").strip()
    password = os.getenv("LINKEDIN_PASSWORD", "").strip()

    def _needs_auth() -> bool:
        u = page.url.lower()
        if any(x in u for x in ("/login", "authwall", "uas/login", "checkpoint", "challenge")):
            return True
        try:
            body = page.inner_text("body", timeout=3000)[:2000].lower()
        except Exception:
            body = ""
        return "sign in" in body and ("join now" in body or "agree & join" in body or "email or phone" in body)

    if not _needs_auth():
        return

    if email and password:
        page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded", timeout=90000)
        time.sleep(2)
        page.fill("#username", email)
        page.fill("#password", password)
        page.click('button[type="submit"]')
        time.sleep(5)

    print(
        "[HITL] LinkedIn login window is open. "
        "Please log in / complete 2FA in Chromium NOW. Waiting up to 8 minutes..."
    )
    deadline = time.time() + 480
    while time.time() < deadline:
        time.sleep(5)
        if not _needs_auth():
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                context.storage_state(path=str(storage_path))
            except Exception:
                pass
            print("[HITL] LinkedIn login detected as complete.")
            return
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        context.storage_state(path=str(storage_path))
    except Exception:
        pass
    shot = ROOT / "data" / "hitl" / f"linkedin_login_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(shot))
    raise_gate(
        GATE_H5_LOGIN,
        "LinkedIn auth wall / 2FA timed out. Log in in the browser, then: "
        "python -m listening hitl resolve --gate H5_login --source linkedin --resolution resume "
        "and re-run collect.",
        source="linkedin",
        payload={"screenshot": str(shot), "storage_state": str(storage_path)},
    )


def collect_linkedin(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    delay_min_ms: int = 3000,
    delay_max_ms: int = 7000,
    min_posts_expected: int = 3,
) -> CollectorResult:
    run_id = new_run_id()
    result = CollectorResult(source="linkedin", brand=brand, run_id=run_id)
    li = brand_cfg.get("linkedin") or {}
    company_url = li.get("company_url")
    if not company_url:
        try:
            raise_gate(
                GATE_H2_TARGET,
                f"LinkedIn company_url missing for {brand}",
                brand=brand,
                source="linkedin",
                payload={"linkedin": li},
            )
        except HitlBlockedError as e:
            result.hitl_flags.append(e.gate_id)
            result.error_summary = str(e)
            log_run(result.finish("hitl_blocked"))
            return result

    posts_url = company_url.rstrip("/") + "/posts/?feedView=all"
    storage_path = _storage_path()
    headed = env_flag("BROWSER_HEADED", True)
    items: list[dict[str, Any]] = []
    blank_rounds = 0

    # soft-ban pause check (cleared when human resolves resume)
    h6 = gate_resolution(GATE_H6_SOFT_BAN, brand=brand, source="linkedin")
    ckpt = load_checkpoint("linkedin", brand)
    if h6 == "resume" and ckpt.get("soft_banned"):
        ckpt["soft_banned"] = False
        save_checkpoint("linkedin", brand, ckpt)
    elif ckpt.get("soft_banned") and h6 is None:
        try:
            raise_gate(
                GATE_H6_SOFT_BAN,
                f"LinkedIn soft-ban pause for {brand}. Wait 24–48h then resolve resume.",
                brand=brand,
                source="linkedin",
                payload=ckpt,
            )
        except HitlBlockedError as e:
            result.hitl_flags.append(e.gate_id)
            result.error_summary = str(e)
            log_run(result.finish("hitl_blocked"))
            return result

    try:
        from playwright.sync_api import sync_playwright

        profile_dir = ROOT / "browser_profiles" / "linkedin"
        profile_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=not headed,
                viewport={"width": 1365, "height": 900},
                locale="en-US",
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(posts_url, wait_until="domcontentloaded", timeout=120000)
            time.sleep(random.uniform(delay_min_ms / 1000, delay_max_ms / 1000))
            _maybe_login(page, context, storage_path)
            # re-navigate after login
            page.goto(posts_url, wait_until="domcontentloaded", timeout=120000)
            time.sleep(random.uniform(delay_min_ms / 1000, delay_max_ms / 1000))

            body_text = page.inner_text("body")[:1500].lower()
            if "restricted" in body_text or "unusual activity" in body_text:
                save_checkpoint(
                    "linkedin",
                    brand,
                    {"soft_banned": True, "updated_at": utc_now().isoformat()},
                )
                shot = ROOT / "data" / "hitl" / f"linkedin_ban_{brand}_{run_id}.png"
                page.screenshot(path=str(shot))
                context.close()
                raise_gate(
                    GATE_H6_SOFT_BAN,
                    f"LinkedIn restriction detected for {brand}. Pause 24–48h, then resolve resume.",
                    brand=brand,
                    source="linkedin",
                    payload={"screenshot": str(shot)},
                )

            seen: set[str] = set()
            for scroll_i in range(30):
                cards = page.query_selector_all(
                    "div.feed-shared-update-v2, div.update-components-text, article, "
                    "div.feed-shared-update-v2__description, div.fie-impression-container"
                )
                if not cards:
                    blank_rounds += 1
                    if blank_rounds >= 3:
                        break
                else:
                    blank_rounds = 0

                for idx, card in enumerate(cards):
                    try:
                        text = card.inner_text(timeout=1500).strip()
                    except Exception:
                        continue
                    if not text or len(text) < 20:
                        continue
                    # strip UI chrome somewhat
                    text = re.sub(r"\n{3,}", "\n\n", text)[:8000]
                    native_id = f"li_{brand}_{abs(hash(text[:300]))}"
                    if native_id in seen:
                        continue
                    seen.add(native_id)
                    items.append(
                        {
                            "id": native_id,
                            "commentary": text,
                            "created_at": None,
                            "url": posts_url,
                            "_content_type": "post",
                            "_collection_method": "browser",
                            "author_type": "page",
                            "_collected_at": utc_now().isoformat(),
                            "_since_filter_unverified": True,
                        }
                    )

                page.mouse.wheel(0, 3500)
                time.sleep(random.uniform(delay_min_ms / 1000, delay_max_ms / 1000))

                # checkpoint every few scrolls
                if scroll_i % 5 == 0:
                    save_checkpoint(
                        "linkedin",
                        brand,
                        {
                            "soft_banned": False,
                            "partial_count": len(items),
                            "scroll_i": scroll_i,
                            "updated_at": utc_now().isoformat(),
                        },
                    )

            storage_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                context.storage_state(path=str(storage_path))
            except Exception:
                pass
            context.close()

        if len(items) < min_posts_expected:
            try:
                raise_gate(
                    GATE_H7_COVERAGE,
                    f"LinkedIn yielded only {len(items)} posts for {brand}. Confirm company page URL.",
                    brand=brand,
                    source="linkedin",
                    blocking=False,
                    payload={"count": len(items), "company_url": company_url},
                )
            except HitlBlockedError:
                pass
            result.hitl_flags.append(GATE_H7_COVERAGE)

        result.item_count = write_raw_items("linkedin", brand, run_id, items)
        save_checkpoint(
            "linkedin",
            brand,
            {
                "soft_banned": False,
                "complete": True,
                "last_run_id": run_id,
                "last_count": result.item_count,
                "updated_at": utc_now().isoformat(),
            },
        )
        result.meta["method"] = "browser"
    except HitlBlockedError as e:
        result.hitl_flags.append(e.gate_id)
        result.error_summary = str(e)
        result.status = "hitl_blocked"
        if items:
            result.item_count = write_raw_items("linkedin", brand, run_id, items)
    except Exception as exc:
        result.error_summary = format_exc(exc)
        result.status = "error"
        if items:
            result.item_count = write_raw_items("linkedin", brand, run_id, items)
            result.status = "partial"

    log_run(result.finish())
    return result
