"""Apple App Store review collector (RSS / app-store-scraper + coverage notes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from listening.collectors import CollectorResult, format_exc, log_run, new_run_id, write_raw_items
from listening.hitl import GATE_COVERAGE_SIGN_OFF, GATE_H2_TARGET, HitlBlockedError, raise_gate
from listening.utils import load_checkpoint, parse_iso, save_checkpoint, utc_now


ITUNES_RSS = (
    "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def _fetch_rss_page(app_id: str, country: str, page: int) -> dict[str, Any]:
    url = ITUNES_RSS.format(country=country, page=page, app_id=app_id)
    resp = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/javascript,*/*",
        },
    )
    resp.raise_for_status()
    # Some storefronts return empty body intermittently
    if not resp.text or not resp.text.strip():
        return {"feed": {"entry": []}}
    return resp.json()


def _parse_rss_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    feed = payload.get("feed") or {}
    entries = feed.get("entry") or []
    if not isinstance(entries, list):
        entries = [entries]
    reviews = []
    for entry in entries:
        # first entry is often the app metadata
        if "im:rating" not in entry and "im:name" in entry and "content" not in entry:
            continue
        if "im:rating" not in entry:
            continue
        review_id = (entry.get("id") or {}).get("label")
        author = ((entry.get("author") or {}).get("name") or {}).get("label")
        title = (entry.get("title") or {}).get("label")
        content = (entry.get("content") or {}).get("label")
        rating = (entry.get("im:rating") or {}).get("label")
        updated = (entry.get("updated") or {}).get("label")
        version = (entry.get("im:version") or {}).get("label")
        vote_count = (entry.get("im:voteCount") or {}).get("label")
        link = None
        links = entry.get("link")
        if isinstance(links, dict):
            link = (links.get("attributes") or {}).get("href")
        reviews.append(
            {
                "id": str(review_id) if review_id else None,
                "userName": author,
                "title": title,
                "review": content,
                "rating": int(rating) if rating else None,
                "date": updated,
                "version": version,
                "voteCount": int(vote_count) if vote_count and str(vote_count).isdigit() else None,
                "url": link,
            }
        )
    return reviews


def _try_app_store_scraper(app_id: str, country: str) -> list[dict[str, Any]]:
    try:
        from app_store_scraper import AppStore
    except ImportError:
        return []
    try:
        app = AppStore(country=country, app_name="app", app_id=int(app_id))
        app.review(how_many=500)
        out = []
        for r in app.reviews or []:
            row = dict(r)
            if "date" in row and row["date"] is not None:
                dt = parse_iso(row["date"])
                row["date"] = dt.isoformat() if dt else str(row["date"])
            if "id" not in row and "userName" in row:
                # synthesize id
                row["id"] = f"{row.get('userName')}_{row.get('date')}_{row.get('rating')}"
            out.append(row)
        return out
    except Exception:
        return []


def collect_app_store(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    min_coverage_months: int = 6,
) -> CollectorResult:
    run_id = new_run_id()
    result = CollectorResult(source="app_store", brand=brand, run_id=run_id)
    as_cfg = brand_cfg.get("app_store") or {}
    app_id = str(as_cfg.get("app_id") or "")
    if not app_id:
        try:
            raise_gate(
                GATE_H2_TARGET,
                f"Missing App Store app_id for {brand}",
                brand=brand,
                source="app_store",
                payload={"field": "app_store.app_id"},
            )
        except HitlBlockedError as e:
            result.hitl_flags.append(e.gate_id)
            result.error_summary = str(e)
            log_run(result.finish("hitl_blocked"))
            return result

    countries = [as_cfg.get("country") or "bd"] + list(as_cfg.get("fallback_countries") or [])
    # dedupe preserve order
    seen_c = set()
    countries = [c for c in countries if not (c in seen_c or seen_c.add(c))]

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dates: list[datetime] = []

    try:
        for country in countries:
            # RSS pages 1..10
            for page in range(1, 11):
                try:
                    payload = _fetch_rss_page(app_id, country, page)
                    batch = _parse_rss_entries(payload)
                except Exception:
                    break
                if not batch:
                    break
                for rev in batch:
                    rid = str(rev.get("id") or "")
                    if not rid or rid in seen_ids:
                        continue
                    created = parse_iso(rev.get("date"))
                    if created and created < since:
                        continue
                    seen_ids.add(rid)
                    if created:
                        dates.append(created)
                    row = dict(rev)
                    row["_country"] = country
                    row["_collection_method"] = "http_lib"
                    row["_collected_at"] = utc_now().isoformat()
                    items.append(row)

            # supplement with app-store-scraper
            extra = _try_app_store_scraper(app_id, country)
            for rev in extra:
                rid = str(rev.get("id") or "")
                if not rid or rid in seen_ids:
                    continue
                created = parse_iso(rev.get("date"))
                if created and created < since:
                    continue
                seen_ids.add(rid)
                if created:
                    dates.append(created)
                row = dict(rev)
                row["id"] = rid
                row["_country"] = country
                row["_collection_method"] = "http_lib"
                row["_collected_at"] = utc_now().isoformat()
                items.append(row)

        coverage_months = None
        coverage_note = None
        if dates:
            span_days = (max(dates) - min(dates)).days
            coverage_months = round(span_days / 30.44, 1)
            if coverage_months < min_coverage_months:
                coverage_note = (
                    f"App Store public feed coverage ~{coverage_months} months "
                    f"(<{min_coverage_months}). Accept partial or request App Store Connect export."
                )
                for row in items:
                    row["_coverage_note"] = coverage_note
                # non-blocking QA gate
                try:
                    raise_gate(
                        GATE_COVERAGE_SIGN_OFF,
                        coverage_note,
                        brand=brand,
                        source="app_store",
                        blocking=False,
                        payload={"coverage_months": coverage_months, "count": len(items)},
                    )
                except HitlBlockedError:
                    pass  # non-blocking should not raise; guard anyway
            else:
                # Clear stale coverage gates from earlier thin runs
                from listening.hitl import resolve_gate

                resolve_gate(
                    GATE_COVERAGE_SIGN_OFF,
                    "resume",
                    note=f"Auto-cleared: coverage now {coverage_months} months / {len(items)} reviews",
                    brand=brand,
                    source="app_store",
                )
        else:
            coverage_note = "No App Store reviews retrieved from public endpoints."
            try:
                raise_gate(
                    GATE_COVERAGE_SIGN_OFF,
                    coverage_note,
                    brand=brand,
                    source="app_store",
                    blocking=False,
                    payload={"coverage_months": 0, "count": 0},
                )
            except HitlBlockedError:
                pass

        result.item_count = write_raw_items("app_store", brand, run_id, items)
        result.meta = {
            "coverage_months": coverage_months,
            "coverage_note": coverage_note,
            "countries": countries,
        }
        save_checkpoint(
            "app_store",
            brand,
            {
                "app_id": app_id,
                "complete": True,
                "last_run_id": run_id,
                "last_count": result.item_count,
                "coverage_months": coverage_months,
                "updated_at": utc_now().isoformat(),
            },
        )
    except HitlBlockedError as e:
        result.hitl_flags.append(e.gate_id)
        result.error_summary = str(e)
        result.status = "hitl_blocked"
    except Exception as exc:
        result.error_summary = format_exc(exc)
        result.status = "error"
        if items:
            result.item_count = write_raw_items("app_store", brand, run_id, items)
            result.status = "partial"

    log_run(result.finish())
    return result
