"""Google Play Store review collector (free http library)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from listening.collectors import CollectorResult, format_exc, log_run, new_run_id, write_raw_items
from listening.hitl import GATE_H2_TARGET, HitlBlockedError, raise_gate
from listening.utils import load_checkpoint, parse_iso, save_checkpoint, utc_now


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def _fetch_batch(app_id: str, lang: str, country: str, count: int, continuation_token: Any):
    from google_play_scraper import Sort, reviews

    result, token = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
        continuation_token=continuation_token,
    )
    return result, token


def collect_google_play(
    brand: str,
    brand_cfg: dict[str, Any],
    since: datetime,
    batch_size: int = 200,
    max_retries: int = 3,
) -> CollectorResult:
    run_id = new_run_id()
    result = CollectorResult(source="google_play", brand=brand, run_id=run_id)
    gp = brand_cfg.get("google_play") or {}
    app_id = gp.get("app_id")
    if not app_id:
        try:
            raise_gate(
                GATE_H2_TARGET,
                f"Missing Google Play app_id for {brand}",
                brand=brand,
                source="google_play",
                payload={"field": "google_play.app_id"},
            )
        except HitlBlockedError as e:
            result.hitl_flags.append(e.gate_id)
            result.error_summary = str(e)
            log_run(result.finish("hitl_blocked"))
            return result

    lang = gp.get("lang") or "en"
    country = gp.get("country") or "bd"
    ckpt = load_checkpoint("google_play", brand)
    continuation = None
    # only reuse token if same app and incomplete prior run
    if ckpt.get("app_id") == app_id and ckpt.get("continuation_token") is not None and not ckpt.get("complete"):
        continuation = ckpt.get("continuation_token")

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set(ckpt.get("seen_ids") or [])
    reached_cutoff = False
    errors = 0

    try:
        while True:
            try:
                batch, continuation = _fetch_batch(app_id, lang, country, batch_size, continuation)
            except Exception as exc:
                errors += 1
                if errors >= max_retries:
                    # HITL-ish alert: scraper may have broken — record and stop
                    result.error_summary = format_exc(exc)
                    result.status = "error"
                    break
                continue

            if not batch:
                break

            for rev in batch:
                rid = str(rev.get("reviewId") or "")
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                created = parse_iso(rev.get("at"))
                if created and created < since:
                    reached_cutoff = True
                    continue
                row = dict(rev)
                # serialize datetime
                if row.get("at") is not None:
                    row["at"] = parse_iso(row["at"]).isoformat() if parse_iso(row["at"]) else str(row["at"])
                if row.get("repliedAt") is not None:
                    ra = parse_iso(row["repliedAt"])
                    row["repliedAt"] = ra.isoformat() if ra else str(row["repliedAt"])
                row["_collection_method"] = "http_lib"
                row["_collected_at"] = utc_now().isoformat()
                items.append(row)

            save_checkpoint(
                "google_play",
                brand,
                {
                    "app_id": app_id,
                    "continuation_token": continuation,
                    "seen_ids": list(seen_ids)[-5000:],
                    "complete": False,
                    "updated_at": utc_now().isoformat(),
                },
            )

            if reached_cutoff or continuation is None:
                break

        result.item_count = write_raw_items("google_play", brand, run_id, items)
        save_checkpoint(
            "google_play",
            brand,
            {
                "app_id": app_id,
                "continuation_token": None,
                "seen_ids": list(seen_ids)[-5000:],
                "complete": True,
                "updated_at": utc_now().isoformat(),
                "last_run_id": run_id,
                "last_count": result.item_count,
            },
        )
        if result.status == "ok" and result.error_summary:
            result.status = "partial"
        elif result.status == "ok":
            result.status = "ok"
    except HitlBlockedError as e:
        result.hitl_flags.append(e.gate_id)
        result.error_summary = str(e)
        result.status = "hitl_blocked"
    except Exception as exc:
        result.error_summary = format_exc(exc)
        result.status = "error"
        if items:
            result.item_count = write_raw_items("google_play", brand, run_id, items)
            result.status = "partial"

    log_run(result.finish())
    return result
