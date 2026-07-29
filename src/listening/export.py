"""Export unified dataset and summary report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from listening.hitl import coverage_summary
from listening.utils import DATA_PROCESSED, REPORTS, ensure_dirs, utc_now, write_json


def export_unified(records: list[dict[str, Any]], stem: str = "unified_mentions") -> dict[str, Path]:
    ensure_dirs()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    # serialize list/dict columns for CSV
    csv_df = df.copy()
    for col in ("engagement", "themes"):
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)

    parquet_path = DATA_PROCESSED / f"{stem}.parquet"
    csv_path = DATA_PROCESSED / f"{stem}.csv"
    jsonl_path = DATA_PROCESSED / f"{stem}.jsonl"

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # pyarrow optional failure
        parquet_path = None  # type: ignore

    csv_df.to_csv(csv_path, index=False, encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # brand/source slices
    slice_paths: dict[str, Path] = {}
    if "brand" in df.columns and "source" in df.columns:
        for (brand, source), g in df.groupby(["brand", "source"]):
            sp = DATA_PROCESSED / f"{stem}_{brand}_{source}.csv"
            gg = g.copy()
            for col in ("engagement", "themes"):
                if col in gg.columns:
                    gg[col] = gg[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)
            gg.to_csv(sp, index=False, encoding="utf-8")
            slice_paths[f"{brand}|{source}"] = sp

    out = {"csv": csv_path, "jsonl": jsonl_path, **{f"slice::{k}": v for k, v in slice_paths.items()}}
    if parquet_path:
        out["parquet"] = parquet_path
    write_json(DATA_PROCESSED / f"{stem}_meta.json", {"count": len(records), "exported_at": utc_now().isoformat()})
    return out


_KNOWN_THEMES = ["app_ux", "cards", "fees", "service", "transfers", "security"]

_PLATFORM_MAP = {
    ("app_store", "review"): "store_review",
    ("google_play", "review"): "store_review",
    ("google_play", "review_reply"): "store_review_reply",
    ("facebook", "post"): "group_post",
    ("facebook", "comment"): "group_comment",
    ("reddit", "post"): "subreddit_post",
    ("reddit", "comment"): "subreddit_comment",
}


def clean_for_analysis(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    """Produce a brand-manager-ready flat CSV from unified_mentions.csv."""
    import ast

    input_csv = input_csv or (DATA_PROCESSED / "unified_mentions.csv")
    output_csv = output_csv or (DATA_PROCESSED / "unified_mentions_clean.csv")

    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    raw_count = len(df)

    # ── Row filters ────────────────────────────────────────────────────────────
    df = df[df["in_scope"].astype(str).str.lower() != "false"]
    text_len = df["text"].fillna("").str.strip().str.len()
    # drop rows where text is shorter than 5 characters
    df = df[text_len >= 5]
    # deduplicate facebook comments with identical text (keep earliest created_at)
    fb_comment_mask = (df["source"] == "facebook") & (df["content_type"] == "comment")
    fb_comments = df[fb_comment_mask].copy()
    fb_comments["created_at"] = pd.to_datetime(fb_comments["created_at"], utc=True, errors="coerce")
    fb_comments = fb_comments.sort_values("created_at").drop_duplicates(subset=["source", "brand", "text"], keep="first")
    df = pd.concat([df[~fb_comment_mask], fb_comments], ignore_index=True)

    # ── Drop engineering columns ───────────────────────────────────────────────
    drop_cols = ["text_original", "raw_json", "coverage_note", "collection_method",
                 "native_id", "collected_at", "in_scope", "sentiment_source",
                 "rating_sentiment"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Flatten engagement JSON ───────────────────────────────────────────────
    def _parse_eng(val):
        if pd.isna(val) or val == "":
            return {}
        if isinstance(val, dict):
            return val
        try:
            return json.loads(val)
        except Exception:
            return {}

    eng_series = df["engagement"].apply(_parse_eng)
    df["engagement_likes"] = eng_series.apply(lambda e: e.get("likes"))
    df["engagement_helpful"] = eng_series.apply(lambda e: e.get("helpful"))
    df["engagement_comments"] = eng_series.apply(lambda e: e.get("comments"))
    df = df.drop(columns=["engagement"])

    for col in ("engagement_likes", "engagement_helpful", "engagement_comments"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ── Theme binary columns ───────────────────────────────────────────────────
    def _parse_themes(val):
        if pd.isna(val) or val == "" or val == "[]":
            return []
        if isinstance(val, list):
            return val
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                return ast.literal_eval(val)
            except Exception:
                return []

    theme_lists = df["themes"].apply(_parse_themes)
    for t in _KNOWN_THEMES:
        df[f"theme_{t}"] = theme_lists.apply(lambda lst, _t=t: _t in lst)
    df = df.drop(columns=["themes"])

    # ── Rename and reshape existing columns ───────────────────────────────────
    df = df.rename(columns={"rating": "star_rating", "author_hash": "author_id_hashed"})
    df["author_known"] = df["author_id_hashed"].notna() & (df["author_id_hashed"].astype(str).str.strip() != "")
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").astype("Int64")

    # merge sentiment: prefer star-based label for store reviews
    star_map = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    df["sentiment_final"] = df.apply(
        lambda r: star_map.get(int(r["star_rating"]), r["sentiment_label"])
        if pd.notna(r["star_rating"])
        else r["sentiment_label"],
        axis=1,
    )
    df = df.drop(columns=["sentiment_label"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").round(2)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["month_year"] = (
        df["created_at"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .where(df["created_at"].notna(), other=None)
    )
    df["is_review"] = df["source"].isin({"app_store", "google_play"})
    df["platform_type"] = df.apply(
        lambda r: _PLATFORM_MAP.get((r["source"], r["content_type"]), r["source"]),
        axis=1,
    )

    # ── Low-quality flag ──────────────────────────────────────────────────────
    text_len_clean = df["text"].fillna("").str.strip().str.len()
    df["low_quality"] = text_len_clean < 30

    # ── Column order (analyst-friendly) ──────────────────────────────────────
    front = [
        "record_id", "brand", "source", "platform_type", "content_type",
        "month_year", "created_at", "is_review", "language",
        "text", "low_quality",
        "star_rating", "sentiment_final", "sentiment_score",
    ]
    theme_cols = [f"theme_{t}" for t in _KNOWN_THEMES]
    eng_cols = ["engagement_likes", "engagement_helpful", "engagement_comments"]
    thread_cols = ["thread_id", "parent_native_id", "author_id_hashed", "author_known", "url"]
    remaining = [c for c in df.columns if c not in front + theme_cols + eng_cols + thread_cols]
    ordered = front + theme_cols + eng_cols + thread_cols + remaining
    df = df[[c for c in ordered if c in df.columns]]

    # ── Write ─────────────────────────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8", na_rep="")

    clean_count = len(df)
    dropped = raw_count - clean_count
    stats = {
        "raw_rows": raw_count,
        "clean_rows": clean_count,
        "dropped_rows": dropped,
        "columns": list(df.columns),
        "output": str(output_csv),
        "by_source": df["source"].value_counts().to_dict(),
        "by_brand": df["brand"].value_counts().to_dict(),
        "low_quality_rows": int(df["low_quality"].sum()),
        "no_theme_rows": int((~df[theme_cols].any(axis=1)).sum()),
    }
    return stats


def write_summary_report(records: list[dict[str, Any]], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (REPORTS / "summary.md")
    in_scope = [r for r in records if r.get("in_scope", True)]
    cov = coverage_summary(records)

    by_brand_source = Counter((r.get("brand"), r.get("source")) for r in in_scope)
    by_sentiment = Counter((r.get("brand"), r.get("sentiment_label")) for r in in_scope)
    theme_counter: dict[str, Counter] = defaultdict(Counter)
    for r in in_scope:
        for t in r.get("themes") or []:
            theme_counter[r.get("brand") or "unknown"][t] += 1

    lines = [
        "# BIBM Social Listening - Summary Report",
        "",
        f"Generated: `{utc_now().isoformat()}`",
        "",
        f"Total records: **{len(records)}** (in-scope: **{len(in_scope)}**)",
        "",
        "## Coverage by brand x source",
        "",
        "| Brand | Source | Count | Min date | Max date | Coverage months |",
        "|-------|--------|------:|----------|----------|----------------:|",
    ]
    for row in cov:
        lines.append(
            f"| {row['brand']} | {row['source']} | {row['count']} | "
            f"{row['min_created_at'] or '—'} | {row['max_created_at'] or '—'} | "
            f"{row['coverage_months'] if row['coverage_months'] is not None else '—'} |"
        )

    lines += ["", "## Sentiment mix (in-scope)", "", "| Brand | Label | Count |", "|-------|-------|------:|"]
    for (brand, label), n in sorted(by_sentiment.items()):
        lines.append(f"| {brand} | {label} | {n} |")

    lines += ["", "## Top themes by brand", ""]
    for brand, ctr in sorted(theme_counter.items()):
        lines.append(f"### {brand}")
        for theme, n in ctr.most_common(10):
            lines.append(f"- `{theme}`: {n}")
        lines.append("")

    lines += [
        "## Volume (brand x source)",
        "",
        "| Brand | Source | Count |",
        "|-------|--------|------:|",
    ]
    for (brand, source), n in sorted(by_brand_source.items()):
        lines.append(f"| {brand} | {source} | {n} |")

    lines += [
        "",
        "## Notes",
        "",
        "- App Store public feeds may not cover a full 12 months; see HITL coverage sign-off.",
        "- Facebook/LinkedIn browser captures may lack precise timestamps.",
        "- `rating_sentiment` on store reviews is a parallel star-based signal; see `sentiment_source` for the text-label method.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
