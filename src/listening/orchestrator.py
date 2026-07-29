"""Pipeline orchestrator: collect → normalize → NLP → export → HITL QA hooks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

from listening.collectors.appstore import collect_app_store
from listening.collectors.facebook import collect_facebook
from listening.collectors.facebook_group import collect_facebook_group
from listening.collectors.linkedin import collect_linkedin
from listening.collectors.play import collect_google_play
from listening.collectors.reddit import collect_reddit
from listening.export import export_unified, write_summary_report
from listening.hitl import (
    GATE_GOLD_SAMPLE,
    GATE_RELEVANCE,
    HitlBlockedError,
    agreement_metrics,
    has_blocking_gates,
    list_open_gates,
    load_gold_sample,
    raise_gate,
    stratified_sample,
    write_coverage_report,
    write_gold_sample,
)
from listening.nlp import enrich_languages, enrich_sentiment, enrich_themes
from listening.normalize import normalize_corpus
from listening.utils import (
    DATA_PROCESSED,
    cutoff_since,
    ensure_dirs,
    load_config,
    load_env,
    parse_iso,
    write_json,
)


def _brand_items(config: dict[str, Any]) -> list[tuple[str, dict]]:
    return list((config.get("brands") or {}).items())


def collect_all(
    config: dict[str, Any],
    since: datetime,
    sources: list[str] | None = None,
    brands: list[str] | None = None,
    force_facebook_browser: bool = False,
) -> list[dict[str, Any]]:
    ensure_dirs()
    coll = config.get("collection") or {}
    wanted_sources = set(
        sources or ["google_play", "app_store", "facebook_group", "reddit"]
    )
    brand_map = dict(_brand_items(config))
    if brands:
        brand_map = {k: v for k, v in brand_map.items() if k in brands}

    results = []

    # Stores in parallel
    store_jobs = []
    if "google_play" in wanted_sources:
        for brand, cfg in brand_map.items():
            store_jobs.append(("google_play", brand, cfg))
    if "app_store" in wanted_sources:
        for brand, cfg in brand_map.items():
            store_jobs.append(("app_store", brand, cfg))

    def _run_store(job):
        source, brand, cfg = job
        if source == "google_play":
            return collect_google_play(
                brand,
                cfg,
                since,
                batch_size=int(coll.get("play_batch_size") or 200),
                max_retries=int(coll.get("max_retries") or 3),
            )
        return collect_app_store(
            brand,
            cfg,
            since,
            min_coverage_months=int(coll.get("app_store_min_coverage_months") or 6),
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_run_store, j): j for j in store_jobs}
        for fut in as_completed(futs):
            results.append(fut.result().to_dict())

    # Social serial
    if "facebook_group" in wanted_sources:
        for brand, cfg in brand_map.items():
            try:
                results.append(
                    collect_facebook_group(
                        brand, cfg, since, config=config
                    ).to_dict()
                )
            except HitlBlockedError as e:
                results.append(
                    {
                        "source": "facebook_group",
                        "brand": brand,
                        "status": "hitl_blocked",
                        "error_summary": str(e),
                        "hitl_flags": [e.gate_id],
                    }
                )

    # Legacy owned-Page collector; no longer part of the default run.
    if "facebook" in wanted_sources:
        for brand, cfg in brand_map.items():
            try:
                results.append(
                    collect_facebook(brand, cfg, since, force_browser=force_facebook_browser).to_dict()
                )
            except HitlBlockedError as e:
                results.append(
                    {
                        "source": "facebook",
                        "brand": brand,
                        "status": "hitl_blocked",
                        "error_summary": str(e),
                        "hitl_flags": [e.gate_id],
                    }
                )

    if "linkedin" in wanted_sources:
        for brand, cfg in brand_map.items():
            try:
                results.append(
                    collect_linkedin(
                        brand,
                        cfg,
                        since,
                        delay_min_ms=int(coll.get("linkedin_delay_ms_min") or 3000),
                        delay_max_ms=int(coll.get("linkedin_delay_ms_max") or 7000),
                    ).to_dict()
                )
            except HitlBlockedError as e:
                results.append(
                    {
                        "source": "linkedin",
                        "brand": brand,
                        "status": "hitl_blocked",
                        "error_summary": str(e),
                        "hitl_flags": [e.gate_id],
                    }
                )

    if "reddit" in wanted_sources:
        for brand, cfg in brand_map.items():
            try:
                results.append(collect_reddit(brand, cfg, since, config=config).to_dict())
            except HitlBlockedError as e:
                results.append(
                    {
                        "source": "reddit",
                        "brand": brand,
                        "status": "hitl_blocked",
                        "error_summary": str(e),
                        "hitl_flags": [e.gate_id],
                    }
                )

    write_json(DATA_PROCESSED / "last_collect_results.json", results)
    return results


def process_all(
    config: dict[str, Any],
    since: datetime,
    skip_transformer: bool = False,
) -> list[dict[str, Any]]:
    records = normalize_corpus(since=since)
    records = enrich_languages(records)
    theme_lex = (config.get("nlp") or {}).get("themes") or {}
    records = enrich_themes(records, theme_lex)

    sent_cfg = config.get("sentiment") or {}
    model = sent_cfg.get("model") or "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    if skip_transformer:
        # force lexicon path by using impossible model then catch — easier: patch via env
        from listening.nlp import sentiment as sentiment_mod

        def _lex_only(text, model_name):
            return sentiment_mod._lexicon_sentiment(text)

        sentiment_mod.predict_sentiment = _lex_only  # type: ignore

    records = enrich_sentiment(
        records,
        model_name=model,
        use_rating_for_reviews=bool(sent_cfg.get("use_rating_sentiment_for_reviews", True)),
    )
    if skip_transformer:
        for record in records:
            record["sentiment_source"] = (
                "lexicon+rating"
                if record.get("content_type") == "review"
                and record.get("rating_sentiment")
                else "lexicon"
            )
    return records


def prepare_hitl_qa(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    coll = config.get("collection") or {}
    gold_n = int(coll.get("gold_sample_size") or 200)
    rel_n = int(coll.get("relevance_sample_size") or 50)

    in_scope = [r for r in records if r.get("in_scope", True) and (r.get("text") or "").strip()]
    gold = stratified_sample(in_scope, n=gold_n)
    gold_path = write_gold_sample(gold)

    social = [r for r in in_scope if r.get("source") in ("facebook", "linkedin", "reddit")]
    relevance = stratified_sample(social, n=min(rel_n, len(social)), strata_keys=("brand", "source"))
    # merge relevance flags into gold file rows when overlapping; also write separate list
    from listening.utils import DATA_HITL, append_jsonl

    rel_path = DATA_HITL / "relevance_samples.jsonl"
    if rel_path.exists():
        rel_path.unlink()
    append_jsonl(
        rel_path,
        [
            {
                "record_id": r.get("record_id"),
                "brand": r.get("brand"),
                "source": r.get("source"),
                "text": r.get("text"),
                "human_relevant": None,
            }
            for r in relevance
        ],
    )

    cov_path = write_coverage_report(records)

    try:
        raise_gate(
            GATE_GOLD_SAMPLE,
            f"Label >={gold_n} gold samples in {gold_path} then run `python -m listening hitl evaluate-gold`.",
            blocking=False,
            payload={"path": str(gold_path), "n": len(gold)},
        )
    except HitlBlockedError:
        pass
    try:
        raise_gate(
            GATE_RELEVANCE,
            f"Spot-check relevance for {len(relevance)} social items in {rel_path}.",
            blocking=False,
            payload={"path": str(rel_path), "n": len(relevance)},
        )
    except HitlBlockedError:
        pass

    return {"gold_path": str(gold_path), "relevance_path": str(rel_path), "coverage_path": str(cov_path)}


def evaluate_gold(config: dict[str, Any]) -> dict[str, Any]:
    coll = config.get("collection") or {}
    rows = load_gold_sample()
    metrics = agreement_metrics(rows)
    f1_th = float(coll.get("sentiment_agreement_threshold") or 0.65)
    k_th = float(coll.get("cohen_kappa_threshold") or 0.5)
    metrics["pass_f1"] = (metrics.get("macro_f1") or 0) >= f1_th if metrics.get("n_labeled") else False
    metrics["pass_kappa"] = (metrics.get("cohen_kappa") or 0) >= k_th if metrics.get("n_labeled") else False
    metrics["decision_needed"] = not (metrics["pass_f1"] and metrics["pass_kappa"])
    write_json(DATA_PROCESSED / "gold_metrics.json", metrics)
    if metrics.get("n_labeled", 0) == 0:
        print("[HITL] No human labels found yet in quality_samples.jsonl")
    elif metrics["decision_needed"]:
        print(
            f"[HITL] Agreement below threshold (macro_f1={metrics.get('macro_f1')}, "
            f"kappa={metrics.get('cohen_kappa')}). Choose: accept_with_caveat | calibrate | switch_model "
            f"via `python -m listening hitl resolve --gate {GATE_GOLD_SAMPLE} --resolution accept_partial`"
        )
        try:
            raise_gate(
                GATE_GOLD_SAMPLE,
                "Gold-sample agreement below threshold; decide before final report freeze.",
                blocking=True,
                payload=metrics,
            )
        except HitlBlockedError:
            pass
    else:
        print("[HITL] Gold-sample agreement OK.")
    return metrics


def evaluate_relevance(config: dict[str, Any]) -> dict[str, Any]:
    from listening.utils import DATA_HITL, iter_jsonl

    coll = config.get("collection") or {}
    threshold = float(coll.get("relevance_noise_threshold") or 0.15)
    path = DATA_HITL / "relevance_samples.jsonl"
    rows = list(iter_jsonl(path)) if path.exists() else []
    labeled = [r for r in rows if r.get("human_relevant") is not None]
    if not labeled:
        return {"n_labeled": 0, "noise_rate": None}
    noise = sum(1 for r in labeled if r.get("human_relevant") in (False, "false", "no", 0, "0"))
    rate = noise / len(labeled)
    out = {"n_labeled": len(labeled), "noise_rate": round(rate, 4), "above_threshold": rate > threshold}
    write_json(DATA_PROCESSED / "relevance_metrics.json", out)
    if out["above_threshold"]:
        print(f"[HITL] Relevance noise {rate:.0%} > {threshold:.0%}. Tighten filters.")
        try:
            raise_gate(
                GATE_RELEVANCE,
                f"Off-brand noise {rate:.0%} exceeds {threshold:.0%}.",
                blocking=False,
                payload=out,
            )
        except HitlBlockedError:
            pass
    return out


def run_pipeline(
    since: Optional[datetime] = None,
    since_days: Optional[int] = None,
    sources: list[str] | None = None,
    brands: list[str] | None = None,
    skip_collect: bool = False,
    skip_nlp_model: bool = False,
    force_facebook_browser: bool = False,
) -> dict[str, Any]:
    load_env()
    ensure_dirs()
    config = load_config()
    since_dt = since or cutoff_since(since_days or int(config.get("since_days") or 365))

    collect_results = []
    if not skip_collect:
        if has_blocking_gates():
            open_g = list_open_gates(blocking_only=True)
            print("[HITL] Blocking gates open — resolve before collect:")
            for g in open_g:
                print(f"  - {g.get('gate_id')}: {g.get('message')}")
            return {"status": "hitl_blocked", "gates": open_g}
        collect_results = collect_all(
            config,
            since_dt,
            sources=sources,
            brands=brands,
            force_facebook_browser=force_facebook_browser,
        )

    records = process_all(config, since_dt, skip_transformer=skip_nlp_model)
    paths = export_unified(records)
    report = write_summary_report(records)
    qa = prepare_hitl_qa(records, config)

    return {
        "status": "ok",
        "since": since_dt.isoformat(),
        "record_count": len(records),
        "collect_results": collect_results,
        "exports": {k: str(v) for k, v in paths.items()},
        "report": str(report),
        "hitl_qa": qa,
        "open_gates": list_open_gates(),
    }
