"""CLI: python -m listening <command>"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow `python -m listening` from repo root without install
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click

from listening.hitl import (
    list_open_gates,
    load_gold_sample,
    resolve_gate,
    save_gold_sample,
)
from listening.orchestrator import (
    collect_all,
    evaluate_gold,
    evaluate_relevance,
    prepare_hitl_qa,
    process_all,
    run_pipeline,
)
from listening.export import clean_for_analysis, export_unified, write_summary_report
from listening.utils import cutoff_since, ensure_dirs, load_config, load_env, parse_iso


@click.group()
def cli() -> None:
    """Banking Social Media Analysis pipeline."""
    load_env()
    ensure_dirs()


@cli.command("run")
@click.option("--since", "since_str", default=None, help="ISO date/datetime lower bound (UTC).")
@click.option("--since-days", default=None, type=int, help="Lookback days (default from config).")
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(
        ["google_play", "app_store", "facebook_group", "facebook", "linkedin", "reddit"]
    ),
)
@click.option("--brand", "brands", multiple=True, type=click.Choice(["brac_bank", "scb_bangladesh"]))
@click.option("--skip-collect", is_flag=True, help="Only normalize/NLP/export from existing raw data.")
@click.option("--skip-nlp-model", is_flag=True, help="Use lexicon sentiment (no transformer download).")
@click.option("--force-facebook-browser", is_flag=True, help="Skip Graph API; use Playwright.")
def run_cmd(since_str, since_days, sources, brands, skip_collect, skip_nlp_model, force_facebook_browser):
    """Collect → normalize → NLP → export → prepare HITL QA."""
    since = parse_iso(since_str) if since_str else None
    result = run_pipeline(
        since=since,
        since_days=since_days,
        sources=list(sources) or None,
        brands=list(brands) or None,
        skip_collect=skip_collect,
        skip_nlp_model=skip_nlp_model,
        force_facebook_browser=force_facebook_browser,
    )
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command("collect")
@click.option("--since", "since_str", default=None)
@click.option("--since-days", default=None, type=int)
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(
        ["google_play", "app_store", "facebook_group", "facebook", "linkedin", "reddit"]
    ),
)
@click.option("--brand", "brands", multiple=True, type=click.Choice(["brac_bank", "scb_bangladesh"]))
@click.option("--force-facebook-browser", is_flag=True)
def collect_cmd(since_str, since_days, sources, brands, force_facebook_browser):
    """Run collectors only."""
    config = load_config()
    since = parse_iso(since_str) if since_str else cutoff_since(since_days or int(config.get("since_days") or 365))
    results = collect_all(
        config,
        since,
        sources=list(sources) or None,
        brands=list(brands) or None,
        force_facebook_browser=force_facebook_browser,
    )
    click.echo(json.dumps(results, indent=2, default=str))


@cli.command("process")
@click.option("--since", "since_str", default=None)
@click.option("--since-days", default=None, type=int)
@click.option("--skip-nlp-model", is_flag=True)
def process_cmd(since_str, since_days, skip_nlp_model):
    """Normalize + NLP + export from raw JSONL."""
    config = load_config()
    since = parse_iso(since_str) if since_str else cutoff_since(since_days or int(config.get("since_days") or 365))
    records = process_all(config, since, skip_transformer=skip_nlp_model)
    paths = export_unified(records)
    report = write_summary_report(records)
    qa = prepare_hitl_qa(records, config)
    click.echo(json.dumps({"count": len(records), "exports": {k: str(v) for k, v in paths.items()}, "report": str(report), "hitl_qa": qa}, indent=2))


@cli.command("clean")
@click.option("--input", "input_path", default=None, help="Input CSV (default: data/processed/unified_mentions.csv)")
@click.option("--output", "output_path", default=None, help="Output CSV (default: data/processed/unified_mentions_clean.csv)")
def clean_cmd(input_path, output_path):
    """Clean and flatten unified_mentions.csv for brand managers."""
    from pathlib import Path

    stats = clean_for_analysis(
        input_csv=Path(input_path) if input_path else None,
        output_csv=Path(output_path) if output_path else None,
    )
    click.echo(json.dumps(stats, indent=2, default=str))


@cli.group("hitl")
def hitl_grp():
    """Human-in-the-loop gates and QA."""


@hitl_grp.command("status")
def hitl_status():
    gates = list_open_gates()
    click.echo(json.dumps(gates, indent=2, default=str))


@hitl_grp.command("resolve")
@click.option("--gate", "gate_id", required=True)
@click.option("--resolution", required=True, type=click.Choice(["resume", "skip", "accept_partial", "abort", "accept_with_caveat", "calibrate", "switch_model"]))
@click.option("--note", default="")
@click.option("--brand", default=None)
@click.option("--source", default=None)
def hitl_resolve(gate_id, resolution, note, brand, source):
    # map accept_with_caveat etc. to accept_partial for storage
    stored = resolution
    if resolution in ("accept_with_caveat", "calibrate", "switch_model"):
        stored = "accept_partial" if resolution == "accept_with_caveat" else resolution
        # allow custom resolutions in note
        note = f"{resolution}: {note}".strip()
        stored = "accept_partial"
    ok = resolve_gate(gate_id, stored, note=note, brand=brand, source=source)
    click.echo(json.dumps({"resolved": ok, "gate": gate_id, "resolution": stored}))


@hitl_grp.command("label-gold")
@click.option("--record-id", required=True)
@click.option("--label", required=True, type=click.Choice(["positive", "neutral", "negative"]))
def label_gold(record_id, label):
    rows = load_gold_sample()
    found = False
    for r in rows:
        if r.get("record_id") == record_id:
            r["human_sentiment_label"] = label
            r["labeled_at"] = datetime.utcnow().isoformat() + "Z"
            found = True
            break
    if not found:
        raise click.ClickException(f"record_id not in gold sample: {record_id}")
    save_gold_sample(rows)
    click.echo(json.dumps({"ok": True, "record_id": record_id, "label": label}))


@hitl_grp.command("evaluate-gold")
def eval_gold_cmd():
    config = load_config()
    metrics = evaluate_gold(config)
    click.echo(json.dumps(metrics, indent=2))


@hitl_grp.command("evaluate-relevance")
def eval_rel_cmd():
    config = load_config()
    metrics = evaluate_relevance(config)
    click.echo(json.dumps(metrics, indent=2))


@hitl_grp.command("sign-off-coverage")
@click.option("--note", default="Accepted coverage gaps for study freeze.")
def sign_off_coverage(note):
    from listening.hitl import GATE_COVERAGE_SIGN_OFF
    from listening.utils import DATA_HITL, read_json, write_json, utc_now

    path = DATA_HITL / "coverage_report.json"
    data = read_json(path, default={})
    data["sign_off"] = {"at": utc_now().isoformat(), "note": note}
    write_json(path, data)
    resolve_gate(GATE_COVERAGE_SIGN_OFF, "accept_partial", note=note)
    click.echo(json.dumps({"ok": True, "path": str(path)}))


def main():
    cli()


if __name__ == "__main__":
    main()
