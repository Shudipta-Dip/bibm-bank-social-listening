"""Human-in-the-loop gates, queue, and QA workflows."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from listening.utils import (
    DATA_HITL,
    HITL_EVENTS,
    HITL_QUEUE,
    append_jsonl,
    ensure_dirs,
    read_json,
    utc_now,
    write_json,
)

# Blocking gate IDs from the plan
GATE_H1_CREDENTIALS = "H1_credentials"
GATE_H2_TARGET = "H2_target_confirm"
GATE_H3_BLOCKER = "H3_blocker"
GATE_H4_COMPLETENESS = "H4_completeness"
GATE_H5_LOGIN = "H5_login"
GATE_H6_SOFT_BAN = "H6_soft_ban"
GATE_H7_COVERAGE = "H7_coverage"
GATE_COVERAGE_SIGN_OFF = "coverage_sign_off"
GATE_GOLD_SAMPLE = "gold_sample_qa"
GATE_RELEVANCE = "relevance_spot_check"
GATE_LEGAL_ABORT = "legal_ops_override"


class HitlBlockedError(Exception):
    """Raised when a blocking HITL gate pauses the pipeline."""

    def __init__(self, gate_id: str, message: str, payload: Optional[dict] = None):
        super().__init__(message)
        self.gate_id = gate_id
        self.payload = payload or {}


def _empty_queue() -> dict[str, Any]:
    return {"updated_at": None, "open_gates": []}


def load_queue() -> dict[str, Any]:
    ensure_dirs()
    return read_json(HITL_QUEUE, default=_empty_queue())


def save_queue(queue: dict[str, Any]) -> None:
    ensure_dirs()
    queue["updated_at"] = utc_now().isoformat()
    write_json(HITL_QUEUE, queue)


def log_event(
    gate_id: str,
    action: str,
    note: str = "",
    resolution: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    ensure_dirs()
    row = {
        "timestamp": utc_now().isoformat(),
        "gate_id": gate_id,
        "action": action,
        "note": note,
        "resolution": resolution,
        "extra": extra or {},
    }
    append_jsonl(HITL_EVENTS, [row])


def raise_gate(
    gate_id: str,
    message: str,
    *,
    brand: str | None = None,
    source: str | None = None,
    blocking: bool = True,
    payload: Optional[dict] = None,
) -> None:
    """Record an open gate. If blocking, also raise HitlBlockedError."""
    queue = load_queue()
    entry = {
        "gate_id": gate_id,
        "message": message,
        "brand": brand,
        "source": source,
        "blocking": blocking,
        "status": "open",
        "opened_at": utc_now().isoformat(),
        "payload": payload or {},
    }
    # replace existing same gate+brand+source
    queue["open_gates"] = [
        g
        for g in queue.get("open_gates", [])
        if not (
            g.get("gate_id") == gate_id
            and g.get("brand") == brand
            and g.get("source") == source
            and g.get("status") == "open"
        )
    ]
    queue["open_gates"].append(entry)
    save_queue(queue)
    log_event(gate_id, "opened", note=message, extra={"brand": brand, "source": source})
    try:
        print(f"[HITL] GATE {gate_id}: {message}")
    except UnicodeEncodeError:
        print(f"[HITL] GATE {gate_id}: {message.encode('ascii', 'replace').decode('ascii')}")
    if blocking:
        raise HitlBlockedError(gate_id, message, payload)


def resolve_gate(
    gate_id: str,
    resolution: str,
    note: str = "",
    brand: str | None = None,
    source: str | None = None,
) -> bool:
    """Mark matching open gate(s) resolved. resolution: resume|skip|accept_partial|abort.

    If no matching open gate exists, records a resolved entry anyway (pre-approve).
    """
    queue = load_queue()
    found = False
    for g in queue.get("open_gates", []):
        if g.get("status") != "open":
            continue
        if g.get("gate_id") != gate_id:
            continue
        if brand is not None and g.get("brand") != brand:
            continue
        if source is not None and g.get("source") != source:
            continue
        g["status"] = "resolved"
        g["resolution"] = resolution
        g["resolved_at"] = utc_now().isoformat()
        g["note"] = note
        found = True
    if not found:
        queue.setdefault("open_gates", []).append(
            {
                "gate_id": gate_id,
                "message": note or f"Pre-approved {gate_id}",
                "brand": brand,
                "source": source,
                "blocking": True,
                "status": "resolved",
                "opened_at": utc_now().isoformat(),
                "resolved_at": utc_now().isoformat(),
                "resolution": resolution,
                "note": note,
                "payload": {},
            }
        )
        found = True
    save_queue(queue)
    log_event(gate_id, "resolved", note=note, resolution=resolution, extra={"brand": brand, "source": source})
    return found


def list_open_gates(blocking_only: bool = False) -> list[dict[str, Any]]:
    queue = load_queue()
    gates = [g for g in queue.get("open_gates", []) if g.get("status") == "open"]
    if blocking_only:
        gates = [g for g in gates if g.get("blocking")]
    return gates


def has_blocking_gates() -> bool:
    return len(list_open_gates(blocking_only=True)) > 0


def gate_resolution(gate_id: str, brand: str | None = None, source: str | None = None) -> Optional[str]:
    """Return latest resolution for a gate if resolved."""
    queue = load_queue()
    matches = [
        g
        for g in queue.get("open_gates", [])
        if g.get("gate_id") == gate_id
        and (brand is None or g.get("brand") == brand)
        and (source is None or g.get("source") == source)
        and g.get("status") == "resolved"
    ]
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("resolved_at") or "", reverse=True)
    return matches[0].get("resolution")


# --- Gold sample / relevance QA ---


def stratified_sample(
    records: list[dict[str, Any]],
    n: int = 200,
    strata_keys: tuple[str, ...] = ("brand", "source", "language"),
) -> list[dict[str, Any]]:
    if not records or n <= 0:
        return []
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) or "und" for k in strata_keys)
        buckets[key].append(r)
    # round-robin across strata
    selected: list[dict] = []
    keys = list(buckets.keys())
    idxs = {k: 0 for k in keys}
    while len(selected) < n and keys:
        progressed = False
        for k in list(keys):
            i = idxs[k]
            if i >= len(buckets[k]):
                keys.remove(k)
                continue
            selected.append(buckets[k][i])
            idxs[k] = i + 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def write_gold_sample(records: list[dict[str, Any]], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_HITL / "quality_samples.jsonl")
    if path.exists():
        path.unlink()
    rows = []
    for r in records:
        rows.append(
            {
                "record_id": r.get("record_id"),
                "brand": r.get("brand"),
                "source": r.get("source"),
                "language": r.get("language"),
                "text": r.get("text"),
                "model_sentiment_label": r.get("sentiment_label"),
                "model_sentiment_score": r.get("sentiment_score"),
                "human_sentiment_label": None,
                "human_relevant": None,
                "labeled_at": None,
            }
        )
    append_jsonl(path, rows)
    return path


def load_gold_sample(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (DATA_HITL / "quality_samples.jsonl")
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_gold_sample(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    path = path or (DATA_HITL / "quality_samples.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy, macro-F1 approx, and Cohen's kappa for labeled gold sample."""
    pairs = [
        (r["human_sentiment_label"], r["model_sentiment_label"])
        for r in rows
        if r.get("human_sentiment_label") and r.get("model_sentiment_label")
    ]
    if not pairs:
        return {"n_labeled": 0, "accuracy": None, "macro_f1": None, "cohen_kappa": None}

    labels = ["positive", "neutral", "negative"]
    n = len(pairs)
    correct = sum(1 for h, m in pairs if h == m)
    accuracy = correct / n

    # per-class F1
    f1s = []
    for lab in labels:
        tp = sum(1 for h, m in pairs if h == lab and m == lab)
        fp = sum(1 for h, m in pairs if h != lab and m == lab)
        fn = sum(1 for h, m in pairs if h == lab and m != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = statistics.mean(f1s) if f1s else 0.0

    # Cohen's kappa
    human_counts = {lab: sum(1 for h, _ in pairs if h == lab) for lab in labels}
    model_counts = {lab: sum(1 for _, m in pairs if m == lab) for lab in labels}
    pe = sum((human_counts[lab] / n) * (model_counts[lab] / n) for lab in labels)
    po = accuracy
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    return {
        "n_labeled": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "cohen_kappa": round(kappa, 4),
    }


def coverage_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        if not r.get("in_scope", True):
            continue
        groups[(r.get("brand"), r.get("source"))].append(r)
    out = []
    for (brand, source), items in sorted(groups.items()):
        dates = []
        for it in items:
            ca = it.get("created_at")
            if ca:
                try:
                    dates.append(datetime.fromisoformat(str(ca).replace("Z", "+00:00")))
                except ValueError:
                    pass
        min_d = min(dates).isoformat() if dates else None
        max_d = max(dates).isoformat() if dates else None
        months = None
        if dates:
            span_days = (max(dates) - min(dates)).days
            months = round(span_days / 30.44, 1)
        out.append(
            {
                "brand": brand,
                "source": source,
                "count": len(items),
                "min_created_at": min_d,
                "max_created_at": max_d,
                "coverage_months": months,
            }
        )
    return out


def write_coverage_report(records: list[dict[str, Any]], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_HITL / "coverage_report.json")
    summary = coverage_summary(records)
    write_json(
        path,
        {
            "generated_at": utc_now().isoformat(),
            "sign_off": None,
            "summary": summary,
        },
    )
    return path
