"""Shared paths, config loading, hashing, JSONL I/O."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_HITL = ROOT / "data" / "hitl"
DATA_CHECKPOINTS = ROOT / "data" / "checkpoints"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "brands.yaml"
HITL_QUEUE = DATA_HITL / "hitl_queue.json"
HITL_EVENTS = DATA_HITL / "hitl_events.jsonl"
COLLECTION_RUNS = DATA_PROCESSED / "collection_runs.jsonl"


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, DATA_HITL, DATA_CHECKPOINTS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cutoff_since(since_days: int | None = None, since: datetime | None = None) -> datetime:
    if since is not None:
        if since.tzinfo is None:
            return since.replace(tzinfo=timezone.utc)
        return since.astimezone(timezone.utc)
    days = since_days if since_days is not None else 365
    return utc_now() - timedelta(days=days)


def parse_iso(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        # treat as ms if huge
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        from dateutil import parser as date_parser

        dt = date_parser.parse(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).strip()


def author_hash(author_key: str | None) -> str | None:
    if not author_key:
        return None
    return hashlib.sha256(str(author_key).encode("utf-8")).hexdigest()


def make_record_id(source: str, brand: str, native_id: str) -> str:
    payload = f"{source}|{brand}|{native_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_path(source: str, brand: str) -> Path:
    return DATA_CHECKPOINTS / f"{source}_{brand}.json"


def load_checkpoint(source: str, brand: str) -> dict[str, Any]:
    return read_json(checkpoint_path(source, brand), default={})


def save_checkpoint(source: str, brand: str, data: dict[str, Any]) -> None:
    write_json(checkpoint_path(source, brand), data)


def raw_path(source: str, brand: str, run_id: str) -> Path:
    return DATA_RAW / source / brand / f"{run_id}.jsonl"


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}
