"""Base collector helpers: run logging, retries, raw writes."""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from listening.utils import (
    COLLECTION_RUNS,
    append_jsonl,
    ensure_dirs,
    raw_path,
    utc_now,
)


@dataclass
class CollectorResult:
    source: str
    brand: str
    run_id: str
    status: str = "ok"  # ok | partial | error | hitl_blocked
    item_count: int = 0
    error_summary: Optional[str] = None
    hitl_flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: utc_now().isoformat())
    finished_at: Optional[str] = None

    def finish(self, status: str | None = None) -> "CollectorResult":
        if status:
            self.status = status
        self.finished_at = utc_now().isoformat()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "brand": self.brand,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "item_count": self.item_count,
            "error_summary": self.error_summary,
            "hitl_flags": self.hitl_flags,
            "meta": self.meta,
        }


def new_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def log_run(result: CollectorResult) -> None:
    ensure_dirs()
    append_jsonl(COLLECTION_RUNS, [result.to_dict()])


def write_raw_items(source: str, brand: str, run_id: str, items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    path = raw_path(source, brand, run_id)
    return append_jsonl(path, items)


def format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
