from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .guardrails import mask_pii


class Trace:
    """Per-request trace (§13): stages, latencies, IDs and scores, written as JSON lines."""

    def __init__(self, kind: str, log_dir: Path | None = None, mask: bool = True):
        self.trace_id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.log_dir = log_dir
        self.mask = mask
        self.started = time.perf_counter()
        self.data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "kind": kind,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "latency_ms": {},
            "errors": [],
        }

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.data["errors"].append({"stage": name, "error": str(exc)})
            raise
        finally:
            self.data["latency_ms"][name] = round((time.perf_counter() - start) * 1000, 1)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def finish(self) -> dict[str, Any]:
        self.data["latency_ms"]["total"] = round((time.perf_counter() - self.started) * 1000, 1)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            data = _mask_strings(self.data) if self.mask else self.data
            record = json.dumps(data, ensure_ascii=False, default=str)
            with (self.log_dir / f"{self.kind}_traces.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record + "\n")
        return self.data


def _mask_strings(value: Any) -> Any:
    """Mask PII in text fields only, so numeric scores and IDs stay intact."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {key: _mask_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_strings(item) for item in value]
    return value
