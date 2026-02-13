from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class StepResult:
    name: str
    seconds: float
    meta: Dict[str, Any]


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def step(self, name: str, meta: Optional[Dict[str, Any]] = None):
        return _StepContext(self, name, meta or {})


class _StepContext:
    def __init__(self, logger: RunLogger, name: str, meta: Dict[str, Any]):
        self.logger = logger
        self.name = name
        self.meta = meta
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.log({"event": "step_start", "step": self.name, **self.meta})
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        payload = {"event": "step_end", "step": self.name, "seconds": round(dt, 3), **self.meta}
        if exc:
            payload["error"] = str(exc)
        self.logger.log(payload)
        return False  # don't swallow exceptions
