"""JSONL audit log of cues + ReAct traces (Liar-Ai 'knowledge base' analogue)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class SessionLog:
    def __init__(self, path=None):
        logs = Path("logs")
        logs.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(path) if path else logs / "session-{}.jsonl".format(stamp)
        self.n = 0

    def write(self, payload, force=False, every=20):
        self.n += 1
        if not force and not payload.get("review") and self.n % every:
            return
        row = {"t": datetime.now().isoformat(timespec="seconds"), "frame": self.n}
        row.update(payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
