from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from memory_inference.llm.base import ReasonerTrace


class ResponseCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Optional[ReasonerTrace]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return ReasonerTrace(**payload)

    def save(self, key: str, trace: ReasonerTrace) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(asdict(trace), indent=2))


def cache_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
