from __future__ import annotations

from typing import Dict, Iterable

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.types import MemoryEntry, MemoryKey, RetrievalResult


class SummaryOnlyMemoryPolicy(BaseMemoryPolicy):
    """Keep only the current latest fact per key."""

    def __init__(self) -> None:
        super().__init__(name="summary_only")
        self.current: Dict[MemoryKey, MemoryEntry] = {}

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        for update in updates:
            existing = self.current.get(update.key)
            if existing is None or update.timestamp >= existing.timestamp:
                self.current[update.key] = update

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        entry = self.current.get((entity, attribute))
        return RetrievalResult(entries=[entry] if entry else [], debug={"policy": self.name})

    def snapshot_size(self) -> int:
        return len(self.current)
