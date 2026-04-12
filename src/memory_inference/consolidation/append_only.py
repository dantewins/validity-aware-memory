from __future__ import annotations

from typing import Iterable, List

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.types import MemoryEntry, RetrievalResult


class AppendOnlyMemoryPolicy(BaseMemoryPolicy):
    def __init__(self) -> None:
        super().__init__(name="append_only")
        self.entries: List[MemoryEntry] = []

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        self.entries.extend(updates)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        matches = [
            entry for entry in self.entries if entry.entity == entity and entry.attribute == attribute
        ]
        matches.sort(key=lambda item: item.timestamp, reverse=True)
        return RetrievalResult(entries=matches[:top_k], debug={"policy": self.name})

    def snapshot_size(self) -> int:
        return len(self.entries)
