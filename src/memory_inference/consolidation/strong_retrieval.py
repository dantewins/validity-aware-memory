from __future__ import annotations

from typing import Iterable, List, Tuple

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.types import MemoryEntry, RetrievalResult


class StrongRetrievalMemoryPolicy(BaseMemoryPolicy):
    """Retrieval-only baseline with stronger ranking over raw history."""

    def __init__(self) -> None:
        super().__init__(name="strong_retrieval")
        self.entries: List[MemoryEntry] = []

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        self.entries.extend(updates)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        ranked = sorted(
            self.entries,
            key=lambda entry: self._score(entry, entity, attribute),
            reverse=True,
        )
        return RetrievalResult(entries=ranked[:top_k], debug={"policy": self.name})

    def snapshot_size(self) -> int:
        return len(self.entries)

    def _score(self, entry: MemoryEntry, entity: str, attribute: str) -> Tuple[float, ...]:
        exact_entity = 1.0 if entry.entity == entity else 0.0
        exact_attribute = 1.0 if entry.attribute == attribute else 0.0
        scope_bonus = 1.0 if entry.scope == "default" else 0.5
        return (
            exact_entity + exact_attribute,
            entry.importance,
            float(entry.access_count),
            scope_bonus,
            float(entry.timestamp),
        )
