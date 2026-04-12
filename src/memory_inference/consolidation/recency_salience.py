from __future__ import annotations

from typing import Iterable, List, Tuple

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.types import MemoryEntry, RetrievalResult


class RecencySalienceMemoryPolicy(BaseMemoryPolicy):
    """Non-consolidating baseline using recency, confidence, and importance."""

    def __init__(self, recency_weight: float = 1.0, importance_weight: float = 0.75) -> None:
        super().__init__(name="recency_salience")
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
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
        entity_match = 1.0 if entry.entity == entity else 0.0
        attribute_match = 1.0 if entry.attribute == attribute else 0.0
        salience = (
            self.importance_weight * entry.importance
            + 0.25 * entry.confidence
            + 0.1 * entry.access_count
        )
        recency = self.recency_weight * entry.timestamp
        return (entity_match + attribute_match, salience, recency)
