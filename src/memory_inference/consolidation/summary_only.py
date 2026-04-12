from __future__ import annotations

from typing import Dict, Iterable

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.open_ended_eval import (
    has_structured_fact_candidates,
    is_open_ended_query,
    lexical_retrieval,
    rerank_structured_candidates,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, MemoryKey, RetrievalResult
from memory_inference.types import Query


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

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            candidates = shortlist_open_ended_candidates(
                self.current.values(),
                query,
                score_fn=lambda entry: (float(entry.timestamp),),
                limit=max(top_k * 8, 32),
            )
            return lexical_retrieval(
                candidates,
                query,
                top_k=max(top_k, 8),
                policy_name=self.name,
                secondary_score_fn=lambda entry: (float(entry.timestamp),),
            )
        candidates = [
            entry for entry in self.current.values()
            if entry.attribute == query.attribute
        ]
        if has_structured_fact_candidates(candidates):
            return rerank_structured_candidates(
                candidates,
                query,
                top_k=top_k,
                policy_name=self.name,
                score_fn=lambda entry: (float(entry.timestamp),),
                support_entries=self.current.values(),
                shortlist_limit=max(top_k * 8, 32),
                support_limit=1,
            )
        return self.retrieve(query.entity, query.attribute, top_k=top_k)

    def snapshot_size(self) -> int:
        return len(self.current)
