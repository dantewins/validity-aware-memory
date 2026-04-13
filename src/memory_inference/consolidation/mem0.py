from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.consolidation.semantic_utils import (
    HashedSemanticEncoder,
    entry_search_text,
    normalize_text,
    query_search_text,
)
from memory_inference.open_ended_eval import expand_with_support_entries, is_open_ended_query
from memory_inference.types import MemoryEntry, Query, RetrievalResult


class Mem0MemoryPolicy(BaseMemoryPolicy):
    """Mem0-style compact memory store with semantic retrieval."""

    def __init__(self) -> None:
        super().__init__(name="mem0")
        self.encoder = HashedSemanticEncoder()
        self.episodic_log: List[MemoryEntry] = []
        self.active_state: Dict[tuple[str, str, str], MemoryEntry] = {}
        self.archived_state: Dict[tuple[str, str], List[MemoryEntry]] = {}
        self.evidence_bank: Dict[tuple[str, str, str, str], MemoryEntry] = {}
        self._entry_vectors: dict[str, tuple[float, ...]] = {}

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        for update in updates:
            self.episodic_log.append(update)
            self._remember(update)

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        query = Query(
            query_id="mem0-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            answer="",
            timestamp=max((entry.timestamp for entry in self.episodic_log), default=0),
            session_id="mem0-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        candidates = self._candidate_pool(query)
        ranked = self._rank(query, candidates)
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        top_entries = ranked[:limit]
        if self._has_supportable_fact(top_entries):
            top_entries = expand_with_support_entries(
                top_entries,
                self.episodic_log,
                support_limit=2,
                max_entries=limit + 2,
            )
        return RetrievalResult(
            entries=top_entries,
            debug={
                "policy": self.name,
                "retrieval_mode": "compact_semantic_memory",
            },
        )

    def snapshot_size(self) -> int:
        return (
            len(self.active_state)
            + len(self.evidence_bank)
            + sum(len(entries) for entries in self.archived_state.values())
        )

    def _remember(self, entry: MemoryEntry) -> None:
        if self._is_state_memory(entry):
            state_key = (entry.entity, entry.attribute, entry.scope)
            existing = self.active_state.get(state_key)
            if existing is None:
                self.active_state[state_key] = self._with_memory_metadata(entry, access_count=1)
            elif normalize_text(existing.value) == normalize_text(entry.value):
                self.active_state[state_key] = self._merge(existing, entry)
            elif entry.timestamp >= existing.timestamp:
                self.archived_state.setdefault(entry.key, []).append(existing)
                self.active_state[state_key] = self._with_memory_metadata(entry, access_count=1)
            else:
                archived = self._with_memory_metadata(entry, access_count=1)
                self.archived_state.setdefault(entry.key, []).append(archived)
                self._cache_entry(archived)
            self._cache_entry(self.active_state[state_key])
            return

        evidence_key = (
            entry.entity,
            entry.attribute,
            entry.scope,
            normalize_text(entry.value),
        )
        existing_evidence = self.evidence_bank.get(evidence_key)
        if existing_evidence is None or entry.timestamp >= existing_evidence.timestamp:
            stored = self._merge(existing_evidence, entry) if existing_evidence is not None else self._with_memory_metadata(entry, access_count=1)
            self.evidence_bank[evidence_key] = stored
            self._cache_entry(stored)

    def _candidate_pool(self, query: Query) -> list[MemoryEntry]:
        active = [
            entry
            for (entity, attribute, _scope), entry in self.active_state.items()
            if self._entity_matches(entity, query.entity)
            and (
                is_open_ended_query(query)
                or attribute == query.attribute
            )
        ]
        evidence = [
            entry
            for entry in self.evidence_bank.values()
            if self._entity_matches(entry.entity, query.entity)
            and (
                is_open_ended_query(query)
                or entry.attribute == query.attribute
                or entry.attribute in {"dialogue", "event"}
            )
        ]
        archived = []
        if query.query_mode == QueryMode.HISTORY:
            archived = [
                entry
                for (entity, attribute), entries in self.archived_state.items()
                if self._entity_matches(entity, query.entity) and attribute == query.attribute
                for entry in entries
            ]

        if is_open_ended_query(query):
            candidates = active + evidence + archived
        else:
            candidates = active + archived
            if not candidates:
                candidates = evidence

        if candidates:
            return self._dedupe(candidates)
        return self._dedupe(self.episodic_log)

    def _rank(self, query: Query, candidates: Iterable[MemoryEntry]) -> list[MemoryEntry]:
        query_vector = self.encoder.encode(query_search_text(query))
        return sorted(
            self._dedupe(candidates),
            key=lambda entry: self._score(entry, query, query_vector),
            reverse=True,
        )

    def _score(
        self,
        entry: MemoryEntry,
        query: Query,
        query_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        dense_similarity = self.encoder.similarity(query_vector, self._entry_vectors[entry.entry_id])
        entity_bonus = 1.0 if self._entity_matches(entry.entity, query.entity) else 0.0
        attribute_bonus = 1.0 if entry.attribute == query.attribute else 0.0
        memory_kind_bonus = 0.2 if entry.metadata.get("memory_kind", "state") == "state" else 0.0
        structured_bonus = 0.25 if entry.metadata.get("source_kind") == "structured_fact" else 0.0
        if query.query_mode == QueryMode.HISTORY:
            time_bias = -float(entry.timestamp)
        else:
            time_bias = float(entry.timestamp)
        return (
            dense_similarity,
            entity_bonus + attribute_bonus + memory_kind_bonus + structured_bonus,
            entry.importance,
            float(entry.access_count),
            entry.confidence,
            time_bias,
        )

    def _merge(self, existing: MemoryEntry, new: MemoryEntry) -> MemoryEntry:
        latest = new if new.timestamp >= existing.timestamp else existing
        access_count = max(existing.access_count, 1) + 1
        return dataclasses.replace(
            latest,
            importance=max(existing.importance, new.importance),
            confidence=max(existing.confidence, new.confidence),
            access_count=access_count,
        )

    def _with_memory_metadata(self, entry: MemoryEntry, *, access_count: int) -> MemoryEntry:
        return dataclasses.replace(
            entry,
            access_count=max(entry.access_count, access_count),
            metadata={
                **entry.metadata,
                "memory_kind": entry.metadata.get("memory_kind", "state" if self._is_state_memory(entry) else "event"),
            },
        )

    def _cache_entry(self, entry: MemoryEntry) -> None:
        self._entry_vectors[entry.entry_id] = self.encoder.encode(entry_search_text(entry))

    def _is_state_memory(self, entry: MemoryEntry) -> bool:
        if entry.metadata.get("memory_kind") == "state":
            return True
        if entry.metadata.get("source_kind") == "structured_fact":
            return True
        return entry.attribute not in {"dialogue", "event"}

    def _entity_matches(self, entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    def _dedupe(self, entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
        unique: dict[str, MemoryEntry] = {}
        for entry in entries:
            unique.setdefault(entry.entry_id, entry)
        return list(unique.values())

    def _has_supportable_fact(self, entries: Iterable[MemoryEntry]) -> bool:
        return any(entry.metadata.get("source_kind") == "structured_fact" for entry in entries)
