from __future__ import annotations

from typing import Iterable, List, Set, Tuple

from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.open_ended_eval import (
    expand_with_support_entries,
    has_structured_fact_candidates,
    is_open_ended_query,
    lexical_retrieval,
    shortlist_open_ended_candidates,
)
from memory_inference.types import MemoryEntry, Query, RetrievalResult


class ODV2HybridMemoryPolicy(OfflineDeltaConsolidationPolicyV2):
    """Retrieval-augmented ODV2 for raw conversational benchmarks.

    This policy keeps the ODV2 consolidation machinery unchanged, but inference
    uses a strong-retrieval backbone over raw evidence and overlays state
    candidates on top. The goal is pragmatic benchmark accuracy rather than a
    pure state-only memory system.
    """

    def __init__(
        self,
        consolidator: BaseConsolidator,
        importance_threshold: float = 0.1,
        support_history_limit: int = 3,
        maintenance_frequency: int = 1,
    ) -> None:
        super().__init__(
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=support_history_limit,
            maintenance_frequency=maintenance_frequency,
        )
        self.name = "odv2_hybrid"

    def retrieve_for_query(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if is_open_ended_query(query):
            return self._retrieve_open_ended_backbone(query, top_k=max(top_k, 8))
        hybrid = self._retrieve_hybrid_structured_query(query, top_k=max(top_k, 8))
        if hybrid is not None:
            return hybrid
        return self._retrieve_backbone_only(query, top_k=max(top_k, 8))

    def _retrieve_open_ended_backbone(self, query: Query, *, top_k: int) -> RetrievalResult:
        candidates = shortlist_open_ended_candidates(
            self.episodic_log,
            query,
            score_fn=lambda entry: self._backbone_score(entry, query.entity, query.attribute),
            limit=max(top_k * 16, 64),
        )
        return lexical_retrieval(
            candidates,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._backbone_score(entry, query.entity, query.attribute),
        )

    def _retrieve_hybrid_structured_query(
        self,
        query: Query,
        *,
        top_k: int,
    ) -> RetrievalResult | None:
        structured_candidates = self._hybrid_structured_candidates(query)
        if not has_structured_fact_candidates(structured_candidates):
            return None

        structured_shortlist = shortlist_open_ended_candidates(
            structured_candidates,
            query,
            score_fn=lambda entry: self._hybrid_structured_score(entry, query),
            limit=max(top_k * 12, 48),
        )
        structured_ranked = lexical_retrieval(
            structured_shortlist,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._hybrid_structured_score(entry, query),
        )

        anchor_source_ids = self._anchor_source_ids(structured_ranked.entries)
        anchor_scopes = self._anchor_scopes(structured_ranked.entries)
        evidence_candidates = self._evidence_candidates(
            query,
            anchor_source_ids=anchor_source_ids,
            anchor_scopes=anchor_scopes,
        )
        evidence_shortlist = shortlist_open_ended_candidates(
            evidence_candidates,
            query,
            score_fn=lambda entry: self._evidence_score(
                entry,
                query,
                anchor_source_ids=anchor_source_ids,
                anchor_scopes=anchor_scopes,
            ),
            limit=max(top_k * 16, 64),
        )
        evidence_ranked = lexical_retrieval(
            evidence_shortlist,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._evidence_score(
                entry,
                query,
                anchor_source_ids=anchor_source_ids,
                anchor_scopes=anchor_scopes,
            ),
        )

        merged = self._merge_hybrid_entries(
            state_entries=structured_ranked.entries,
            evidence_entries=evidence_ranked.entries,
            top_k=top_k,
        )
        expanded = expand_with_support_entries(
            merged,
            self.episodic_log,
            support_limit=self.support_history_limit,
            max_entries=top_k + self.support_history_limit,
        )
        return RetrievalResult(
            entries=expanded,
            debug={
                "policy": self.name,
                "retrieval_mode": "hybrid_state_evidence",
            },
        )

    def _retrieve_backbone_only(self, query: Query, *, top_k: int) -> RetrievalResult:
        candidates = [
            entry
            for entry in self.episodic_log
            if self._entity_matches(entry.entity, query.entity)
            and entry.attribute in {query.attribute, "dialogue", "event"}
        ]
        shortlisted = shortlist_open_ended_candidates(
            candidates,
            query,
            score_fn=lambda entry: self._backbone_score(entry, query.entity, query.attribute),
            limit=max(top_k * 16, 64),
        )
        return lexical_retrieval(
            shortlisted,
            query,
            top_k=top_k,
            policy_name=self.name,
            secondary_score_fn=lambda entry: self._backbone_score(entry, query.entity, query.attribute),
        )

    def _hybrid_structured_candidates(self, query: Query) -> List[MemoryEntry]:
        if query.query_mode == QueryMode.HISTORY:
            entries = [
                entry
                for entry in self.episodic_log
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
            ]
        else:
            current = self._current_entries(query.entity, query.attribute)
            archived = self._archive_entries(query.entity, query.attribute)
            conflicts = self._conflict_entries(query.entity, query.attribute)
            episodic_structured = [
                entry
                for entry in self.episodic_log
                if entry.attribute == query.attribute
                and self._entity_matches(entry.entity, query.entity)
                and self._is_structured_fact(entry)
            ]
            entries = current + archived + conflicts + episodic_structured

        combined: List[MemoryEntry] = []
        seen_ids: Set[str] = set()
        for entry in entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            combined.append(entry)
        return combined

    def _evidence_candidates(
        self,
        query: Query,
        *,
        anchor_source_ids: Set[str],
        anchor_scopes: Set[str],
    ) -> List[MemoryEntry]:
        candidates: List[MemoryEntry] = []
        seen_ids: Set[str] = set()

        for entry in self.episodic_log:
            if not self._entity_matches(entry.entity, query.entity):
                continue
            if entry.attribute not in {"dialogue", "event", query.attribute}:
                continue
            if entry.entry_id in seen_ids:
                continue
            if (
                entry.entry_id in anchor_source_ids
                or entry.scope in anchor_scopes
                or entry.attribute in {"dialogue", "event"}
                or entry.attribute == query.attribute
            ):
                seen_ids.add(entry.entry_id)
                candidates.append(entry)
        return candidates

    def _hybrid_structured_score(self, entry: MemoryEntry, query: Query) -> tuple[float, ...]:
        status_bonus = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.REINFORCED: 0.9,
            MemoryStatus.CONFLICTED: 0.5,
            MemoryStatus.SUPERSEDED: 0.3,
            MemoryStatus.ARCHIVED: 0.2,
        }.get(entry.status, 0.0)
        memory_kind_bonus = 0.4 if self._memory_kind(entry) == "state" else 0.0
        support_bonus = 0.2 if entry.metadata.get("source_entry_id") else 0.0
        return self._backbone_score(entry, query.entity, query.attribute) + (
            status_bonus,
            memory_kind_bonus,
            support_bonus,
        )

    def _evidence_score(
        self,
        entry: MemoryEntry,
        query: Query,
        *,
        anchor_source_ids: Set[str],
        anchor_scopes: Set[str],
    ) -> tuple[float, ...]:
        anchor_bonus = 1.5 if entry.entry_id in anchor_source_ids else 0.0
        scope_bonus = 1.0 if entry.scope in anchor_scopes else 0.0
        attribute_bonus = (
            1.0 if entry.attribute == query.attribute
            else 0.8 if entry.attribute in {"dialogue", "event"}
            else 0.0
        )
        if query.query_mode == QueryMode.HISTORY:
            time_bias = -float(entry.timestamp)
        else:
            time_bias = float(entry.timestamp)
        return (
            anchor_bonus,
            scope_bonus,
            attribute_bonus,
            entry.importance,
            entry.confidence,
            time_bias,
        )

    def _merge_hybrid_entries(
        self,
        *,
        state_entries: Iterable[MemoryEntry],
        evidence_entries: Iterable[MemoryEntry],
        top_k: int,
    ) -> List[MemoryEntry]:
        state_budget = max(2, min(3, top_k // 3 + 1))
        evidence_budget = max(1, top_k - state_budget)
        merged: List[MemoryEntry] = []
        seen_ids: Set[str] = set()

        for entry in state_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= state_budget:
                break

        evidence_count = 0
        for entry in evidence_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            evidence_count += 1
            if evidence_count >= evidence_budget:
                break

        if len(merged) >= top_k:
            return merged[:top_k]

        for source in (state_entries, evidence_entries):
            for entry in source:
                if entry.entry_id in seen_ids:
                    continue
                seen_ids.add(entry.entry_id)
                merged.append(entry)
                if len(merged) >= top_k:
                    return merged
        return merged

    def _anchor_source_ids(self, entries: Iterable[MemoryEntry]) -> Set[str]:
        return {
            source_entry_id
            for entry in entries
            if (source_entry_id := entry.metadata.get("source_entry_id"))
        }

    def _anchor_scopes(self, entries: Iterable[MemoryEntry]) -> Set[str]:
        return {
            entry.scope
            for entry in entries
            if entry.scope and entry.scope != "default"
        }

    def _backbone_score(self, entry: MemoryEntry, entity: str, attribute: str) -> Tuple[float, ...]:
        exact_entity = 1.0 if self._entity_matches(entry.entity, entity) else 0.0
        exact_attribute = 1.0 if entry.attribute == attribute else 0.0
        scope_bonus = 1.0 if entry.scope == "default" else 0.5
        return (
            exact_entity + exact_attribute,
            entry.importance,
            float(entry.access_count),
            scope_bonus,
            float(entry.timestamp),
        )

    def _is_structured_fact(self, entry: MemoryEntry) -> bool:
        return entry.metadata.get("source_kind") == "structured_fact"

    def _memory_kind(self, entry: MemoryEntry) -> str:
        return entry.metadata.get("memory_kind", "state")
