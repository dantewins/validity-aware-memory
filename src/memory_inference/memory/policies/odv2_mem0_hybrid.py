from __future__ import annotations

from typing import Iterable

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.odv2 import ODV2Policy
from memory_inference.memory.retrieval import DenseEncoder, DenseRanker, expand_with_support_entries, is_open_ended_query


class ODV2Mem0HybridPolicy(BaseMemoryPolicy):
    """ODV2-first retrieval with dense episodic evidence support.

    Unlike Mem0ValidityGuardPolicy, this policy does not start from Mem0 retrieval.
    It treats ODV2's validity ledger as the primary state source, then supplements
    that state with dense evidence drawn from the full episodic log.
    """

    def __init__(
        self,
        *,
        name: str,
        consolidator: BaseConsolidator,
        encoder: DenseEncoder | None = None,
        write_top_k: int = 10,
        importance_threshold: float = 0.1,
        support_history_limit: int = 3,
    ) -> None:
        super().__init__(name=name)
        self.support_history_limit = support_history_limit
        self.validity = ODV2Policy(
            name=f"{name}::validity",
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=support_history_limit,
        )
        self.dense_ranker = DenseRanker(encoder=encoder, write_top_k=write_top_k)
        self.episodic_log: list[MemoryRecord] = []

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        update_list = list(updates)
        if not update_list:
            return
        self.episodic_log.extend(update_list)
        self.validity.ingest(update_list)
        for entry in update_list:
            self.dense_ranker.index(entry)

    def maybe_consolidate(self) -> None:
        self.validity.maybe_consolidate()
        self.maintenance_tokens = self.validity.maintenance_tokens
        self.maintenance_latency_ms = self.validity.maintenance_latency_ms
        self.maintenance_calls = self.validity.maintenance_calls

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id=f"{self.name}-retrieve",
            context_id=f"{self.name}-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=max((record.timestamp for record in self.episodic_log), default=0),
            session_id=f"{self.name}-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        limit = max(top_k, 8) if is_open_ended_query(query) else top_k
        if query.query_mode == QueryMode.HISTORY:
            return self._retrieve_history(query, top_k=limit)

        current_entries = self.validity.current_entries_for_query(query)
        archive_entries = self.validity.archive_entries_for_query(query)
        conflict_entries = self.validity.conflict_entries_for_query(query)
        anchor_entries = self._anchor_entries(
            query,
            current_entries=current_entries,
            archive_entries=archive_entries,
            conflict_entries=conflict_entries,
        )
        evidence_entries = self._dense_evidence(
            query,
            current_entries=current_entries,
            archive_entries=archive_entries,
            conflict_entries=conflict_entries,
            limit=max(limit * 3, 12),
        )
        merged = self._merge(anchor_entries, evidence_entries, top_k=limit)
        expanded = expand_with_support_entries(
            merged,
            self.episodic_log,
            support_limit=self.support_history_limit,
            max_entries=limit + self.support_history_limit,
        )
        retrieval_mode = (
            "odv2_mem0_hybrid_conflict"
            if query.query_mode == QueryMode.CONFLICT_AWARE
            else "odv2_mem0_hybrid_provenance"
            if query.query_mode == QueryMode.STATE_WITH_PROVENANCE
            else "odv2_mem0_hybrid_current"
        )
        return RetrievalBundle(
            records=expanded,
            debug={
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "anchor_count": str(len(anchor_entries)),
                "conflict_count": str(len(conflict_entries)),
            },
        )

    def snapshot_size(self) -> int:
        return self.validity.snapshot_size()

    def _retrieve_history(self, query: RuntimeQuery, *, top_k: int) -> RetrievalBundle:
        candidates = [
            entry
            for entry in self.episodic_log
            if self._entity_matches(entry.entity, query.entity)
            and entry.attribute == query.attribute
        ]
        ranked = self.dense_ranker.rank_query(
            query,
            candidates,
            entity_matches=self._entity_matches,
            history=True,
        )
        expanded = expand_with_support_entries(
            ranked[:top_k],
            self.episodic_log,
            support_limit=self.support_history_limit,
            max_entries=top_k + self.support_history_limit,
        )
        return RetrievalBundle(
            records=expanded,
            debug={
                "policy": self.name,
                "retrieval_mode": "odv2_mem0_hybrid_history",
            },
        )

    def _anchor_entries(
        self,
        query: RuntimeQuery,
        *,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
        conflict_entries: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if query.query_mode == QueryMode.CONFLICT_AWARE and conflict_entries:
            return self._dedupe(conflict_entries + current_entries[:2])
        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            return self._dedupe(current_entries + archive_entries[:2])
        if current_entries:
            return self._dedupe(current_entries)
        return self._dedupe(archive_entries[:2] + conflict_entries[:1])

    def _dense_evidence(
        self,
        query: RuntimeQuery,
        *,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
        conflict_entries: list[MemoryRecord],
        limit: int,
    ) -> list[MemoryRecord]:
        ranked = self.dense_ranker.rank_query(
            query,
            self.episodic_log,
            entity_matches=self._entity_matches,
        )
        if query.query_mode == QueryMode.CONFLICT_AWARE and conflict_entries:
            return ranked[:limit]
        return self._filter_stale_state_records(
            ranked[:limit],
            current_entries=current_entries,
            archive_entries=archive_entries,
        )

    def _filter_stale_state_records(
        self,
        records: list[MemoryRecord],
        *,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if not current_entries:
            return records
        current_entry_ids = {entry.entry_id for entry in current_entries}
        current_values = {self._normalized_value(entry.value) for entry in current_entries}
        current_source_ids = self._source_ids(current_entries)
        archived_source_ids = self._source_ids(archive_entries) - current_source_ids
        filtered: list[MemoryRecord] = []
        for record in records:
            if record.entry_id in archived_source_ids:
                continue
            if self._is_state_record(record):
                if record.entry_id not in current_entry_ids and self._normalized_value(record.value) not in current_values:
                    continue
            filtered.append(record)
        return filtered

    @staticmethod
    def _merge(
        anchor_entries: list[MemoryRecord],
        evidence_entries: list[MemoryRecord],
        *,
        top_k: int,
    ) -> list[MemoryRecord]:
        anchor_budget = max(2, min(4, top_k // 2 + 1))
        evidence_budget = max(1, top_k - anchor_budget)
        merged: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for entry in anchor_entries:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            merged.append(entry)
            if len(merged) >= anchor_budget:
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

        for source in (anchor_entries, evidence_entries):
            for entry in source:
                if entry.entry_id in seen_ids:
                    continue
                seen_ids.add(entry.entry_id)
                merged.append(entry)
                if len(merged) >= top_k:
                    return merged
        return merged

    @staticmethod
    def _entity_matches(entry_entity: str, query_entity: str) -> bool:
        return query_entity in {"conversation", "all"} or entry_entity == query_entity

    @staticmethod
    def _dedupe(records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        deduped: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.entry_id in seen_ids:
                continue
            seen_ids.add(record.entry_id)
            deduped.append(record)
        return deduped

    @staticmethod
    def _source_ids(entries: Iterable[MemoryRecord]) -> set[str]:
        return {
            entry.source_entry_id
            for entry in entries
            if entry.source_entry_id
        }

    @staticmethod
    def _is_state_record(entry: MemoryRecord) -> bool:
        return entry.memory_kind == "state" or entry.source_kind == "structured_fact"

    @staticmethod
    def _normalized_value(value: str) -> str:
        return " ".join(value.lower().split())
