from __future__ import annotations

from typing import Iterable

from memory_inference.domain.enums import QueryMode
from memory_inference.domain.memory import MemoryRecord, RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.memory.policies.mem0 import Mem0Policy
from memory_inference.memory.policies.odv2 import ODV2Policy
from memory_inference.memory.retrieval.semantic import DenseEncoder


class Mem0ValidityGuardPolicy(BaseMemoryPolicy):
    """Mem0 retrieval guarded by an ODV2 validity ledger.

    Mem0 supplies broad semantic recall. ODV2 supplies explicit validity state:
    current facts, archived superseded facts, and unresolved conflicts. The guard
    prevents current-state queries from being answered from stale state support
    while keeping Mem0's high-recall behavior for history/open-ended queries.
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
        self.retriever = Mem0Policy(
            name=f"{name}::mem0",
            encoder=encoder,
            write_top_k=write_top_k,
            history_enabled=True,
            archive_conflict_enabled=True,
        )
        self.validity = ODV2Policy(
            name=f"{name}::validity",
            consolidator=consolidator,
            importance_threshold=importance_threshold,
            support_history_limit=support_history_limit,
        )

    @property
    def current_state(self):
        return self.validity.current_state

    @property
    def archive(self):
        return self.validity.archive

    @property
    def conflict_table(self):
        return self.validity.conflict_table

    def ingest(self, updates: Iterable[MemoryRecord]) -> None:
        update_list = list(updates)
        self.retriever.ingest(update_list)
        self.validity.ingest(update_list)

    def maybe_consolidate(self) -> None:
        self.retriever.maybe_consolidate()
        self.validity.maybe_consolidate()
        self.maintenance_tokens = self.retriever.maintenance_tokens + self.validity.maintenance_tokens
        self.maintenance_latency_ms = (
            self.retriever.maintenance_latency_ms + self.validity.maintenance_latency_ms
        )
        self.maintenance_calls = self.retriever.maintenance_calls + self.validity.maintenance_calls

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalBundle:
        query = RuntimeQuery(
            query_id=f"{self.name}-retrieve",
            context_id=f"{self.name}-retrieve",
            entity=entity,
            attribute=attribute,
            question=f"What is the current value of {attribute} for {entity}?",
            timestamp=0,
            session_id=f"{self.name}-retrieve",
        )
        return self.retrieve_for_query(query, top_k=top_k)

    def retrieve_for_query(self, query: RuntimeQuery, top_k: int = 5) -> RetrievalBundle:
        retrieval_limit = max(top_k, 8)
        base = self.retriever.retrieve_for_query(query, top_k=retrieval_limit)
        current_entries = self._current_entries(query)
        archive_entries = self._archive_entries(query)
        conflict_entries = self._conflict_entries(query)

        if query.query_mode == QueryMode.HISTORY:
            records = self._dedupe(list(base.records) + current_entries + archive_entries)
            return self._bundle(
                records,
                base=base,
                retrieval_mode="mem0_validity_history",
                conflicts=conflict_entries,
                archive=archive_entries,
            )

        if query.query_mode == QueryMode.CONFLICT_AWARE and conflict_entries:
            records = self._dedupe(
                conflict_entries
                + current_entries
                + self._filter_stale_current_state_records(
                    list(base.records),
                    current_entries=current_entries,
                    archive_entries=archive_entries,
                )
            )
            return self._bundle(
                records,
                base=base,
                retrieval_mode="mem0_validity_conflict_guard",
                conflicts=conflict_entries,
                archive=archive_entries,
            )

        filtered_base = self._filter_stale_current_state_records(
            list(base.records),
            current_entries=current_entries,
            archive_entries=archive_entries,
        )
        records = current_entries + filtered_base
        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            records.extend(archive_entries[:2])
        return self._bundle(
            self._dedupe(records),
            base=base,
            retrieval_mode="mem0_validity_current_guard",
            conflicts=conflict_entries,
            archive=archive_entries,
        )

    def snapshot_size(self) -> int:
        return self.retriever.snapshot_size() + self.validity.snapshot_size()

    def _filter_stale_current_state_records(
        self,
        records: list[MemoryRecord],
        *,
        current_entries: list[MemoryRecord],
        archive_entries: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if not current_entries or not archive_entries:
            return records

        current_entry_ids = {entry.entry_id for entry in current_entries}
        current_values = {self._normalized_value(entry) for entry in current_entries}
        current_source_ids = self._source_ids(current_entries)
        archived_source_ids = self._source_ids(archive_entries) - current_source_ids
        filtered: list[MemoryRecord] = []
        for record in records:
            if record.entry_id in archived_source_ids:
                continue
            if self._is_state_record(record):
                if record.entry_id not in current_entry_ids:
                    continue
                if self._normalized_value(record) not in current_values:
                    continue
            filtered.append(record)
        return filtered

    def _current_entries(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self.validity.current_entries_for_query(query)

    def _archive_entries(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self.validity.archive_entries_for_query(query)

    def _conflict_entries(self, query: RuntimeQuery) -> list[MemoryRecord]:
        return self.validity.conflict_entries_for_query(query)

    def _bundle(
        self,
        records: list[MemoryRecord],
        *,
        base: RetrievalBundle,
        retrieval_mode: str,
        conflicts: list[MemoryRecord],
        archive: list[MemoryRecord],
    ) -> RetrievalBundle:
        return RetrievalBundle(
            records=records,
            debug={
                **base.debug,
                "policy": self.name,
                "retrieval_mode": retrieval_mode,
                "base_retrieval_mode": base.debug.get("retrieval_mode", ""),
                "validity_conflicts": str(len(conflicts)),
                "validity_archive": str(len(archive)),
            },
        )

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
    def _normalized_value(entry: MemoryRecord) -> str:
        return " ".join(entry.value.lower().split())

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
