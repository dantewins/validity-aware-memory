from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Set

from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.consolidation_types import UpdateType
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.types import MemoryEntry, MemoryKey, RetrievalResult


class OfflineDeltaConsolidationPolicy(BaseMemoryPolicy):
    """Offline Delta Consolidation (ODC).

    Online path  — ingest(): fast append, no LLM calls.
    Offline path — maybe_consolidate(): LLM-assisted, fires at session boundary.

    Four consolidation components:
    1. Session-boundary trigger (caller invokes maybe_consolidate at session end)
    2. LLM-assisted semantic conflict classification via BaseConsolidator
    3. Importance-weighted retention with archive tier
    4. Semantic deduplication/compression for reinforcement entries
    """

    def __init__(
        self,
        consolidator: BaseConsolidator,
        importance_threshold: float = 0.1,
        support_history_limit: int = 2,
    ) -> None:
        super().__init__(name="offline_delta_consolidation")
        self.consolidator = consolidator
        self.importance_threshold = importance_threshold
        self.support_history_limit = support_history_limit

        self.episodic_log: List[MemoryEntry] = []
        self.current: Dict[MemoryKey, MemoryEntry] = {}
        self.archived: DefaultDict[MemoryKey, List[MemoryEntry]] = defaultdict(list)
        self.history: DefaultDict[MemoryKey, List[MemoryEntry]] = defaultdict(list)
        self.superseded_ids: Set[str] = set()
        self.conflict_flags: Dict[str, str] = {}  # entry_id → conflicting entry_id
        self._pending: List[MemoryEntry] = []

    def ingest(self, updates: Iterable[MemoryEntry]) -> None:
        """Fast online path: append to episodic log and pending buffer. No LLM calls."""
        for update in updates:
            self.episodic_log.append(update)
            self.history[update.key].append(update)
            self._pending.append(update)

    def maybe_consolidate(self) -> None:
        """Offline consolidation: classify and process all pending entries."""
        if not self._pending:
            return

        for entry in self._pending:
            existing = self.current.get(entry.key)
            if existing is None:
                self.current[entry.key] = entry
                continue

            update_type = self.consolidator.classify_update(entry, existing)

            if update_type == UpdateType.NEW:
                self.current[entry.key] = entry

            elif update_type == UpdateType.REINFORCEMENT:
                merged = self.consolidator.merge_entries([existing, entry])
                self.superseded_ids.add(existing.entry_id)
                self.superseded_ids.add(entry.entry_id)
                self.current[entry.key] = merged
                self.episodic_log.append(merged)

            elif update_type == UpdateType.SUPERSESSION:
                self.superseded_ids.add(existing.entry_id)
                archived = dataclasses.replace(
                    existing, importance=max(0.0, existing.importance - 0.2)
                )
                self.archived[entry.key].append(archived)
                self.current[entry.key] = entry

            elif update_type == UpdateType.CONFLICT:
                self.conflict_flags[existing.entry_id] = entry.entry_id
                self.conflict_flags[entry.entry_id] = existing.entry_id
                if entry.timestamp >= existing.timestamp:
                    self.current[entry.key] = entry

        self._pending.clear()
        self._apply_importance_threshold()

    def _apply_importance_threshold(self) -> None:
        to_archive = [
            key for key, entry in self.current.items()
            if entry.importance < self.importance_threshold
        ]
        for key in to_archive:
            self.archived[key].append(self.current.pop(key))

    def retrieve(self, entity: str, attribute: str, top_k: int = 5) -> RetrievalResult:
        key = (entity, attribute)
        current = self.current.get(key)
        history = self.history.get(key, [])

        active = [
            e for e in sorted(history, key=lambda x: x.timestamp, reverse=True)
            if e.entry_id not in self.superseded_ids
        ]

        entries: List[MemoryEntry] = []
        if current is not None:
            entries.append(current)
        for e in active:
            if current is None or e.entry_id != current.entry_id:
                entries.append(e)
            if len(entries) >= min(top_k, 1 + self.support_history_limit):
                break

        if not entries:
            archived = sorted(
                self.archived.get(key, []), key=lambda x: x.timestamp, reverse=True
            )
            entries = archived[:top_k]

        return RetrievalResult(
            entries=entries,
            debug={
                "policy": self.name,
                "conflicts": str(len(self.conflict_flags) // 2),
                "snapshot_size": str(self.snapshot_size()),
            },
        )

    def snapshot_size(self) -> int:
        return len(self.current) + sum(len(v) for v in self.archived.values())
