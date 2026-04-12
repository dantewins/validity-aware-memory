from __future__ import annotations

from typing import Sequence

from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.llm.base import BaseReasoner
from memory_inference.metrics import ABSTAIN_TOKEN
from memory_inference.types import MemoryEntry, Query


class DeterministicValidityReader(BaseReasoner):
    """Deterministic reader that respects query mode and abstention."""

    def answer(self, query: Query, context: Sequence[MemoryEntry]) -> str:
        candidates = [
            entry
            for entry in context
            if entry.entity == query.entity and entry.attribute == query.attribute
        ]
        if not candidates:
            return "UNKNOWN"

        if query.query_mode == QueryMode.HISTORY:
            return min(candidates, key=lambda item: item.timestamp).value

        if query.supports_abstention or query.query_mode == QueryMode.CONFLICT_AWARE:
            latest_timestamp = max(entry.timestamp for entry in candidates)
            latest_values = {entry.value for entry in candidates if entry.timestamp == latest_timestamp}
            if len(latest_values) > 1:
                return ABSTAIN_TOKEN

        if query.query_mode == QueryMode.STATE_WITH_PROVENANCE:
            active_candidates = [
                entry for entry in candidates if entry.status.name in {"ACTIVE", "REINFORCED"}
            ]
            if active_candidates:
                return max(active_candidates, key=lambda item: item.timestamp).value

        return max(candidates, key=lambda item: item.timestamp).value
