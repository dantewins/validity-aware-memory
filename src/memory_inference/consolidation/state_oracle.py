from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from memory_inference.consolidation.revision_types import MemoryStatus
from memory_inference.types import MemoryEntry


class StateOracle:
    """Evaluates the validity state implied by a collection of MemoryEntry objects.

    Used as a gold-standard evaluator: given a stream of entries annotated with
    status/scope/supersedes_id, answers questions like "what is currently active?"
    and "are there unresolved conflicts?".
    """

    def __init__(self, entries: Sequence[MemoryEntry]) -> None:
        self._entries = list(entries)

    # ------------------------------------------------------------------ #
    # Core queries                                                         #
    # ------------------------------------------------------------------ #

    def active_value(self, entity: str, attribute: str) -> Optional[MemoryEntry]:
        """Return the most-recent ACTIVE entry for (entity, attribute), or None."""
        candidates = [
            e for e in self._entries
            if e.entity == entity and e.attribute == attribute
            and e.status in (MemoryStatus.ACTIVE, MemoryStatus.REINFORCED)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.timestamp)

    def superseded_chain(self, entity: str, attribute: str) -> List[MemoryEntry]:
        """Return all SUPERSEDED entries for (entity, attribute)."""
        return [
            e for e in self._entries
            if e.entity == entity and e.attribute == attribute
            and e.status == MemoryStatus.SUPERSEDED
        ]

    def unresolved_conflicts(self, entity: str, attribute: str) -> List[MemoryEntry]:
        """Return all CONFLICTED entries for (entity, attribute)."""
        return [
            e for e in self._entries
            if e.entity == entity and e.attribute == attribute
            and e.status == MemoryStatus.CONFLICTED
        ]

    def scope_splits(self, entity: str, attribute: str) -> Dict[str, List[MemoryEntry]]:
        """Return active entries grouped by scope for (entity, attribute).

        Entries with status ARCHIVED or SUPERSEDED are excluded.
        """
        active_statuses = {MemoryStatus.ACTIVE, MemoryStatus.REINFORCED}
        groups: Dict[str, List[MemoryEntry]] = defaultdict(list)
        for e in self._entries:
            if e.entity == entity and e.attribute == attribute and e.status in active_statuses:
                groups[e.scope].append(e)
        return dict(groups)

    def current_state_match(self, entity: str, attribute: str, gold_value: str) -> bool:
        """Check whether the active entry's value matches the provided gold value."""
        active = self.active_value(entity, attribute)
        if active is None:
            return False
        return active.value == gold_value
