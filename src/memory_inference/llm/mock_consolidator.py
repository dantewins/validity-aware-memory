from __future__ import annotations

import uuid
from typing import List, Optional, Set

from memory_inference.consolidation.revision_types import RevisionOp
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.types import MemoryEntry

_DEFAULT_LOW_CONF = 0.2


class MockConsolidator(BaseConsolidator):
    """Deterministic consolidator for unit tests. Makes no LLM API calls."""

    def __init__(self, low_confidence_threshold: float = _DEFAULT_LOW_CONF) -> None:
        super().__init__()
        self.low_confidence_threshold = low_confidence_threshold

    def extract_facts(
        self, text: str, entity: str, session_id: str, timestamp: int
    ) -> List[MemoryEntry]:
        """Parse semicolon-separated key=value pairs from text."""
        self.total_calls += 1
        facts: List[MemoryEntry] = []
        for part in text.split(";"):
            part = part.strip()
            if "=" in part:
                k, _, v = part.partition("=")
                facts.append(
                    MemoryEntry(
                        entry_id=str(uuid.uuid4()),
                        entity=entity,
                        attribute=k.strip(),
                        value=v.strip(),
                        timestamp=timestamp,
                        session_id=session_id,
                    )
                )
        return facts

    # ------------------------------------------------------------------ #
    # Validity-state interface                                             #
    # ------------------------------------------------------------------ #

    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        """Deterministic revision classification for unit tests.

        Decision priority:
        1. No existing → ADD
        2. Low confidence → LOW_CONFIDENCE
        3. Scopes differ → SPLIT_SCOPE
        4. Same value → REINFORCE
        5. Equal timestamps, different value → CONFLICT_UNRESOLVED
        6. New value was seen before (prior_values) → REVERT
        7. Newer timestamp, different value → REVISE
        """
        self.total_calls += 1

        if new_entry.confidence < self.low_confidence_threshold:
            return RevisionOp.LOW_CONFIDENCE

        if existing is None:
            return RevisionOp.ADD

        if new_entry.scope != existing.scope:
            return RevisionOp.SPLIT_SCOPE

        if new_entry.value == existing.value:
            return RevisionOp.REINFORCE

        if new_entry.timestamp == existing.timestamp:
            return RevisionOp.CONFLICT_UNRESOLVED

        if prior_values and new_entry.value in prior_values:
            return RevisionOp.REVERT

        return RevisionOp.REVISE
