from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Set

from memory_inference.consolidation.revision_types import RevisionOp
from memory_inference.types import MemoryEntry


class BaseConsolidator(ABC):
    """Offline LLM-call interface for consolidation and fact extraction.

    All methods run offline (between sessions), never during inference.
    Implementations should increment total_calls for cost tracking.
    """

    def __init__(self) -> None:
        self.total_calls: int = 0

    @abstractmethod
    def extract_facts(
        self, text: str, entity: str, session_id: str, timestamp: int
    ) -> List[MemoryEntry]:
        """Extract structured MemoryEntry objects from a raw text turn."""
        raise NotImplementedError

    @abstractmethod
    def classify_revision(
        self,
        new_entry: MemoryEntry,
        existing: Optional[MemoryEntry],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        """Classify the revision operation for new_entry relative to existing.

        Args:
            new_entry: The incoming candidate entry.
            existing: The current active entry for the same key, or None.
            prior_values: Set of values previously seen for this key (for REVERT detection).

        Returns:
            The appropriate RevisionOp.
        """
        raise NotImplementedError
