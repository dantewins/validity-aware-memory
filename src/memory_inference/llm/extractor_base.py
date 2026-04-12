from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from memory_inference.types import MemoryEntry


class BaseExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        text: str,
        *,
        entity: str,
        session_id: str,
        timestamp: int,
    ) -> Sequence[MemoryEntry]:
        raise NotImplementedError
