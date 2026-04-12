from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Sequence

from memory_inference.types import MemoryEntry, Query


@dataclass(slots=True)
class ReasonerTrace:
    answer: str
    model_id: str = "unknown"
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    raw_output: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseReasoner(ABC):
    """Frozen inference interface.

    A real implementation can call an API or local model. For now we keep the base
    model logically frozen and only vary the memory policy.
    """

    @abstractmethod
    def answer(self, query: Query, context: Sequence[MemoryEntry]) -> str:
        raise NotImplementedError

    def answer_with_trace(self, query: Query, context: Sequence[MemoryEntry]) -> ReasonerTrace:
        started = time.perf_counter()
        answer = self.answer(query, context)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ReasonerTrace(
            answer=answer,
            model_id=self.__class__.__name__,
            raw_output=answer,
            latency_ms=latency_ms,
        )
