from __future__ import annotations

from typing import Sequence

from memory_inference.llm.base import BaseReasoner
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.types import MemoryEntry, Query


class FixedPromptReader(BaseReasoner):
    """Stable reader stub representing a frozen prompt-based LLM policy."""

    def __init__(self, prompt_template: str = "Answer from the provided memory context.") -> None:
        self.prompt_template = prompt_template
        self._fallback = DeterministicValidityReader()

    def answer(self, query: Query, context: Sequence[MemoryEntry]) -> str:
        # The prompt is held fixed across experiments; the answer heuristic is deterministic
        # so the memory layer remains the manipulated variable in this scaffold.
        return self._fallback.answer(query, context)
