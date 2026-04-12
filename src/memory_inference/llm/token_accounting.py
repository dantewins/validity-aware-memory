from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SupportsEncode(Protocol):
    def encode(self, text: str, *args, **kwargs) -> list[int]: ...


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def count_tokens(text: str, tokenizer: SupportsEncode | None = None) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))
