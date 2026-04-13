from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, Sequence

from memory_inference.types import MemoryEntry, Query

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VECTOR_DIM = 48
_HASH_BYTES = 64


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def query_search_text(query: Query) -> str:
    extra_attributes = " ".join(query.multi_attributes)
    return " ".join(
        part
        for part in (
            query.entity,
            query.attribute,
            extra_attributes,
            query.question,
        )
        if part
    )


def entry_search_text(entry: MemoryEntry) -> str:
    metadata_parts = [
        f"{key}={value}"
        for key, value in sorted(entry.metadata.items())
        if value
    ]
    return " ".join(
        part
        for part in (
            entry.entity,
            entry.attribute,
            entry.value,
            entry.provenance,
            " ".join(metadata_parts),
        )
        if part
    )


class HashedSemanticEncoder:
    """Dependency-free dense encoder based on deterministic feature hashing."""

    def __init__(self, dim: int = _VECTOR_DIM) -> None:
        self.dim = dim
        self._cache: dict[str, tuple[float, ...]] = {}

    def encode(self, text: str) -> tuple[float, ...]:
        normalized = normalize_text(text)
        if normalized in self._cache:
            return self._cache[normalized]

        vector = [0.0] * self.dim
        features = _feature_weights(normalized)
        for feature, weight in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=_HASH_BYTES).digest()
            for idx in range(self.dim):
                raw = digest[idx % len(digest)]
                scale = digest[(idx * 7 + 11) % len(digest)]
                component = ((raw / 255.0) * 2.0 - 1.0) * (0.5 + scale / 510.0)
                vector[idx] += weight * component

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            encoded = tuple(value / norm for value in vector)
        else:
            encoded = tuple(0.0 for _ in range(self.dim))
        self._cache[normalized] = encoded
        return encoded

    def similarity(self, left: str | Sequence[float], right: str | Sequence[float]) -> float:
        left_vector = self.encode(left) if isinstance(left, str) else left
        right_vector = self.encode(right) if isinstance(right, str) else right
        return sum(left_value * right_value for left_value, right_value in zip(left_vector, right_vector))


def _feature_weights(text: str) -> Counter[str]:
    tokens = _TOKEN_RE.findall(text)
    features: Counter[str] = Counter()
    for token in tokens:
        features[f"tok:{token}"] += 1.0
        if len(token) >= 4:
            for idx in range(len(token) - 2):
                features[f"tri:{token[idx:idx + 3]}"] += 0.35
    for first, second in zip(tokens, tokens[1:]):
        features[f"bg:{first}_{second}"] += 0.6
    return features
