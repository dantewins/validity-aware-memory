from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.llm.consolidator_base import BaseConsolidator
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query


@dataclass(slots=True)
class LoCoMoCacheEntry:
    dialogue_id: str
    updates: list[dict]
    query: dict


class LoCoMoAdapter:
    """Two-stage LoCoMo-style adapter with optional JSON caching."""

    def __init__(self, consolidator: BaseConsolidator, cache_path: str | Path | None = None) -> None:
        self.consolidator = consolidator
        self.cache_path = Path(cache_path) if cache_path is not None else None

    def from_records(self, records: Iterable[Mapping[str, object]]) -> List[BenchmarkBatch]:
        entries = [
            LoCoMoCacheEntry(
                dialogue_id=str(record["dialogue_id"]),
                updates=list(record["updates"]),  # type: ignore[arg-type]
                query=dict(record["query"]),  # type: ignore[arg-type]
            )
            for record in records
        ]
        if self.cache_path is not None:
            self.cache_path.write_text(json.dumps([asdict(entry) for entry in entries]))
        return [self._to_batch(entry) for entry in entries]

    def from_cache(self) -> List[BenchmarkBatch]:
        if self.cache_path is None or not self.cache_path.exists():
            raise FileNotFoundError("LoCoMo cache is not available")
        payload = json.loads(self.cache_path.read_text())
        return [self._to_batch(LoCoMoCacheEntry(**row)) for row in payload]

    def _to_batch(self, entry: LoCoMoCacheEntry) -> BenchmarkBatch:
        updates = [
            MemoryEntry(
                entry_id=str(update.get("entry_id", f"{entry.dialogue_id}-{index}")),
                entity=str(update["entity"]),
                attribute=str(update["relation"]),
                value=str(update["value"]),
                timestamp=int(update["timestamp"]),
                session_id=str(update.get("session_id", entry.dialogue_id)),
                confidence=float(update.get("confidence", 1.0)),
                scope=str(update.get("scope", "default")),
                provenance=str(update.get("provenance", "locomo")),
            )
            for index, update in enumerate(entry.updates)
        ]
        query_payload = entry.query
        query = Query(
            query_id=str(query_payload.get("query_id", f"{entry.dialogue_id}-q")),
            entity=str(query_payload["entity"]),
            attribute=str(query_payload["relation"]),
            question=str(query_payload["question"]),
            answer=str(query_payload["answer"]),
            timestamp=int(query_payload["timestamp"]),
            session_id=str(query_payload.get("session_id", entry.dialogue_id)),
            query_mode=_parse_query_mode(query_payload.get("query_mode")),
            supports_abstention=bool(query_payload.get("supports_abstention", False)),
        )
        return BenchmarkBatch(session_id=entry.dialogue_id, updates=updates, queries=[query])


def _parse_query_mode(raw: object) -> QueryMode:
    if raw is None:
        return QueryMode.CURRENT_STATE
    if isinstance(raw, QueryMode):
        return raw
    return QueryMode[str(raw)]
