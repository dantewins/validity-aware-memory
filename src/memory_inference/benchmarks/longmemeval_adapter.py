from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query


@dataclass(slots=True)
class LongMemEvalRecord:
    conversation_id: str
    updates: list[dict]
    query: dict


class LongMemEvalAdapter:
    """Adapter for cached LongMemEval-style structured records."""

    def from_records(self, records: Iterable[Mapping[str, object]]) -> List[BenchmarkBatch]:
        batches: List[BenchmarkBatch] = []
        for raw in records:
            record = LongMemEvalRecord(
                conversation_id=str(raw["conversation_id"]),
                updates=list(raw["updates"]),  # type: ignore[arg-type]
                query=dict(raw["query"]),  # type: ignore[arg-type]
            )
            batches.append(self._to_batch(record))
        return batches

    def from_json(self, path: str | Path) -> List[BenchmarkBatch]:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, list):
            raise ValueError("LongMemEval adapter expects a list of records")
        return self.from_records(payload)

    def _to_batch(self, record: LongMemEvalRecord) -> BenchmarkBatch:
        updates = [
            MemoryEntry(
                entry_id=str(update.get("entry_id", f"{record.conversation_id}-{index}")),
                entity=str(update["entity"]),
                attribute=str(update["relation"]),
                value=str(update["value"]),
                timestamp=int(update["timestamp"]),
                session_id=str(update.get("session_id", record.conversation_id)),
                confidence=float(update.get("confidence", 1.0)),
                scope=str(update.get("scope", "default")),
                provenance=str(update.get("provenance", "")),
            )
            for index, update in enumerate(record.updates)
        ]
        query_payload = record.query
        query = Query(
            query_id=str(query_payload.get("query_id", f"{record.conversation_id}-q")),
            entity=str(query_payload["entity"]),
            attribute=str(query_payload["relation"]),
            question=str(query_payload["question"]),
            answer=str(query_payload["answer"]),
            timestamp=int(query_payload["timestamp"]),
            session_id=str(query_payload.get("session_id", record.conversation_id)),
            query_mode=_parse_query_mode(query_payload.get("query_mode")),
            supports_abstention=bool(query_payload.get("supports_abstention", False)),
        )
        return BenchmarkBatch(
            session_id=record.conversation_id,
            updates=updates,
            queries=[query],
        )


def _parse_query_mode(raw: object) -> QueryMode:
    if raw is None:
        return QueryMode.CURRENT_STATE
    if isinstance(raw, QueryMode):
        return raw
    return QueryMode[str(raw)]
