from __future__ import annotations

import json
from typing import Sequence

from memory_inference.llm.extractor_base import BaseExtractor
from memory_inference.llm.local_hf_reasoner import LocalHFReasoner
from memory_inference.types import MemoryEntry, Query


class LocalHFExtractor(BaseExtractor):
    """Simple local-model extractor that expects JSON lines from the model."""

    def __init__(self, reasoner: LocalHFReasoner) -> None:
        self.reasoner = reasoner

    def extract(
        self,
        text: str,
        *,
        entity: str,
        session_id: str,
        timestamp: int,
    ) -> Sequence[MemoryEntry]:
        query = Query(
            query_id=f"extract-{session_id}-{timestamp}",
            entity=entity,
            attribute="memory_facts",
            question=(
                "Extract current factual memory candidates as a JSON array with fields "
                "`relation`, `value`, `scope`, and optional `confidence`."
            ),
            answer="",
            timestamp=timestamp,
            session_id=session_id,
        )
        synthetic_context = [
            MemoryEntry(
                entry_id=f"source-{session_id}-{timestamp}",
                entity=entity,
                attribute="source_text",
                value=text,
                timestamp=timestamp,
                session_id=session_id,
            )
        ]
        trace = self.reasoner.answer_with_trace(query, synthetic_context)
        try:
            payload = json.loads(trace.answer)
        except json.JSONDecodeError:
            raw_candidate = trace.raw_output
            if trace.prompt and raw_candidate.startswith(trace.prompt):
                raw_candidate = raw_candidate[len(trace.prompt):]
            try:
                payload = json.loads(raw_candidate.strip())
            except json.JSONDecodeError:
                return []
        if not isinstance(payload, list):
            return []
        entries: list[MemoryEntry] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            relation = item.get("relation")
            value = item.get("value")
            if relation is None or value is None:
                continue
            entries.append(
                MemoryEntry(
                    entry_id=f"extract-{session_id}-{timestamp}-{index}",
                    entity=entity,
                    attribute=str(relation),
                    value=str(value),
                    timestamp=timestamp,
                    session_id=session_id,
                    scope=str(item.get("scope", "default")),
                    confidence=float(item.get("confidence", 0.5)),
                    provenance="local-extractor",
                )
            )
        return entries
