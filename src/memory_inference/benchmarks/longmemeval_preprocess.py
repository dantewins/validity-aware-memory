from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, List

from memory_inference.benchmarks.longmemeval_adapter import LongMemEvalAdapter
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query


def preprocess_longmemeval(source_path: str | Path, output_path: str | Path) -> list[BenchmarkBatch]:
    adapter = LongMemEvalAdapter()
    batches = adapter.from_json(source_path)
    serialized = [
        {
            "session_id": batch.session_id,
            "updates": [_json_ready(asdict(update)) for update in batch.updates],
            "queries": [_json_ready(asdict(query)) for query in batch.queries],
        }
        for batch in batches
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(serialized, indent=2))
    return batches


def load_preprocessed_longmemeval(path: str | Path) -> list[BenchmarkBatch]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("Expected a list of preprocessed LongMemEval batches")
    batches: list[BenchmarkBatch] = []
    for row in payload:
        updates = [MemoryEntry(**_restore_memory_entry(update)) for update in row["updates"]]
        queries = [Query(**_restore_query(query)) for query in row["queries"]]
        batches.append(BenchmarkBatch(session_id=row["session_id"], updates=updates, queries=queries))
    return batches


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _restore_query(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    if "query_mode" in restored and isinstance(restored["query_mode"], str):
        restored["query_mode"] = QueryMode[restored["query_mode"]]
    if "multi_attributes" in restored and isinstance(restored["multi_attributes"], list):
        restored["multi_attributes"] = tuple(restored["multi_attributes"])
    return restored


def _restore_memory_entry(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    if "status" in restored and isinstance(restored["status"], str):
        restored["status"] = MemoryStatus[restored["status"]]
    return restored
