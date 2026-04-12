from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from memory_inference.benchmarks.locomo_adapter import LoCoMoAdapter
from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.types import BenchmarkBatch


def preprocess_locomo(
    adapter: LoCoMoAdapter,
    records_path: str | Path,
    output_path: str | Path,
) -> list[BenchmarkBatch]:
    payload = json.loads(Path(records_path).read_text())
    if not isinstance(payload, list):
        raise ValueError("LoCoMo preprocessing expects a list of records")
    batches = adapter.from_records(payload)
    serialized = [
        {
            "session_id": batch.session_id,
            "updates": [_json_ready(asdict(update)) for update in batch.updates],
            "queries": [_json_ready(asdict(query)) for query in batch.queries],
        }
        for batch in batches
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serialized, indent=2))
    return batches


def load_preprocessed_locomo(path: str | Path) -> list[BenchmarkBatch]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("Expected a list of preprocessed LoCoMo batches")
    from memory_inference.types import BenchmarkBatch, MemoryEntry, Query

    batches: list[BenchmarkBatch] = []
    for row in payload:
        updates = []
        for update in row["updates"]:
            restored_update = dict(update)
            if "status" in restored_update and isinstance(restored_update["status"], str):
                restored_update["status"] = MemoryStatus[restored_update["status"]]
            updates.append(MemoryEntry(**restored_update))
        queries = []
        for query in row["queries"]:
            restored = dict(query)
            if "query_mode" in restored and isinstance(restored["query_mode"], str):
                restored["query_mode"] = QueryMode[restored["query_mode"]]
            if "multi_attributes" in restored and isinstance(restored["multi_attributes"], list):
                restored["multi_attributes"] = tuple(restored["multi_attributes"])
            queries.append(Query(**restored))
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
