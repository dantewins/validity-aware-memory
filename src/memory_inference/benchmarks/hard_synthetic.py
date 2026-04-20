from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from memory_inference.types import BenchmarkBatch, MemoryEntry, Query


@dataclass(slots=True)
class HardBenchmarkConfig:
    entities: int = 10
    attributes: tuple[str, ...] = (
        "favorite_editor",
        "preferred_language",
        "home_city",
        "meeting_time",
    )
    distractor_count: int = 3        # S1: wrong-entity entries per query
    reversal_rate: float = 0.5       # S2: fraction of attributes that revert to prior value
    conflict_density: int = 2        # S3: conflicting same-timestamp updates per entity
    session_count: int = 10          # S6: sessions in long-gap scenario
    confidence_decay_steps: int = 3  # S7: steps of decreasing confidence


def build_s1_distractor_injection(config: HardBenchmarkConfig) -> List[BenchmarkBatch]:
    """S1: Inject distractor entries from wrong entities to confuse retrieval."""
    batches: List[BenchmarkBatch] = []
    timestamp = 1000

    for entity_index in range(config.entities):
        entity = f"user_{entity_index:02d}"
        session_id = f"s1_{entity_index:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []

        for attribute in config.attributes:
            correct_value = f"s1_correct_{attribute}_{entity_index}"
            updates.append(MemoryEntry(
                entry_id=f"s1-{entity}-{attribute}-correct",
                entity=entity, attribute=attribute, value=correct_value,
                timestamp=timestamp, session_id=session_id,
            ))
            for d in range(config.distractor_count):
                other = f"user_{(entity_index + d + 1) % config.entities:02d}"
                updates.append(MemoryEntry(
                    entry_id=f"s1-{entity}-{attribute}-dist-{d}",
                    entity=other, attribute=attribute,
                    value=f"s1_distractor_{attribute}_{d}",
                    timestamp=timestamp - 1, session_id=session_id,
                ))
            queries.append(Query(
                query_id=f"s1-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is the current {attribute} for {entity}?",
                answer=correct_value,
                timestamp=timestamp + 1, session_id=session_id,
            ))
            timestamp += 5

        batches.append(BenchmarkBatch(session_id=session_id, updates=updates, queries=queries))

    return batches


def build_s2_value_reversal(config: HardBenchmarkConfig) -> List[BenchmarkBatch]:
    """S2: v1 → v2 → v1 reversal. Policies that don't track reversions fail."""
    batches: List[BenchmarkBatch] = []
    timestamp = 2000
    reversal_count = int(len(config.attributes) * config.reversal_rate)
    reversal_attrs = set(config.attributes[:reversal_count])

    for entity_index in range(config.entities):
        entity = f"user_{entity_index:02d}"
        session_id = f"s2_{entity_index:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []

        for attribute in config.attributes:
            v1 = f"s2_v1_{attribute}_{entity_index}"
            v2 = f"s2_v2_{attribute}_{entity_index}"

            if attribute in reversal_attrs:
                updates.extend([
                    MemoryEntry(entry_id=f"s2-{entity}-{attribute}-1", entity=entity, attribute=attribute, value=v1, timestamp=timestamp, session_id=session_id),
                    MemoryEntry(entry_id=f"s2-{entity}-{attribute}-2", entity=entity, attribute=attribute, value=v2, timestamp=timestamp + 1, session_id=session_id),
                    MemoryEntry(entry_id=f"s2-{entity}-{attribute}-3", entity=entity, attribute=attribute, value=v1, timestamp=timestamp + 2, session_id=session_id),
                ])
                correct = v1
            else:
                updates.extend([
                    MemoryEntry(entry_id=f"s2-{entity}-{attribute}-1", entity=entity, attribute=attribute, value=v1, timestamp=timestamp, session_id=session_id),
                    MemoryEntry(entry_id=f"s2-{entity}-{attribute}-2", entity=entity, attribute=attribute, value=v2, timestamp=timestamp + 1, session_id=session_id),
                ])
                correct = v2

            queries.append(Query(
                query_id=f"s2-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is the current {attribute} for {entity}?",
                answer=correct,
                timestamp=timestamp + 3, session_id=session_id,
            ))
            timestamp += 6

        batches.append(BenchmarkBatch(session_id=session_id, updates=updates, queries=queries))

    return batches
