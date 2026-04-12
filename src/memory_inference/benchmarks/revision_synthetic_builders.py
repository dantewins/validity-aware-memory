"""Internal builders for revision_synthetic scenarios.

Each builder produces a list of RevisionScenario objects — one per entity.
Gold state is constructed by annotating MemoryEntry objects with the correct
MemoryStatus before passing them to StateOracle.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List

from memory_inference.consolidation.revision_types import MemoryStatus, QueryMode
from memory_inference.consolidation.state_oracle import StateOracle
from memory_inference.types import BenchmarkBatch, MemoryEntry, Query

if TYPE_CHECKING:
    from memory_inference.benchmarks.revision_synthetic import (
        RevisionBenchmarkConfig,
        RevisionScenario,
    )

_TS_BASE = {
    "S1": 10000,
    "S2": 20000,
    "S3": 30000,
    "S4": 40000,
    "S5": 50000,
    "S6": 60000,
    "S7": 70000,
}


def _e(entry_id: str, entity: str, attribute: str, value: str, ts: int,
       session_id: str, status: MemoryStatus = MemoryStatus.ACTIVE,
       scope: str = "default",
       supersedes_id: str | None = None) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id, entity=entity, attribute=attribute,
        value=value, timestamp=ts, session_id=session_id,
        status=status, scope=scope, supersedes_id=supersedes_id,
    )


def _build_s1_distractors(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S1: Inject distractor entries from other entities. Gold = correct entity's entry."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S1"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s1_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            correct_val = f"s1_correct_{attribute}_{ei}"
            # Correct entry — ACTIVE in gold
            correct = _e(f"s1-{entity}-{attribute}-c", entity, attribute,
                         correct_val, ts, session_id, status=MemoryStatus.ACTIVE)
            updates.append(correct)
            gold_entries.append(correct)

            # Distractors — not in gold (wrong entity), added to batch updates only
            n_dist = max(1, int(len(config.attributes) * config.distractor_rate))
            for d in range(n_dist):
                other = f"user_{(ei + d + 1) % config.entities:02d}"
                dist = _e(f"s1-{entity}-{attribute}-d{d}", other, attribute,
                          f"s1_dist_{d}", ts - 1, session_id)
                updates.append(dist)

            queries.append(Query(
                query_id=f"s1-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is {attribute} for {entity}?",
                answer=correct_val, timestamp=ts + 1, session_id=session_id,
            ))
            ts += 5

        scenarios.append(RevisionScenario(
            scenario_id=f"S1_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s2_revise(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S2: v1 → v2. Gold active = v2, v1 superseded."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S2"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s2_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            v1 = f"s2_v1_{attribute}_{ei}"
            v2 = f"s2_v2_{attribute}_{ei}"

            old = _e(f"s2-{entity}-{attribute}-1", entity, attribute, v1, ts, session_id,
                     status=MemoryStatus.SUPERSEDED)
            new = _e(f"s2-{entity}-{attribute}-2", entity, attribute, v2, ts + 1, session_id,
                     status=MemoryStatus.ACTIVE, supersedes_id=old.entry_id)

            updates.extend([
                dataclasses.replace(old, status=MemoryStatus.ACTIVE),   # raw stream: v1 arrives first
                dataclasses.replace(new, status=MemoryStatus.ACTIVE),   # then v2
            ])
            gold_entries.extend([old, new])

            queries.append(Query(
                query_id=f"s2-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is {attribute} for {entity}?",
                answer=v2, timestamp=ts + 2, session_id=session_id,
            ))
            ts += 6

        scenarios.append(RevisionScenario(
            scenario_id=f"S2_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s3_revert(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S3: v1 → v2 → v1 (reversion). Gold active = v1 (second time), v2 superseded."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S3"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s3_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            v1 = f"s3_v1_{attribute}_{ei}"
            v2 = f"s3_v2_{attribute}_{ei}"

            e1 = _e(f"s3-{entity}-{attribute}-1", entity, attribute, v1, ts, session_id)
            e2 = _e(f"s3-{entity}-{attribute}-2", entity, attribute, v2, ts + 1, session_id)
            e3 = _e(f"s3-{entity}-{attribute}-3", entity, attribute, v1, ts + 2, session_id)

            updates.extend([
                dataclasses.replace(e1, status=MemoryStatus.ACTIVE),
                dataclasses.replace(e2, status=MemoryStatus.ACTIVE),
                dataclasses.replace(e3, status=MemoryStatus.ACTIVE),
            ])

            # Gold: e1 and e2 superseded, e3 active
            gold_entries.extend([
                dataclasses.replace(e1, status=MemoryStatus.SUPERSEDED),
                dataclasses.replace(e2, status=MemoryStatus.SUPERSEDED),
                dataclasses.replace(e3, status=MemoryStatus.ACTIVE),
            ])

            queries.append(Query(
                query_id=f"s3-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is {attribute} for {entity}?",
                answer=v1, timestamp=ts + 3, session_id=session_id,
            ))
            ts += 8

        scenarios.append(RevisionScenario(
            scenario_id=f"S3_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s4_conflict(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S4: same timestamp, two different values — unresolvable conflict."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S4"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s4_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            v_a = f"s4_va_{attribute}_{ei}"
            v_b = f"s4_vb_{attribute}_{ei}"

            ca = _e(f"s4-{entity}-{attribute}-a", entity, attribute, v_a, ts, session_id,
                    status=MemoryStatus.CONFLICTED)
            cb = _e(f"s4-{entity}-{attribute}-b", entity, attribute, v_b, ts, session_id,
                    status=MemoryStatus.CONFLICTED)

            updates.extend([
                dataclasses.replace(ca, status=MemoryStatus.ACTIVE),
                dataclasses.replace(cb, status=MemoryStatus.ACTIVE),
            ])
            gold_entries.extend([ca, cb])

            queries.append(Query(
                query_id=f"s4-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is {attribute} for {entity}?",
                answer=v_a,   # reference answer (first ingested)
                timestamp=ts + 1, session_id=session_id,
                supports_abstention=True,
                query_mode=QueryMode.CONFLICT_AWARE,
            ))
            ts += 5

        scenarios.append(RevisionScenario(
            scenario_id=f"S4_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s5_scope_split(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S5: same key, two different scopes — both entries remain ACTIVE."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    if len(config.attributes) < 1:
        return []

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S5"]
    scopes = ("boston", "miami")

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s5_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            entries_for_attr: List[MemoryEntry] = []
            for sc in scopes:
                val = f"s5_{attribute}_{ei}_{sc}"
                e = _e(f"s5-{entity}-{attribute}-{sc}", entity, attribute, val, ts, session_id,
                       status=MemoryStatus.ACTIVE, scope=sc)
                updates.append(e)
                gold_entries.append(e)
                entries_for_attr.append(e)
                ts += 1

            # Query asks about the "default" scope — answer = first scope's value for reference
            queries.append(Query(
                query_id=f"s5-q-{entity}-{attribute}",
                entity=entity, attribute=attribute,
                question=f"What is {attribute} for {entity}?",
                answer=entries_for_attr[0].value,
                timestamp=ts + 1, session_id=session_id,
            ))
            ts += 4

        scenarios.append(RevisionScenario(
            scenario_id=f"S5_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s6_long_gap_partial_update(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S6: one attribute updates after a long gap; others remain unchanged."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S6"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s6_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        revised_attribute = config.attributes[0]
        for attribute in config.attributes:
            old = _e(f"s6-{entity}-{attribute}-old", entity, attribute, f"s6_old_{attribute}_{ei}", ts, session_id)
            updates.append(old)
            if attribute == revised_attribute:
                new = _e(
                    f"s6-{entity}-{attribute}-new",
                    entity,
                    attribute,
                    f"s6_new_{attribute}_{ei}",
                    ts + config.long_gap_delta,
                    session_id,
                )
                updates.append(new)
                gold_entries.extend([
                    dataclasses.replace(old, status=MemoryStatus.SUPERSEDED),
                    dataclasses.replace(new, status=MemoryStatus.ACTIVE, supersedes_id=old.entry_id),
                ])
                answer = new.value
            else:
                gold_entries.append(dataclasses.replace(old, status=MemoryStatus.ACTIVE))
                answer = old.value

            queries.append(Query(
                query_id=f"s6-q-{entity}-{attribute}",
                entity=entity,
                attribute=attribute,
                question=f"What is the current {attribute} for {entity} after later sessions?",
                answer=answer,
                timestamp=ts + config.long_gap_delta + 1,
                session_id=session_id,
            ))
            ts += 7

        scenarios.append(RevisionScenario(
            scenario_id=f"S6_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios


def _build_s7_alias_noise(config: "RevisionBenchmarkConfig") -> List["RevisionScenario"]:
    """S7: low-confidence alias/noise competes with a later canonical value."""
    from memory_inference.benchmarks.revision_synthetic import RevisionScenario

    scenarios: List[RevisionScenario] = []
    ts = _TS_BASE["S7"]

    for ei in range(config.entities):
        entity = f"user_{ei:02d}"
        session_id = f"s7_{ei:02d}"
        updates: List[MemoryEntry] = []
        queries: List[Query] = []
        gold_entries: List[MemoryEntry] = []

        for attribute in config.attributes:
            canonical = _e(
                f"s7-{entity}-{attribute}-canonical",
                entity,
                attribute,
                f"s7_canonical_{attribute}_{ei}",
                ts + 1,
                session_id,
            )
            updates.append(canonical)
            gold_entries.append(dataclasses.replace(canonical, status=MemoryStatus.ACTIVE))

            if config.noise_rate > 0:
                noisy = MemoryEntry(
                    entry_id=f"s7-{entity}-{attribute}-noise",
                    entity=entity,
                    attribute=attribute,
                    value=f"s7_alias_{attribute}_{ei}",
                    timestamp=ts,
                    session_id=session_id,
                    confidence=0.1,
                    status=MemoryStatus.ARCHIVED,
                    provenance="alias-noise",
                )
                updates.insert(0, dataclasses.replace(noisy, status=MemoryStatus.ACTIVE))
                gold_entries.append(noisy)

            queries.append(Query(
                query_id=f"s7-q-{entity}-{attribute}",
                entity=entity,
                attribute=attribute,
                question=f"What is the canonical current {attribute} for {entity}?",
                answer=canonical.value,
                timestamp=ts + 2,
                session_id=session_id,
            ))
            ts += 6

        scenarios.append(RevisionScenario(
            scenario_id=f"S7_{entity}",
            batch=BenchmarkBatch(session_id=session_id, updates=updates, queries=queries),
            gold_state=StateOracle(gold_entries),
        ))

    return scenarios
