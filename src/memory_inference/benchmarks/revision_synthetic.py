"""Revision-aware synthetic benchmark with gold state annotations.

Each RevisionScenario pairs a BenchmarkBatch (the input stream) with a
StateOracle (the gold validity state), letting evaluators check both QA
accuracy and state reconstruction quality.

Scenario families
-----------------
S1 — Distractor injection          (retrieval precision under entity-level noise)
S2 — Standard revise (v1 → v2)    (supersession)
S3 — Reversion (v1 → v2 → v1)     (REVERT op, stale-v2 trap)
S4 — Unresolved conflict           (equal-timestamp contradiction)
S5 — Scope split                   (two scopes coexist, no overwrite)
S6 — Long-gap partial update       (only some attributes change after a gap)
S7 — Extraction noise / aliasing   (low-confidence or aliased updates compete)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from memory_inference.benchmarks.revision_synthetic_builders import (
    _build_s1_distractors,
    _build_s2_revise,
    _build_s3_revert,
    _build_s4_conflict,
    _build_s5_scope_split,
    _build_s6_long_gap_partial_update,
    _build_s7_alias_noise,
)
from memory_inference.consolidation.state_oracle import StateOracle
from memory_inference.types import BenchmarkBatch


@dataclass(slots=True)
class RevisionBenchmarkConfig:
    entities: int = 10
    attributes: tuple[str, ...] = (
        "home_city",
        "preferred_language",
        "favorite_editor",
        "meeting_time",
    )
    distractor_rate: float = 0.5   # S1
    reversal_rate: float = 1.0     # S3
    conflict_rate: float = 1.0     # S4
    scope_split_rate: float = 1.0  # S5
    long_gap_delta: int = 1000     # S6
    noise_rate: float = 1.0        # S7


@dataclass
class RevisionScenario:
    scenario_id: str          # e.g. "S1_user_00"
    batch: BenchmarkBatch
    gold_state: StateOracle   # ground-truth validity state for this batch


def build_revision_benchmark(config: RevisionBenchmarkConfig) -> List[RevisionScenario]:
    """Return all scenarios across S1–S7 for each entity."""
    scenarios: List[RevisionScenario] = []
    scenarios.extend(_build_s1_distractors(config))
    scenarios.extend(_build_s2_revise(config))
    scenarios.extend(_build_s3_revert(config))
    scenarios.extend(_build_s4_conflict(config))
    scenarios.extend(_build_s5_scope_split(config))
    scenarios.extend(_build_s6_long_gap_partial_update(config))
    scenarios.extend(_build_s7_alias_noise(config))
    return scenarios
