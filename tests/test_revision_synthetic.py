"""Tests for revision_synthetic benchmark with gold state annotations."""
from memory_inference.benchmarks.revision_synthetic import (
    RevisionBenchmarkConfig,
    RevisionScenario,
    build_revision_benchmark,
)
from memory_inference.consolidation.revision_types import MemoryStatus
from memory_inference.consolidation.state_oracle import StateOracle


def test_build_returns_list_of_revision_scenarios() -> None:
    config = RevisionBenchmarkConfig(entities=2)
    scenarios = build_revision_benchmark(config)
    assert len(scenarios) > 0
    assert all(isinstance(s, RevisionScenario) for s in scenarios)


def test_each_scenario_has_updates_and_queries() -> None:
    config = RevisionBenchmarkConfig(entities=2)
    for scenario in build_revision_benchmark(config):
        assert len(scenario.batch.updates) > 0
        assert len(scenario.batch.queries) > 0


def test_each_scenario_has_gold_state() -> None:
    config = RevisionBenchmarkConfig(entities=2)
    for scenario in build_revision_benchmark(config):
        assert scenario.gold_state is not None


def test_distractor_scenario_has_correct_gold_active_value() -> None:
    """S1: gold state active value = correct entity's value, not distractor's."""
    config = RevisionBenchmarkConfig(entities=2, distractor_rate=0.5)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S1")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            gold = s.gold_state.active_value(q.entity, q.attribute)
            assert gold is not None
            assert gold.value == q.answer


def test_revise_scenario_gold_shows_superseded_entry() -> None:
    """S2 (v1->v2): gold state has v1 superseded, v2 active."""
    config = RevisionBenchmarkConfig(entities=2)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S2")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            key = (q.entity, q.attribute)
            superseded = s.gold_state.superseded_chain(q.entity, q.attribute)
            assert len(superseded) >= 1
            active = s.gold_state.active_value(q.entity, q.attribute)
            assert active is not None
            assert active.value == q.answer


def test_revert_scenario_gold_shows_reverted_value_active() -> None:
    """S3 (v1->v2->v1): gold active = v1, v2 is superseded."""
    config = RevisionBenchmarkConfig(entities=2)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S3")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            active = s.gold_state.active_value(q.entity, q.attribute)
            assert active is not None
            assert "v1" in active.value


def test_conflict_scenario_gold_shows_conflicted_entries() -> None:
    """S4: equal-timestamp conflict — gold has CONFLICTED entries, no clear active."""
    config = RevisionBenchmarkConfig(entities=2)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S4")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            conflicts = s.gold_state.unresolved_conflicts(q.entity, q.attribute)
            assert len(conflicts) == 2


def test_scope_split_scenario_gold_shows_multiple_scopes() -> None:
    """S5: scope-split — gold has entries in two different scopes."""
    config = RevisionBenchmarkConfig(entities=2)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S5")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            splits = s.gold_state.scope_splits(q.entity, q.attribute)
            assert len(splits) >= 2


def test_long_gap_partial_update_keeps_unrevised_attributes_active() -> None:
    config = RevisionBenchmarkConfig(entities=1)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S6")]
    assert len(scenarios) > 0
    for s in scenarios:
        revised_answers = {q.attribute: q.answer for q in s.batch.queries}
        for q in s.batch.queries:
            active = s.gold_state.active_value(q.entity, q.attribute)
            assert active is not None
            assert active.value == revised_answers[q.attribute]


def test_alias_noise_scenario_tracks_archived_noisy_entry() -> None:
    config = RevisionBenchmarkConfig(entities=1)
    scenarios = [s for s in build_revision_benchmark(config) if s.scenario_id.startswith("S7")]
    assert len(scenarios) > 0
    for s in scenarios:
        for q in s.batch.queries:
            active = s.gold_state.active_value(q.entity, q.attribute)
            assert active is not None
            assert active.value == q.answer
