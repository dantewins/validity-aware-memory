from memory_inference.agent import AgentRunner
from memory_inference.benchmarks.revision_synthetic import RevisionBenchmarkConfig, build_revision_benchmark
from memory_inference.consolidation.exact_match import ExactMatchMemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.consolidation.strong_retrieval import StrongRetrievalMemoryPolicy
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.metrics import ABSTAIN_TOKEN, attach_state_metrics, compute_metrics


def test_agent_uses_query_mode_aware_retrieval_for_conflicts() -> None:
    policy = OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator())
    runner = AgentRunner(policy=policy, reasoner=DeterministicValidityReader())
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S4")
    )
    query = scenario.batch.queries[0]
    query.query_mode = QueryMode.CONFLICT_AWARE
    results = runner.run_batches([scenario.batch])
    assert results[0].prediction == ABSTAIN_TOKEN


def test_exact_match_policy_preserves_multiple_scopes() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S5")
    )
    policy = ExactMatchMemoryPolicy()
    policy.ingest(scenario.batch.updates)
    retrieved = policy.retrieve(scenario.batch.queries[0].entity, scenario.batch.queries[0].attribute)
    assert {entry.scope for entry in retrieved.entries} == {"boston", "miami"}


def test_strong_retrieval_prioritizes_exact_entity_and_attribute() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=2))
        if scenario.scenario_id.startswith("S1")
    )
    policy = StrongRetrievalMemoryPolicy()
    policy.ingest(scenario.batch.updates)
    query = scenario.batch.queries[0]
    top = policy.retrieve(query.entity, query.attribute, top_k=1).entries[0]
    assert top.entity == query.entity
    assert top.attribute == query.attribute


def test_metrics_include_maintenance_cost_and_state_scores() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S2")
    )
    policy = OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator())
    runner = AgentRunner(policy=policy, reasoner=DeterministicValidityReader())
    examples = runner.run_batches([scenario.batch])
    base = compute_metrics(
        policy.name,
        examples,
        snapshot_sizes=[policy.snapshot_size()],
        maintenance_tokens=policy.maintenance_tokens,
        maintenance_latency_ms=policy.maintenance_latency_ms,
    )
    combined = attach_state_metrics(base, [scenario], [policy])
    assert combined.maintenance_tokens > 0
    assert combined.current_state_exact_match >= 0.0
    assert combined.state_table_edit_distance >= 0.0
