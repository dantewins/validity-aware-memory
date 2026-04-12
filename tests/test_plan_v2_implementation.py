import json

from memory_inference.agent import AgentRunner
from memory_inference.benchmarks.longmemeval_adapter import LongMemEvalAdapter
from memory_inference.benchmarks.locomo_adapter import LoCoMoAdapter
from memory_inference.benchmarks.revision_synthetic import RevisionBenchmarkConfig, build_revision_benchmark
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.fixed_prompt_reader import FixedPromptReader
from memory_inference.llm.mock_consolidator import MockConsolidator


def test_recency_salience_prioritizes_recent_exact_match() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=2))
        if scenario.scenario_id.startswith("S1")
    )
    query = scenario.batch.queries[0]
    policy = RecencySalienceMemoryPolicy()
    policy.ingest(scenario.batch.updates)
    top = policy.retrieve(query.entity, query.attribute, top_k=1).entries[0]
    assert top.entity == query.entity
    assert top.attribute == query.attribute


def test_deterministic_reader_handles_history_queries() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S3")
    )
    query = scenario.batch.queries[0]
    query.query_mode = QueryMode.HISTORY
    query.answer = next(
        entry.value for entry in scenario.batch.updates
        if entry.entity == query.entity and entry.attribute == query.attribute
    )
    runner = AgentRunner(
        policy=RecencySalienceMemoryPolicy(),
        reasoner=DeterministicValidityReader(),
    )
    examples = runner.run_batches([scenario.batch])
    assert examples[0].prediction == query.answer


def test_fixed_prompt_reader_is_stable_stub() -> None:
    reader = FixedPromptReader(prompt_template="Use only memory.")
    assert reader.prompt_template == "Use only memory."


def test_offline_delta_respects_maintenance_frequency() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S2")
    )
    policy = OfflineDeltaConsolidationPolicyV2(
        consolidator=MockConsolidator(),
        maintenance_frequency=2,
    )
    policy.ingest(scenario.batch.updates)
    policy.maybe_consolidate()
    assert policy.retrieve("user_00", scenario.batch.queries[0].attribute).entries == []
    policy.maybe_consolidate()
    assert policy.retrieve("user_00", scenario.batch.queries[0].attribute).entries


def test_revision_benchmark_includes_long_gap_and_alias_scenarios() -> None:
    scenarios = build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
    scenario_ids = {scenario.scenario_id.split("_")[0] for scenario in scenarios}
    assert "S6" in scenario_ids
    assert "S7" in scenario_ids


def test_longmemeval_adapter_maps_records(tmp_path) -> None:
    path = tmp_path / "longmemeval.json"
    payload = [
        {
            "conversation_id": "conv-1",
            "updates": [
                {
                    "entity": "user_a",
                    "relation": "home_city",
                    "value": "Boston",
                    "timestamp": 1,
                }
            ],
            "query": {
                "entity": "user_a",
                "relation": "home_city",
                "question": "Where does user_a live now?",
                "answer": "Boston",
                "timestamp": 2,
                "query_mode": "CURRENT_STATE",
            },
        }
    ]
    path.write_text(json.dumps(payload))
    adapter = LongMemEvalAdapter()
    batches = adapter.from_json(path)
    assert len(batches) == 1
    assert batches[0].updates[0].attribute == "home_city"
    assert batches[0].queries[0].query_mode == QueryMode.CURRENT_STATE


def test_locomo_adapter_round_trips_cache(tmp_path) -> None:
    cache_path = tmp_path / "locomo_cache.json"
    adapter = LoCoMoAdapter(consolidator=MockConsolidator(), cache_path=cache_path)
    records = [
        {
            "dialogue_id": "dlg-1",
            "updates": [
                {
                    "entity": "user_a",
                    "relation": "favorite_editor",
                    "value": "vim",
                    "timestamp": 5,
                }
            ],
            "query": {
                "entity": "user_a",
                "relation": "favorite_editor",
                "question": "What editor does user_a prefer now?",
                "answer": "vim",
                "timestamp": 6,
            },
        }
    ]
    batches = adapter.from_records(records)
    cached_batches = adapter.from_cache()
    assert len(batches) == 1
    assert len(cached_batches) == 1
    assert cached_batches[0].queries[0].answer == "vim"
