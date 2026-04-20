from memory_inference.benchmarks.hard_synthetic import (
    HardBenchmarkConfig,
    build_s1_distractor_injection,
    build_s2_value_reversal,
)
from memory_inference.types import BenchmarkBatch


def test_s1_produces_correct_number_of_batches() -> None:
    config = HardBenchmarkConfig(entities=3, distractor_count=2)
    batches = build_s1_distractor_injection(config)
    assert len(batches) == 3


def test_s1_batch_contains_distractors() -> None:
    config = HardBenchmarkConfig(entities=2, distractor_count=3)
    batches = build_s1_distractor_injection(config)
    # Each attribute should have 1 correct + 3 distractor entries
    batch = batches[0]
    for attr in config.attributes:
        attr_entries = [e for e in batch.updates if e.attribute == attr]
        assert len(attr_entries) == 1 + config.distractor_count


def test_s1_query_answer_matches_correct_entry() -> None:
    config = HardBenchmarkConfig(entities=2, distractor_count=2)
    batches = build_s1_distractor_injection(config)
    for batch in batches:
        for query in batch.queries:
            correct_entries = [
                e for e in batch.updates
                if e.entity == query.entity and e.attribute == query.attribute
                and e.value == query.answer
            ]
            assert len(correct_entries) == 1


def test_s2_reversal_answer_is_reverted_value() -> None:
    config = HardBenchmarkConfig(entities=2, reversal_rate=1.0)
    batches = build_s2_value_reversal(config)
    for batch in batches:
        for query in batch.queries:
            # With reversal_rate=1.0, all attributes revert: answer should be v1
            assert "v1" in query.answer


def test_s2_no_reversal_answer_is_v2() -> None:
    config = HardBenchmarkConfig(entities=2, reversal_rate=0.0)
    batches = build_s2_value_reversal(config)
    for batch in batches:
        for query in batch.queries:
            assert "v2" in query.answer
