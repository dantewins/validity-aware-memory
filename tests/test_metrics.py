from memory_inference.metrics import compute_metrics
from memory_inference.types import InferenceExample, MemoryEntry, Query


def test_metrics_report_context_tokens_from_retrieved_memory_not_full_prompt() -> None:
    query = Query(
        query_id="q",
        entity="user",
        attribute="home_city",
        question="Where does the user live now?",
        answer="Boston",
        timestamp=1,
        session_id="s",
    )
    retrieved = [
        MemoryEntry(
            entry_id="m1",
            entity="user",
            attribute="home_city",
            value="Boston",
            timestamp=0,
            session_id="s",
        )
    ]
    example = InferenceExample(
        query=query,
        retrieved=retrieved,
        prediction="Boston",
        correct=True,
        policy_name="mem0",
        prompt_tokens=99,
        completion_tokens=3,
    )

    metrics = compute_metrics("mem0", [example])

    assert metrics.avg_prompt_tokens == 99.0
    assert metrics.avg_context_tokens == 5.0
    assert metrics.amortized_end_to_end_tokens == 102.0
