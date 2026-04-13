from memory_inference.consolidation.dense_retrieval import DenseRetrievalMemoryPolicy
from memory_inference.consolidation.mem0 import Mem0MemoryPolicy
from memory_inference.experiment_registry import policy_factory_by_name
from memory_inference.types import MemoryEntry, Query


def test_dense_retrieval_prioritizes_semantic_match_for_dialogue_query() -> None:
    policy = DenseRetrievalMemoryPolicy()
    policy.ingest(
        [
            MemoryEntry(
                entry_id="target",
                entity="user",
                attribute="dialogue",
                value="I graduated with Business Administration.",
                timestamp=1,
                session_id="s",
            ),
            MemoryEntry(
                entry_id="distractor",
                entity="user",
                attribute="dialogue",
                value="I went hiking in the mountains this weekend.",
                timestamp=2,
                session_id="s",
            ),
        ]
    )
    query = Query(
        query_id="q1",
        entity="user",
        attribute="dialogue",
        question="What degree did I graduate with?",
        answer="Business Administration",
        timestamp=3,
        session_id="s",
    )

    retrieved = policy.retrieve_for_query(query)

    assert retrieved.entries
    assert retrieved.entries[0].entry_id == "target"


def test_dense_retrieval_expands_structured_fact_with_support_text() -> None:
    policy = DenseRetrievalMemoryPolicy()
    policy.ingest(
        [
            MemoryEntry(
                entry_id="support-target",
                entity="user",
                attribute="dialogue",
                value="I redeemed a $5 coupon on coffee creamer at Target.",
                timestamp=0,
                session_id="s",
            ),
            MemoryEntry(
                entry_id="fact-target",
                entity="user",
                attribute="venue",
                value="Target",
                timestamp=0,
                session_id="s",
                metadata={
                    "source_kind": "structured_fact",
                    "source_entry_id": "support-target",
                    "support_text": "I redeemed a $5 coupon on coffee creamer at Target.",
                },
            ),
            MemoryEntry(
                entry_id="fact-other",
                entity="user",
                attribute="venue",
                value="Trader Joe's",
                timestamp=3,
                session_id="s",
                metadata={
                    "source_kind": "structured_fact",
                    "support_text": "I bought bread at Trader Joe's.",
                },
            ),
        ]
    )
    query = Query(
        query_id="q2",
        entity="user",
        attribute="venue",
        question="Where did I redeem a $5 coupon on coffee creamer?",
        answer="Target",
        timestamp=4,
        session_id="s",
    )

    retrieved = policy.retrieve_for_query(query)

    assert retrieved.entries[0].value == "Target"
    assert any(entry.entry_id == "support-target" for entry in retrieved.entries)


def test_mem0_compacts_latest_state_and_retrieves_current_value() -> None:
    policy = Mem0MemoryPolicy()
    policy.ingest(
        [
            MemoryEntry(
                entry_id="old",
                entity="user",
                attribute="home_city",
                value="Boston",
                timestamp=1,
                session_id="s",
                scope="default",
                metadata={
                    "source_kind": "structured_fact",
                    "support_text": "I live in Boston.",
                },
            ),
            MemoryEntry(
                entry_id="new",
                entity="user",
                attribute="home_city",
                value="Seattle",
                timestamp=2,
                session_id="s",
                scope="default",
                metadata={
                    "source_kind": "structured_fact",
                    "support_text": "I moved to Seattle.",
                },
            ),
        ]
    )
    query = Query(
        query_id="q3",
        entity="user",
        attribute="home_city",
        question="Where do I live now?",
        answer="Seattle",
        timestamp=3,
        session_id="s",
    )

    retrieved = policy.retrieve_for_query(query)

    assert len(policy.active_state) == 1
    assert policy.active_state[("user", "home_city", "default")].value == "Seattle"
    assert retrieved.entries[0].value == "Seattle"


def test_policy_registry_returns_dense_and_mem0_factories() -> None:
    dense = policy_factory_by_name("dense_retrieval")()
    mem0 = policy_factory_by_name("mem0")()

    assert dense.name == "dense_retrieval"
    assert mem0.name == "mem0"
