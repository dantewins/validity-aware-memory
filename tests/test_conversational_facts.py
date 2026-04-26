from memory_inference.annotation.fact_extractor import extract_structured_facts
from memory_inference.annotation.query_intent import choose_query_attribute, infer_query_attributes
from tests.factories import make_record


def test_extract_structured_facts_covers_benchmark_event_patterns() -> None:
    facts = extract_structured_facts(
        "I redeemed a $5 coupon on coffee creamer at Target and created a playlist on Spotify called Summer Vibes."
    )
    by_key = {(fact.attribute, fact.value): fact for fact in facts}

    assert by_key[("venue", "Target")].is_stateful is False
    assert by_key[("created_name", "Summer Vibes")].is_stateful is False


def test_extract_structured_facts_covers_identity_and_relationship_status() -> None:
    facts = extract_structured_facts(
        "I identify as a transgender woman and my relationship status is single."
    )
    by_key = {(fact.attribute, fact.value): fact for fact in facts}

    assert by_key[("identity", "transgender woman")].is_stateful is True
    assert by_key[("relationship_status", "single")].is_stateful is True


def test_infer_query_attributes_covers_benchmark_question_templates() -> None:
    assert "attended_event" in infer_query_attributes(
        "What play did I attend at the local community theater?"
    )
    assert "venue" in infer_query_attributes(
        "Where did I redeem a $5 coupon on coffee creamer?"
    )
    assert "created_name" in infer_query_attributes(
        "What is the name of the playlist I created on Spotify?"
    )
    assert "commute_duration" in infer_query_attributes(
        "How long is my daily commute to work?"
    )


def test_extract_structured_facts_handles_relocation_and_started_work_patterns() -> None:
    facts = extract_structured_facts(
        "I relocated to Seattle and started working at Anthropic last month."
    )
    by_key = {(fact.attribute, fact.value): fact for fact in facts}

    assert ("home_city", "Seattle") in by_key
    assert ("employer", "Anthropic") in by_key


def test_choose_query_attribute_uses_ranked_available_attributes_before_dialogue_default() -> None:
    updates = [
        make_record(
            entry_id="turn-1",
            entity="Alice",
            attribute="dialogue",
            value="I now work at Google.",
            timestamp=1,
            session_id="s",
        ),
        make_record(
            entry_id="fact-1",
            entity="Alice",
            attribute="employer",
            value="Google",
            timestamp=1,
            session_id="s",
            metadata={
                "source_kind": "structured_fact",
                "support_text": "I now work at Google.",
                "memory_kind": "state",
            },
        ),
    ]

    assert choose_query_attribute(
        "What is Alice's affiliation now?",
        "Alice",
        updates,
        default_attribute="dialogue",
    ) == "employer"
