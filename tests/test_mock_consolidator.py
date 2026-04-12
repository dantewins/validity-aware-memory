from memory_inference.llm.mock_consolidator import MockConsolidator


def test_extract_facts_parses_key_value_pairs() -> None:
    c = MockConsolidator()
    facts = c.extract_facts(
        "home_city=Boston; preferred_language=Python",
        entity="user_01",
        session_id="s1",
        timestamp=5,
    )
    assert len(facts) == 2
    assert any(f.attribute == "home_city" and f.value == "Boston" for f in facts)
    assert any(f.attribute == "preferred_language" and f.value == "Python" for f in facts)
    assert all(f.entity == "user_01" and f.session_id == "s1" and f.timestamp == 5 for f in facts)


def test_consolidator_tracks_total_calls() -> None:
    c = MockConsolidator()
    assert c.total_calls == 0
    c.extract_facts("a=b", entity="u", session_id="s", timestamp=1)
    assert c.total_calls == 1


def test_extract_facts_returns_empty_for_no_key_value_pairs() -> None:
    c = MockConsolidator()
    facts = c.extract_facts("no equals signs here", entity="u", session_id="s", timestamp=1)
    assert facts == []
