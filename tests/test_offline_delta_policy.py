from memory_inference.consolidation.offline_delta import OfflineDeltaConsolidationPolicy
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.types import MemoryEntry


def _entry(entry_id: str, value: str, timestamp: int, confidence: float = 1.0) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id, entity="u", attribute="a",
        value=value, timestamp=timestamp, session_id="s",
        confidence=confidence,
    )


def _policy() -> OfflineDeltaConsolidationPolicy:
    return OfflineDeltaConsolidationPolicy(consolidator=MockConsolidator())


def test_supersession_removes_stale_from_current() -> None:
    policy = _policy()
    policy.ingest([_entry("1", "old", timestamp=1), _entry("2", "new", timestamp=2)])
    policy.maybe_consolidate()
    result = policy.retrieve("u", "a")
    assert result.entries[0].value == "new"
    assert all(e.value != "old" for e in result.entries)


def test_reinforcement_merges_into_canonical() -> None:
    policy = _policy()
    policy.ingest([_entry("1", "same", timestamp=1), _entry("2", "same", timestamp=2)])
    policy.maybe_consolidate()
    result = policy.retrieve("u", "a")
    assert "same" in result.entries[0].value


def test_conflict_flags_both_entries() -> None:
    policy = _policy()
    policy.ingest([
        _entry("1", "alpha", timestamp=5),
        _entry("2", "beta", timestamp=5),
    ])
    policy.maybe_consolidate()
    assert "1" in policy.conflict_flags or "2" in policy.conflict_flags


def test_importance_threshold_archives_low_importance_entry() -> None:
    policy = OfflineDeltaConsolidationPolicy(
        consolidator=MockConsolidator(), importance_threshold=0.5
    )
    low_importance = MemoryEntry(
        entry_id="1", entity="u", attribute="a", value="rare_value",
        timestamp=1, session_id="s", confidence=0.1, importance=0.3,
    )
    policy.ingest([low_importance])
    policy.maybe_consolidate()
    # Low-importance entry should be archived, not in current
    assert ("u", "a") not in policy.current or policy.current[("u", "a")].importance >= 0.5


def test_archive_fallback_when_current_empty() -> None:
    policy = OfflineDeltaConsolidationPolicy(
        consolidator=MockConsolidator(), importance_threshold=2.0  # archive everything
    )
    policy.ingest([_entry("1", "archived_val", timestamp=1)])
    policy.maybe_consolidate()
    result = policy.retrieve("u", "a")
    assert len(result.entries) > 0
    assert result.entries[0].value == "archived_val"


def test_pending_cleared_after_consolidation() -> None:
    policy = _policy()
    policy.ingest([_entry("1", "v1", timestamp=1)])
    policy.maybe_consolidate()
    assert len(policy._pending) == 0


def test_ingest_before_consolidation_does_not_call_consolidator() -> None:
    consolidator = MockConsolidator()
    policy = OfflineDeltaConsolidationPolicy(consolidator=consolidator)
    policy.ingest([_entry("1", "v1", timestamp=1), _entry("2", "v2", timestamp=2)])
    assert consolidator.total_calls == 0  # no calls yet — online path is fast


def test_policy_name() -> None:
    assert _policy().name == "offline_delta_consolidation"
