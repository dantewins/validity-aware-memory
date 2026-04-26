from memory_inference.domain.enums import RevisionOp
from memory_inference.llm.benchmark_consolidator import BenchmarkHeuristicConsolidator
from tests.factories import make_record


def test_benchmark_consolidator_flags_low_confidence_revisions() -> None:
    consolidator = BenchmarkHeuristicConsolidator()
    existing = make_record(
        entry_id="old",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
        confidence=0.9,
    )
    new = make_record(
        entry_id="new",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        confidence=0.2,
    )

    assert consolidator.classify_revision(new, existing, prior_values={"Google"}) == RevisionOp.LOW_CONFIDENCE


def test_benchmark_consolidator_detects_reverts() -> None:
    consolidator = BenchmarkHeuristicConsolidator()
    existing = make_record(
        entry_id="current",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=3,
        session_id="s",
        confidence=0.9,
    )
    new = make_record(
        entry_id="revert",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=4,
        session_id="s",
        confidence=0.9,
    )

    assert consolidator.classify_revision(new, existing, prior_values={"Google", "Meta"}) == RevisionOp.REVERT


def test_benchmark_consolidator_detects_same_timestamp_conflicts() -> None:
    consolidator = BenchmarkHeuristicConsolidator()
    existing = make_record(
        entry_id="google",
        entity="Alice",
        attribute="employer",
        value="Google",
        timestamp=2,
        session_id="s",
        confidence=0.9,
    )
    new = make_record(
        entry_id="meta",
        entity="Alice",
        attribute="employer",
        value="Meta",
        timestamp=2,
        session_id="s",
        confidence=0.9,
    )

    assert consolidator.classify_revision(new, existing, prior_values={"Google"}) == RevisionOp.CONFLICT_UNRESOLVED
