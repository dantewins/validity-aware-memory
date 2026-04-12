from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Protocol, Sequence

from memory_inference.benchmarks.revision_synthetic import RevisionScenario
from memory_inference.consolidation.state_oracle import StateOracle
from memory_inference.types import InferenceExample, MemoryEntry

ABSTAIN_TOKEN = "ABSTAIN"


class SupportsStateView(Protocol):
    name: str
    maintenance_tokens: int
    maintenance_latency_ms: float

    def retrieve(self, entity: str, attribute: str, top_k: int = 5): ...

    def snapshot_size(self) -> int: ...


@dataclass(slots=True)
class ExperimentMetrics:
    policy_name: str
    total_queries: int
    correct_queries: int
    accuracy: float
    abstention_accuracy: float
    proactive_interference_rate: float
    avg_retrieved_items: float
    avg_retrieved_chars: float
    avg_context_tokens: float
    avg_completion_tokens: float
    avg_snapshot_size: float
    maintenance_tokens: int
    maintenance_latency_ms: float
    amortized_end_to_end_tokens: float
    avg_query_latency_ms: float
    cache_hit_rate: float
    current_state_exact_match: float = 0.0
    supersession_precision: float = 0.0
    supersession_recall: float = 0.0
    conflict_detection_f1: float = 0.0
    scope_split_accuracy: float = 0.0
    temporal_query_accuracy: float = 0.0
    state_table_edit_distance: float = 0.0


def compute_metrics(
    policy_name: str,
    examples: Iterable[InferenceExample],
    *,
    snapshot_sizes: Optional[Sequence[int]] = None,
    maintenance_tokens: int = 0,
    maintenance_latency_ms: float = 0.0,
) -> ExperimentMetrics:
    rows: List[InferenceExample] = list(examples)
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    abstention_queries = [row for row in rows if row.query.supports_abstention]
    abstention_correct = sum(
        1
        for row in abstention_queries
        if row.prediction == ABSTAIN_TOKEN
    )
    interference_count = sum(1 for row in rows if _has_proactive_interference(row))
    avg_items = sum(len(row.retrieved) for row in rows) / total if total else 0.0
    avg_chars = (
        sum(sum(len(entry.text()) for entry in row.retrieved) for row in rows) / total if total else 0.0
    )
    avg_prompt_tokens = (
        sum(row.prompt_tokens for row in rows) / total if total else 0.0
    )
    fallback_context_tokens = (
        sum(sum(_token_count(entry.text()) for entry in row.retrieved) for row in rows) / total
        if total
        else 0.0
    )
    avg_context_tokens = avg_prompt_tokens if avg_prompt_tokens > 0 else fallback_context_tokens
    avg_completion_tokens = (
        sum(row.completion_tokens for row in rows) / total if total else 0.0
    )
    avg_query_latency_ms = (
        sum(row.latency_ms for row in rows) / total if total else 0.0
    )
    cache_hit_rate = (
        sum(1 for row in rows if row.cache_hit) / total if total else 0.0
    )
    snapshot_values = list(snapshot_sizes or [])
    avg_snapshot_size = (
        sum(snapshot_values) / len(snapshot_values) if snapshot_values else 0.0
    )
    return ExperimentMetrics(
        policy_name=policy_name,
        total_queries=total,
        correct_queries=correct,
        accuracy=(correct / total) if total else 0.0,
        abstention_accuracy=(
            abstention_correct / len(abstention_queries) if abstention_queries else 0.0
        ),
        proactive_interference_rate=(interference_count / total) if total else 0.0,
        avg_retrieved_items=avg_items,
        avg_retrieved_chars=avg_chars,
        avg_context_tokens=avg_context_tokens,
        avg_completion_tokens=avg_completion_tokens,
        avg_snapshot_size=avg_snapshot_size,
        maintenance_tokens=maintenance_tokens,
        maintenance_latency_ms=maintenance_latency_ms,
        amortized_end_to_end_tokens=(
            avg_context_tokens + avg_completion_tokens + (maintenance_tokens / total if total else 0.0)
        ),
        avg_query_latency_ms=avg_query_latency_ms,
        cache_hit_rate=cache_hit_rate,
    )


def attach_state_metrics(
    base: ExperimentMetrics,
    scenarios: Sequence[RevisionScenario],
    policies: Sequence[SupportsStateView],
) -> ExperimentMetrics:
    exact_matches = 0
    total_queries = 0
    tp = fp = fn = 0
    supersession_tp = supersession_fp = supersession_fn = 0
    scope_correct = 0
    scope_total = 0
    edit_distance_total = 0.0
    temporal_total = 0
    temporal_correct = 0

    for scenario, policy in zip(scenarios, policies):
        for query in scenario.batch.queries:
            total_queries += 1
            predicted_entries = list(policy.retrieve(query.entity, query.attribute).entries)
            gold_active = scenario.gold_state.active_value(query.entity, query.attribute)
            predicted_active = _latest_active(predicted_entries)
            if gold_active is not None and predicted_active is not None and predicted_active.value == gold_active.value:
                exact_matches += 1

            gold_has_conflict = bool(scenario.gold_state.unresolved_conflicts(query.entity, query.attribute))
            predicted_has_conflict = _predict_conflict(policy, predicted_entries, query.entity, query.attribute)
            tp += int(gold_has_conflict and predicted_has_conflict)
            fp += int((not gold_has_conflict) and predicted_has_conflict)
            fn += int(gold_has_conflict and (not predicted_has_conflict))

            gold_scopes = scenario.gold_state.scope_splits(query.entity, query.attribute)
            if len(gold_scopes) > 1:
                scope_total += 1
                predicted_scopes = {entry.scope for entry in predicted_entries}
                if predicted_scopes == set(gold_scopes):
                    scope_correct += 1

            predicted_superseded = _predicted_superseded_values(policy, query.entity, query.attribute)
            gold_superseded = {entry.value for entry in scenario.gold_state.superseded_chain(query.entity, query.attribute)}
            supersession_tp += len(predicted_superseded & gold_superseded)
            supersession_fp += len(predicted_superseded - gold_superseded)
            supersession_fn += len(gold_superseded - predicted_superseded)

            if query.query_mode.name == "HISTORY":
                temporal_total += 1
                if predicted_active is not None and predicted_active.value == query.answer:
                    temporal_correct += 1

            edit_distance_total += _state_edit_distance(
                scenario.gold_state,
                predicted_entries,
                query.entity,
                query.attribute,
            )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    supersession_precision = (
        supersession_tp / (supersession_tp + supersession_fp)
        if (supersession_tp + supersession_fp)
        else 0.0
    )
    supersession_recall = (
        supersession_tp / (supersession_tp + supersession_fn)
        if (supersession_tp + supersession_fn)
        else 0.0
    )
    return replace(
        base,
        current_state_exact_match=exact_matches / total_queries if total_queries else 0.0,
        supersession_precision=supersession_precision,
        supersession_recall=supersession_recall,
        conflict_detection_f1=f1,
        scope_split_accuracy=scope_correct / scope_total if scope_total else 0.0,
        temporal_query_accuracy=temporal_correct / temporal_total if temporal_total else 0.0,
        state_table_edit_distance=edit_distance_total / total_queries if total_queries else 0.0,
    )


def _has_proactive_interference(row: InferenceExample) -> bool:
    mismatched = [
        entry for entry in row.retrieved
        if entry.entity == row.query.entity
        and entry.attribute == row.query.attribute
        and entry.value != row.query.answer
    ]
    return bool(mismatched and row.prediction != row.query.answer)


def _latest_active(entries: Sequence[MemoryEntry]) -> Optional[MemoryEntry]:
    if not entries:
        return None
    return max(entries, key=lambda entry: entry.timestamp)


def _predict_conflict(
    policy: SupportsStateView,
    entries: Sequence[MemoryEntry],
    entity: str,
    attribute: str,
) -> bool:
    conflict_table = getattr(policy, "conflict_table", None)
    if isinstance(conflict_table, dict):
        return bool(conflict_table.get((entity, attribute), []))
    if len(entries) < 2:
        return False
    latest_timestamp = max(entry.timestamp for entry in entries)
    latest_values = {entry.value for entry in entries if entry.timestamp == latest_timestamp}
    return len(latest_values) > 1


def _state_edit_distance(
    oracle: StateOracle,
    predicted_entries: Sequence[MemoryEntry],
    entity: str,
    attribute: str,
) -> float:
    gold_active = oracle.active_value(entity, attribute)
    predicted_active = _latest_active(predicted_entries)
    active_penalty = 0.0 if _same_value(gold_active, predicted_active) else 1.0

    gold_conflicts = oracle.unresolved_conflicts(entity, attribute)
    if predicted_entries:
        latest_timestamp = max(entry.timestamp for entry in predicted_entries)
        predicted_has_conflict = (
            len({entry.value for entry in predicted_entries if entry.timestamp == latest_timestamp}) > 1
        )
    else:
        predicted_has_conflict = False
    conflict_penalty = 0.0 if bool(gold_conflicts) == predicted_has_conflict else 1.0

    gold_scopes = set(oracle.scope_splits(entity, attribute))
    predicted_scopes = {entry.scope for entry in predicted_entries}
    scope_penalty = 0.0 if gold_scopes == predicted_scopes else 1.0
    return active_penalty + conflict_penalty + scope_penalty


def _predicted_superseded_values(
    policy: SupportsStateView,
    entity: str,
    attribute: str,
) -> set[str]:
    archive = getattr(policy, "archive", None)
    if not isinstance(archive, dict):
        return set()
    return {
        entry.value
        for entry in archive.get((entity, attribute), [])
        if getattr(entry, "status", None) is not None
    }


def _same_value(left: Optional[MemoryEntry], right: Optional[MemoryEntry]) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return left.value == right.value


def _token_count(text: str) -> int:
    return len(text.split())
