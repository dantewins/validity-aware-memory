from __future__ import annotations

import dataclasses
import re
import uuid
from typing import List, Optional, Set

from memory_inference.annotation.fact_extractor import extract_structured_facts
from memory_inference.annotation.salience import estimate_confidence, estimate_importance
from memory_inference.domain.enums import RevisionOp, UpdateType
from memory_inference.domain.memory import MemoryRecord
from memory_inference.llm.consolidator_base import BaseConsolidator

_UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|might|perhaps|possibly|probably|unclear|not sure|unsure|guess|seems)\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_LOW_CONFIDENCE_THRESHOLD = 0.45


class BenchmarkHeuristicConsolidator(BaseConsolidator):
    """Benchmark-oriented deterministic consolidator.

    This is stronger than the unit-test MockConsolidator but still fully local and
    reproducible. Its purpose is to make the ODV2 family meaningfully exercise
    revision-state behavior during benchmark runs.
    """

    def __init__(self, low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD) -> None:
        super().__init__()
        self.low_confidence_threshold = low_confidence_threshold

    def classify_update(self, new_entry: MemoryRecord, existing: MemoryRecord) -> UpdateType:
        self.total_calls += 1
        if self._same_value(new_entry.value, existing.value):
            return UpdateType.REINFORCEMENT
        if new_entry.timestamp > existing.timestamp:
            return UpdateType.SUPERSESSION
        return UpdateType.CONFLICT

    def merge_entries(self, entries: List[MemoryRecord]) -> MemoryRecord:
        self.total_calls += 1
        if not entries:
            raise ValueError("merge_entries requires at least one entry")
        richest = max(
            entries,
            key=lambda entry: (
                entry.confidence,
                entry.importance,
                len(entry.support_text),
                entry.timestamp,
            ),
        )
        merged_support = " | ".join(
            dict.fromkeys(entry.support_text.strip() for entry in entries if entry.support_text.strip())
        )
        return dataclasses.replace(
            richest,
            confidence=min(1.0, max(entry.confidence for entry in entries) + 0.05),
            importance=max(entry.importance for entry in entries),
            support_text=merged_support or richest.support_text,
        )

    def extract_facts(
        self,
        text: str,
        entity: str,
        session_id: str,
        timestamp: int,
    ) -> List[MemoryRecord]:
        self.total_calls += 1
        confidence = estimate_confidence(text, speaker=entity, attribute="dialogue")
        importance = estimate_importance(text, speaker=entity, attribute="dialogue")
        facts: List[MemoryRecord] = []
        for fact in extract_structured_facts(text):
            facts.append(
                MemoryRecord(
                    record_id=str(uuid.uuid4()),
                    entity=entity,
                    attribute=fact.attribute,
                    value=fact.value,
                    timestamp=timestamp,
                    session_id=session_id,
                    confidence=min(1.0, confidence + 0.08),
                    importance=min(1.8, importance + 0.15),
                    source_kind="structured_fact",
                    source_attribute="dialogue",
                    memory_kind="state" if fact.is_stateful else "event",
                    support_text=text,
                )
            )
        return facts

    def classify_revision(
        self,
        new_entry: MemoryRecord,
        existing: Optional[MemoryRecord],
        prior_values: Optional[Set[str]] = None,
    ) -> RevisionOp:
        self.total_calls += 1

        normalized_new = self._normalize_value(new_entry.value)
        normalized_prior = {self._normalize_value(value) for value in (prior_values or set())}

        if existing is None:
            if self._is_low_confidence(new_entry):
                return RevisionOp.LOW_CONFIDENCE
            return RevisionOp.ADD

        normalized_existing = self._normalize_value(existing.value)

        if self._is_low_confidence(new_entry):
            return RevisionOp.LOW_CONFIDENCE

        if self._should_split_scope(new_entry, existing):
            return RevisionOp.SPLIT_SCOPE

        if self._same_value(normalized_new, normalized_existing):
            return RevisionOp.REINFORCE

        if new_entry.timestamp == existing.timestamp:
            return RevisionOp.CONFLICT_UNRESOLVED

        if normalized_new in normalized_prior:
            return RevisionOp.REVERT

        if new_entry.timestamp > existing.timestamp:
            return RevisionOp.REVISE

        return RevisionOp.CONFLICT_UNRESOLVED

    def _is_low_confidence(self, entry: MemoryRecord) -> bool:
        if entry.confidence < self.low_confidence_threshold:
            return True
        content = " ".join(part for part in (entry.value, entry.support_text) if part)
        return bool(_UNCERTAINTY_RE.search(content))

    def _should_split_scope(self, new_entry: MemoryRecord, existing: MemoryRecord) -> bool:
        if new_entry.scope == existing.scope:
            return False
        if "default" in {new_entry.scope, existing.scope}:
            return False
        return new_entry.memory_kind != "state"

    def _same_value(self, left: str, right: str) -> bool:
        normalized_left = self._normalize_value(left)
        normalized_right = self._normalize_value(right)
        if normalized_left == normalized_right:
            return True
        return self._one_contains_other(normalized_left, normalized_right)

    @staticmethod
    def _one_contains_other(left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_tokens = left.split()
        right_tokens = right.split()
        if min(len(left_tokens), len(right_tokens)) < 2:
            return False
        return left in right or right in left

    @staticmethod
    def _normalize_value(value: str) -> str:
        normalized = value.lower()
        normalized = _PUNCT_RE.sub(" ", normalized)
        normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
        return normalized
