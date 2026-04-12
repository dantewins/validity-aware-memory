from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from memory_inference.benchmarks.longmemeval_adapter import LongMemEvalAdapter
from memory_inference.benchmarks.revision_synthetic import RevisionBenchmarkConfig, build_revision_benchmark
from memory_inference.consolidation.append_only import AppendOnlyMemoryPolicy
from memory_inference.consolidation.base import BaseMemoryPolicy
from memory_inference.consolidation.exact_match import ExactMatchMemoryPolicy
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
from memory_inference.consolidation.strong_retrieval import StrongRetrievalMemoryPolicy
from memory_inference.consolidation.summary_only import SummaryOnlyMemoryPolicy
from memory_inference.llm.mock_consolidator import MockConsolidator


PolicyFactory = Callable[[], BaseMemoryPolicy]


def default_policy_factories() -> list[PolicyFactory]:
    return [
        AppendOnlyMemoryPolicy,
        RecencySalienceMemoryPolicy,
        SummaryOnlyMemoryPolicy,
        ExactMatchMemoryPolicy,
        StrongRetrievalMemoryPolicy,
        lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
    ]


def policy_factory_by_name(name: str) -> PolicyFactory:
    lookup = {
        "append_only": AppendOnlyMemoryPolicy,
        "recency_salience": RecencySalienceMemoryPolicy,
        "summary_only": SummaryOnlyMemoryPolicy,
        "exact_match": ExactMatchMemoryPolicy,
        "strong_retrieval": StrongRetrievalMemoryPolicy,
        "offline_delta_v2": lambda: OfflineDeltaConsolidationPolicyV2(consolidator=MockConsolidator()),
    }
    if name not in lookup:
        raise KeyError(f"Unknown policy preset: {name}")
    return lookup[name]


@dataclass(slots=True)
class ExperimentPreset:
    name: str
    description: str


def default_presets() -> list[ExperimentPreset]:
    return [
        ExperimentPreset(
            name="synthetic_revision",
            description="Structured synthetic validity-maintenance benchmark with oracle state labels.",
        ),
        ExperimentPreset(
            name="longmemeval",
            description="LongMemEval-style structured evaluation using cached local JSON records.",
        ),
    ]


def build_synthetic_batches():
    return build_revision_benchmark(RevisionBenchmarkConfig())


def load_longmemeval_batches(path: str):
    return LongMemEvalAdapter().from_json(path)
