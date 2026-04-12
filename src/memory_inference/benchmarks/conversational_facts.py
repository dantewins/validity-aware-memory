from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_FACT_END_RE = re.compile(
    r"\s+(?:instead|now|though|actually|really|yesterday|today|tomorrow|"
    r"last\s+\w+|next\s+\w+|four\s+years\s+ago|years?\s+ago|months?\s+ago)\b.*$",
    re.IGNORECASE,
)
_CLAUSE_TAIL_RE = re.compile(
    r"\s+and\s+(?:i|we|you|he|she|they|my|our|your|his|her|their|created|made|started|went|watched|saw|researched)\b.*$",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StructuredFact:
    attribute: str
    value: str
    is_stateful: bool = True


_STATEFUL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("home_city", re.compile(r"\b(?:live|lived|living|based)\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("home_city", re.compile(r"\b(?:move|moved|moving)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("origin", re.compile(r"\b(?:move|moved|moving)\s+from\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\b(?:job|work|worked)\s+(?:at|for)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\b(?:switched|switching)\s+to\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("employer", re.compile(r"\bjoined\s+(?P<value>[A-Z][^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\bgraduated\s+with(?:\s+a\s+degree\s+in)?\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\bdegree\s+in\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("education", re.compile(r"\b(?:study|studied)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\b(?:prefer|preferred|like|love)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\bfavorite\s+(?:[^.!?;,]+?)\s+is\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("possession", re.compile(r"\b(?:bought|purchased|own|owns)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    ("possession", re.compile(r"\bgot(?:\s+a\s+new)?\s+(?!job\b)(?P<value>[^.!?;,]+)", re.IGNORECASE)),
    (
        "commute_duration",
        re.compile(
            r"\bcommute(?:\s+to\s+work)?(?:\s+(?:is|was|takes?|took))?\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "identity",
        re.compile(
            r"\b(?:identify as|identifies as|identity is|i am a|i'm a|she is a|he is a|they are a)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "relationship_status",
        re.compile(
            r"\b(?:relationship status (?:is|was)\s+|(?:i am|i'm|she is|he is|they are)\s+)(?P<value>single|married|engaged|divorced|widowed|in a relationship)\b",
            re.IGNORECASE,
        ),
    ),
)
_EPISODIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "research_topic",
        re.compile(
            r"\b(?:research|researched|researching|looked into|looking into|exploring|explored)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "attended_event",
        re.compile(r"\b(?:attended|watched|saw)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE),
    ),
    (
        "attended_event",
        re.compile(r"\bwent\s+to\s+(?:see\s+)?(?P<value>[^.!?;,]+)", re.IGNORECASE),
    ),
    (
        "venue",
        re.compile(r"\b(?:redeemed?|used)\b[^.!?;,]*\b(?:at|in)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE),
    ),
    (
        "venue",
        re.compile(
            r"\b(?:bought|purchased|ordered|picked up|grabbed)\b[^.!?;,]*\b(?:at|from)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "created_name",
        re.compile(
            r"\b(?:created|made|started)\b[^.!?;,]*\b(?:called|named|titled)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "created_name",
        re.compile(
            r"\bnamed\s+(?:my|the)\s+(?:new\s+)?(?:playlist|list|document|spreadsheet|album|project)\s+(?P<value>[^.!?;,]+)",
            re.IGNORECASE,
        ),
    ),
)

_ALL_PATTERNS: tuple[tuple[str, bool, re.Pattern[str]], ...] = tuple(
    (attribute, True, pattern) for attribute, pattern in _STATEFUL_PATTERNS
) + tuple(
    (attribute, False, pattern) for attribute, pattern in _EPISODIC_PATTERNS
)


def extract_structured_facts(text: str) -> list[StructuredFact]:
    """Extract lightweight slot/value facts from conversational text.

    These facts are intentionally coarse. The goal is not full IE fidelity;
    it is to give the memory layer attribute keys that can be revised over time
    instead of forcing every benchmark sample through a single `dialogue` slot.
    """
    content = text.strip()
    if not content or content.endswith("?"):
        return []

    facts: list[StructuredFact] = []
    seen: set[tuple[str, str]] = set()
    for attribute, is_stateful, pattern in _ALL_PATTERNS:
        for match in pattern.finditer(content):
            value = _clean_value(match.group("value"))
            if not value:
                continue
            key = (attribute, value.casefold(), is_stateful)
            if key in seen:
                continue
            seen.add(key)
            facts.append(StructuredFact(attribute=attribute, value=value, is_stateful=is_stateful))
    return facts


def infer_query_attributes(question: str) -> tuple[str, ...]:
    normalized = question.lower()
    if re.search(r"\b(when|what time|date|day|month|year)\b", normalized):
        return ()

    candidates: list[str] = []
    location_question = bool(re.search(r"\b(where|city|town|location)\b", normalized))
    if location_question and re.search(r"\b(move|moved)\s+from\b|\bfrom where\b", normalized):
        candidates.append("origin")
    elif location_question and re.search(r"\b(live|living|moved|move|based)\b", normalized):
        candidates.append("home_city")

    if re.search(r"\b(work|works|worked|job|employer|company)\b", normalized):
        candidates.append("employer")
    if re.search(r"\b(commute|travel to work|drive to work)\b", normalized):
        candidates.append("commute_duration")
    if re.search(r"\b(degree|graduate|graduated|study|studied|major)\b", normalized):
        candidates.append("education")
    if re.search(r"\b(prefer|preferred|favorite|like|love)\b", normalized):
        candidates.append("preference")
    if re.search(r"\b(bought|buy|purchased|purchase|own|owns|got)\b", normalized):
        candidates.append("possession")
    if re.search(r"\b(redeem|coupon|store|shop|bought .* at|purchase .* from)\b", normalized):
        candidates.append("venue")
    if re.search(r"\b(play|concert|show|movie|event)\b.*\b(attend|attended|watch|watched|see|saw)\b", normalized):
        candidates.append("attended_event")
    if re.search(r"\bname of\b|\bplaylist\b|\bcalled\b|\bnamed\b|\btitled\b", normalized):
        candidates.append("created_name")
    if re.search(r"\bidentity\b|\bidentify\b", normalized):
        candidates.append("identity")
    if re.search(r"\brelationship status\b|\bsingle\b|\bmarried\b|\bengaged\b|\bdivorced\b", normalized):
        candidates.append("relationship_status")
    if re.search(r"\bresearch\b|\bresearched\b|\blooking into\b|\blooked into\b|\bexploring\b", normalized):
        candidates.append("research_topic")
    return tuple(dict.fromkeys(candidates))


def choose_query_attribute(
    question: str,
    entity: str,
    updates: Sequence[object],
    *,
    fallback: str,
) -> str:
    candidates = infer_query_attributes(question)
    if not candidates:
        return fallback

    for attribute in candidates:
        if _has_matching_attribute(updates, entity=entity, attribute=attribute):
            return attribute
    return fallback


def _has_matching_attribute(updates: Sequence[object], *, entity: str, attribute: str) -> bool:
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute != attribute:
            continue
        if entity in {"conversation", "all"} or update_entity == entity:
            return True
    return False


def _clean_value(value: str) -> str:
    cleaned = _FACT_END_RE.sub("", value)
    cleaned = _CLAUSE_TAIL_RE.sub("", cleaned)
    cleaned = cleaned.strip(" \t\n\r\"'`.,;:!?")
    cleaned = re.sub(r"^(?:a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) < 2:
        return ""
    return cleaned
