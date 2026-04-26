from __future__ import annotations

import re
from typing import Sequence

from memory_inference.domain.enums import QueryMode

_LOCOMO_CATEGORY_TO_MODE = {
    "single-hop": QueryMode.CURRENT_STATE,
    "multi-hop": QueryMode.CURRENT_STATE,
    "temporal": QueryMode.HISTORY,
    "open-ended": QueryMode.CURRENT_STATE,
    "adversarial": QueryMode.CONFLICT_AWARE,
    "1": QueryMode.CURRENT_STATE,
    "2": QueryMode.HISTORY,
    "3": QueryMode.CURRENT_STATE,
    "4": QueryMode.CURRENT_STATE,
    "5": QueryMode.CONFLICT_AWARE,
}

_LONGMEMEVAL_QUESTION_TYPE_TO_MODE = {
    "single-session-user": QueryMode.CURRENT_STATE,
    "single-session-assistant": QueryMode.CURRENT_STATE,
    "single-session-preference": QueryMode.CURRENT_STATE,
    "multi-session": QueryMode.CURRENT_STATE,
    "temporal-reasoning": QueryMode.HISTORY,
    "knowledge-update": QueryMode.CURRENT_STATE,
    "temporal-ordering": QueryMode.HISTORY,
}

_ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "home_city": ("where", "city", "town", "location", "live", "living", "moved", "based", "relocated"),
    "origin": ("from", "origin", "originally", "moved", "relocated"),
    "employer": ("work", "works", "worked", "job", "employer", "company", "firm", "startup", "office", "hired", "affiliation"),
    "commute_duration": ("commute", "travel", "drive", "ride", "minutes", "hours"),
    "education": ("degree", "graduate", "graduated", "study", "studied", "major", "school", "college"),
    "preference": ("prefer", "preferred", "favorite", "like", "love", "enjoy"),
    "possession": ("bought", "buy", "purchased", "purchase", "own", "owns", "got"),
    "venue": ("redeem", "coupon", "store", "shop", "venue", "from", "at"),
    "attended_event": ("concert", "show", "movie", "event", "attended", "watched", "saw"),
    "created_name": ("called", "named", "titled", "playlist", "document", "project", "name"),
    "identity": ("identity", "identify"),
    "relationship_status": ("relationship", "single", "married", "engaged", "divorced"),
    "research_topic": ("research", "researched", "looking", "exploring", "topic"),
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "did", "do", "does", "for", "from", "he", "her",
    "him", "his", "i", "in", "is", "it", "me", "my", "now", "of", "on", "she", "the", "their",
    "them", "they", "to", "user", "was", "what", "when", "where", "which", "who", "with", "you",
}


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

    if re.search(r"\b(work|works|worked|job|employer|company|firm|startup|hired)\b", normalized):
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
    default_attribute: str,
) -> str:
    candidates = infer_query_attributes(question)
    available_attributes = _available_attributes(updates, entity=entity)
    if candidates:
        for attribute in candidates:
            if attribute in available_attributes:
                return attribute
        ranked_candidates = _rank_attributes(
            candidates,
            question=question,
            entity=entity,
            updates=updates,
        )
        if ranked_candidates and ranked_candidates[0][1] > 0.0:
            return ranked_candidates[0][0]

    ranked_available = _rank_attributes(
        available_attributes,
        question=question,
        entity=entity,
        updates=updates,
    )
    if ranked_available and ranked_available[0][1] >= 1.0:
        return ranked_available[0][0]
    return default_attribute


def locomo_query_mode(category: object) -> QueryMode:
    return _LOCOMO_CATEGORY_TO_MODE.get(str(category).strip().lower(), QueryMode.CURRENT_STATE)


def longmemeval_query_mode(question_type: str) -> QueryMode:
    return _LONGMEMEVAL_QUESTION_TYPE_TO_MODE.get(question_type, QueryMode.CURRENT_STATE)


def should_skip_locomo_question(category: object, answer: object) -> bool:
    normalized_category = str(category).strip().lower()
    if answer is not None and str(answer).strip():
        return False
    return normalized_category in {"5", "adversarial"}


def infer_locomo_query_entity(question: str, speakers: set[str]) -> str:
    question_lower = question.lower()
    matches = [
        speaker
        for speaker in speakers
        if speaker and speaker.lower() in question_lower
    ]
    if len(matches) == 1:
        return matches[0]
    return "conversation"


def infer_longmemeval_query_entity(question_type: str) -> str:
    if question_type == "single-session-assistant":
        return "assistant"
    return "user"


def _has_matching_attribute(updates: Sequence[object], *, entity: str, attribute: str) -> bool:
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute != attribute:
            continue
        if entity in {"conversation", "all"} or update_entity == entity:
            return True
    return False


def _available_attributes(updates: Sequence[object], *, entity: str) -> tuple[str, ...]:
    attributes: list[str] = []
    seen: set[str] = set()
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute in {"dialogue", "event"}:
            continue
        if entity not in {"conversation", "all"} and update_entity != entity:
            continue
        if update_attribute in seen:
            continue
        seen.add(update_attribute)
        attributes.append(update_attribute)
    return tuple(attributes)


def _rank_attributes(
    attributes: Sequence[str],
    *,
    question: str,
    entity: str,
    updates: Sequence[object],
) -> list[tuple[str, float]]:
    question_terms = _question_terms(question)
    scored: list[tuple[str, float]] = []
    for attribute in attributes:
        alias_score = _alias_overlap(attribute, question_terms) * 3.0
        evidence_score = _attribute_evidence_score(
            attribute,
            question_terms=question_terms,
            entity=entity,
            updates=updates,
        )
        scored.append((attribute, alias_score + evidence_score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _attribute_evidence_score(
    attribute: str,
    *,
    question_terms: set[str],
    entity: str,
    updates: Sequence[object],
) -> float:
    best = 0.0
    for update in updates:
        update_entity = getattr(update, "entity", "")
        update_attribute = getattr(update, "attribute", "")
        if update_attribute != attribute:
            continue
        if entity not in {"conversation", "all"} and update_entity != entity:
            continue
        content = " ".join(
            part
            for part in (
                str(getattr(update, "value", "")),
                str(getattr(update, "support_text", "")),
                str(getattr(update, "source_attribute", "")),
            )
            if part
        ).lower()
        content_terms = {
            token
            for token in _TOKEN_RE.findall(content)
            if token not in _STOPWORDS
        }
        overlap = len(question_terms & content_terms)
        recency = float(getattr(update, "timestamp", 0)) * 0.001
        structured_bonus = 0.5 if getattr(update, "source_kind", "") == "structured_fact" else 0.0
        best = max(best, overlap * 2.0 + structured_bonus + recency)
    return best


def _alias_overlap(attribute: str, question_terms: set[str]) -> int:
    aliases = _ATTRIBUTE_ALIASES.get(attribute, ())
    return sum(1 for alias in aliases if alias in question_terms)


def _question_terms(question: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(question.lower())
        if token not in _STOPWORDS
    }
