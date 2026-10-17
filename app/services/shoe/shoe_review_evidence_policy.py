from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence, TypeVar

from app.services.shoe.shoe_feature_rules import (
    _has_negated_term,
    _has_unnegated_term,
    classify_point_evidence,
    normalize_review_text,
)


REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")
NO_REVIEW_SUMMARY = (
    "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다."
)

Polarity = Literal["POSITIVE", "NEGATIVE", "MIXED"]


@dataclass(frozen=True)
class ReviewCandidate:
    """One displayable review selected as evidence for a recommendation area.

    ``sentence`` is the bounded, reason-specific excerpt used by the summary.
    ``source_text`` is the original review body.  Feetfit_Server currently stores
    only a review ID and later displays that complete body, so batch selection
    uses it to reject reviews that contain evidence for another reason area.
    """

    review_id: int
    sentence: str
    similarity: float
    source_text: str | None = None


class CandidateLike(Protocol):
    review_id: int
    sentence: str


@dataclass(frozen=True)
class ReasonReviewAnalysis:
    excerpt: str
    polarity: Polarity
    concepts: tuple[str, ...]
    domains: frozenset[str]


CandidateT = TypeVar("CandidateT", bound=CandidateLike)


_CLAUSE_SPLIT_PATTERN = re.compile(r"[.!?;\n]+")
_INLINE_DOMAIN_BOUNDARY_PATTERN = re.compile(
    r"\s+(?=(?:발볼|앞볼|앞코|앞꿈치|전족부|앞발|발가락|엄지|발등|토박스|"
    r"뒤꿈치|뒤축|발목|쿠션|푹신|폭신|충격|반발|바닥|밑창|미드솔|아웃솔|"
    r"깔창|인솔|통풍|통기|열감|습기|땀|신발\s*내부))",
    re.IGNORECASE,
)
_HEEL_LOCK_PATTERN = re.compile(
    r"(?:heel\s*[- ]?\s*lock|힐\s*락|레이싱|"
    r"(?:뒤꿈치|뒷꿈치|뒤축|뒷굽|발목).{0,14}(?:고정|잡아|잡히|잡혀|헐렁|들뜨|벗겨|빠지)|"
    r"(?:고정|잡아|잡히|잡혀|헐렁|들뜨|벗겨|빠지).{0,14}(?:뒤꿈치|뒷꿈치|뒤축|뒷굽|발목))",
    re.IGNORECASE,
)
_UNTRUSTED_INSTRUCTION_PATTERN = re.compile(
    r"(?:시스템\s*지시|이전\s*지시|프롬프트|명령을\s*따|json\s*(?:출력|작성))",
    re.IGNORECASE,
)

_FOREFOOT_PATTERN = re.compile(
    r"(?:발볼|앞볼|앞코|앞꿈치|전족부|앞발|발가락|엄지|발등|토박스|toe\s*box|정사이즈|사이즈|반업|"
    r"한\s*(?:사이즈|치수)\s*(?:크게|작게|업|다운)|사이즈\s*(?:업|다운))",
    re.IGNORECASE,
)
_HEEL_ANATOMY_PATTERN = re.compile(r"(?:뒤꿈치|뒤축|뒷굽|발목)")
_HEEL_PERFORMANCE_PATTERN = re.compile(
    r"(?:쿠션|푹신|폭신|딱딱|단단|충격|반발|에너지\s*리턴|바닥|밑창|"
    r"미드솔|아웃솔|안정감|안정적|착화감|"
    r"(?:장시간|오래|종일|하루종일).{0,24}(?:신|착용|걷|뛰|달리|피로|불편|아프|부담|편하|편안)|"
    r"(?:피로|불편|아프|부담|편하|편안).{0,24}(?:장시간|오래|종일|하루종일))",
    re.IGNORECASE,
)
_INSOLE_PATTERN = re.compile(
    r"(?:깔창|인솔|신발\s*내부|내부\s*(?:착용|공간|환경)|통풍|통기|열감|습기|"
    r"땀|후끈|쾌적|어퍼.{0,8}시원|시원.{0,8}(?:통풍|통기|어퍼|내부))",
    re.IGNORECASE,
)
_NON_FIT_CONTEXT_PATTERN = re.compile(
    r"(?:디자인|색상|색감|코디|배송|포장|가격|사이즈\s*(?:표|안내|선택지))",
    re.IGNORECASE,
)

_CONCEPT_DIRECTION: dict[str, Literal["POSITIVE", "NEGATIVE"]] = {
    "WIDTH_ROOMY": "POSITIVE",
    "TOEBOX_ROOMY": "POSITIVE",
    "TRUE_SIZE": "POSITIVE",
    "FIT_COMFORTABLE": "POSITIVE",
    "WIDTH_TIGHT": "NEGATIVE",
    "TOEBOX_TIGHT": "NEGATIVE",
    "SIZE_UP": "NEGATIVE",
    "SIZE_HALF_UP": "NEGATIVE",
    "SIZE_FULL_UP": "NEGATIVE",
    "SIZE_DOWN": "NEGATIVE",
    "CUSHION_SOFT": "POSITIVE",
    "CUSHION_GOOD": "POSITIVE",
    "IMPACT_COMFORT": "POSITIVE",
    "LONG_WEAR_COMFORT": "POSITIVE",
    "STABLE_COMFORT": "POSITIVE",
    "CUSHION_FIRM": "NEGATIVE",
    "CUSHION_POOR": "NEGATIVE",
    "IMPACT_BURDEN": "NEGATIVE",
    "LONG_WEAR_DISCOMFORT": "NEGATIVE",
    "WEAR_DISCOMFORT": "NEGATIVE",
    "BREATHABLE": "POSITIVE",
    "COOL_INTERIOR": "POSITIVE",
    "DRY_INTERIOR": "POSITIVE",
    "INSOLE_COMFORT": "POSITIVE",
    "STUFFY": "NEGATIVE",
    "HOT_INTERIOR": "NEGATIVE",
    "MOIST_INTERIOR": "NEGATIVE",
    "INSOLE_DISCOMFORT": "NEGATIVE",
}

_CONCEPT_COPY = {
    "WIDTH_ROOMY": "발볼에 여유가 있고 편안하다는",
    "TOEBOX_ROOMY": "앞코와 발가락 공간이 여유롭다는",
    "TRUE_SIZE": "정사이즈가 잘 맞는다는",
    "FIT_COMFORTABLE": "발볼과 앞코의 착용감이 편안하다는",
    "WIDTH_TIGHT": "발볼이 좁거나 압박된다는",
    "TOEBOX_TIGHT": "앞코나 발가락이 조인다는",
    "SIZE_UP": "여유 있는 사이즈로 올려 신기를 권한다는",
    "SIZE_HALF_UP": "반 사이즈 업을 권한다는",
    "SIZE_FULL_UP": "한 사이즈 업을 권한다는",
    "SIZE_DOWN": "작은 사이즈도 함께 비교할 필요가 있다는",
    "CUSHION_SOFT": "쿠션이 부드럽고 푹신하다는",
    "CUSHION_GOOD": "쿠션감이 편안하다는",
    "IMPACT_COMFORT": "바닥 충격 부담이 적다는",
    "LONG_WEAR_COMFORT": "오래 신어도 발 부담이 적다는",
    "STABLE_COMFORT": "착화감이 안정적이고 편안하다는",
    "CUSHION_FIRM": "쿠션이나 바닥이 단단하게 느껴진다는",
    "CUSHION_POOR": "쿠션감이 부족하다는",
    "IMPACT_BURDEN": "바닥 충격이나 발 부담이 느껴진다는",
    "LONG_WEAR_DISCOMFORT": "장시간 착용하면 피로나 불편이 생긴다는",
    "WEAR_DISCOMFORT": "착화감이 불편하다는",
    "BREATHABLE": "통풍과 통기성이 좋다는",
    "COOL_INTERIOR": "신발 내부가 시원하고 쾌적하다는",
    "DRY_INTERIOR": "신발 내부의 습기나 땀이 적다는",
    "INSOLE_COMFORT": "깔창과 내부 착용감이 편안하다는",
    "STUFFY": "통풍이 부족해 내부가 답답하다는",
    "HOT_INTERIOR": "신발 내부에 열감이 느껴진다는",
    "MOIST_INTERIOR": "신발 내부에 땀이나 습기가 찬다는",
    "INSOLE_DISCOMFORT": "깔창이나 내부 착용감이 불편하다는",
}

_CONCEPT_PRIORITY = tuple(_CONCEPT_COPY)

_BANNED_UNGROUNDED_COPY = (
    "현재 발 상태와 비교적 잘 맞",
    "큰 주의점은 없",
    "적합합니다",
)

_SUMMARY_FORBIDDEN_BY_REASON = {
    "FOREFOOT": (
        "뒤꿈치",
        "발목",
        "쿠션",
        "충격",
        "반발력",
        "통풍",
        "통기성",
        "열감",
        "습기",
        "깔창",
    ),
    "HEEL": ("발볼", "앞코", "발가락", "통풍", "통기성", "열감", "습기", "깔창"),
    "INSOLE": ("발볼", "앞코", "발가락", "뒤꿈치", "고정감", "반발력", "충격 완화"),
}


def _clauses(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", text)
    pieces: list[str] = []
    for sentence in _CLAUSE_SPLIT_PATTERN.split(normalized):
        pieces.extend(_INLINE_DOMAIN_BOUNDARY_PATTERN.split(sentence))
    return tuple(re.sub(r"\s+", " ", piece).strip(" ,") for piece in pieces if piece.strip(" ,"))


def _domains_for_clause(clause: str) -> frozenset[str]:
    if _HEEL_LOCK_PATTERN.search(clause) or _UNTRUSTED_INSTRUCTION_PATTERN.search(clause):
        return frozenset()

    domains: set[str] = set()
    explicit_insole = bool(_INSOLE_PATTERN.search(clause))
    if _FOREFOOT_PATTERN.search(clause):
        domains.add("FOREFOOT")
    if explicit_insole:
        domains.add("INSOLE")

    # An explicit insole/interior subject owns a cushion predicate in the same
    # clause. Generic cushioning, impact and long-wear experience belong to the
    # HEEL load/impact area under the current measurement contract.
    heel_specific = bool(_HEEL_ANATOMY_PATTERN.search(clause))
    heel_performance = bool(_HEEL_PERFORMANCE_PATTERN.search(clause))
    if heel_specific or (heel_performance and not explicit_insole):
        domains.add("HEEL")
    return frozenset(domains)


def matched_reason_types(review_text: str) -> frozenset[str]:
    domains: set[str] = set()
    for clause in _clauses(review_text):
        domains.update(_domains_for_clause(clause))
    return frozenset(domains)


def _has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _point_signal_concepts(reason_type: str, clause: str) -> set[str]:
    concepts: set[str] = set()
    for signal in classify_point_evidence(clause):
        if reason_type == "FOREFOOT":
            if signal.subject == "WIDTH_SPACE":
                concepts.add("WIDTH_TIGHT" if signal.value == "TIGHT" else "WIDTH_ROOMY")
            elif signal.subject == "TOEBOX_SPACE":
                concepts.add("TOEBOX_TIGHT" if signal.value == "TIGHT" else "TOEBOX_ROOMY")
            elif signal.subject == "SIZE_OPTION":
                if signal.value == "FULL_UP":
                    concepts.add("SIZE_FULL_UP")
                elif signal.value == "HALF_UP":
                    concepts.add("SIZE_HALF_UP")
                elif signal.value == "GENERIC_UP":
                    concepts.add("SIZE_UP")
                elif signal.value == "DOWN":
                    concepts.add("SIZE_DOWN")
                elif signal.value == "TRUE_SIZE" and signal.stance == "POSITIVE":
                    concepts.add("TRUE_SIZE")
            elif signal.subject == "OVERALL_FIT" and signal.stance == "POSITIVE":
                concepts.add("FIT_COMFORTABLE")
        elif reason_type == "HEEL":
            if signal.subject == "CUSHION_SOFTNESS":
                concepts.add("CUSHION_SOFT" if signal.value == "SOFT" else "CUSHION_FIRM")
            elif signal.subject == "CUSHION_QUALITY":
                concepts.add("CUSHION_GOOD" if signal.stance == "POSITIVE" else "CUSHION_POOR")
            elif signal.subject == "SHOCK_FEEL":
                concepts.add("IMPACT_COMFORT" if signal.stance == "POSITIVE" else "IMPACT_BURDEN")
            elif signal.subject == "LONG_WEAR_COMFORT":
                concepts.add(
                    "LONG_WEAR_COMFORT"
                    if signal.stance == "POSITIVE"
                    else "LONG_WEAR_DISCOMFORT"
                )
            elif signal.subject == "HEEL_COMFORT":
                concepts.add("STABLE_COMFORT" if signal.stance == "POSITIVE" else "WEAR_DISCOMFORT")
    return concepts


def _forefoot_concepts(clause: str) -> set[str]:
    text = normalize_review_text(clause)
    concepts = _point_signal_concepts("FOREFOOT", clause)
    size_adjustment = bool(
        {"SIZE_UP", "SIZE_HALF_UP", "SIZE_FULL_UP", "SIZE_DOWN"} & concepts
    )
    width_pressure_is_low = bool(
        re.search(r"(?:발볼|앞볼).{0,18}(?:압박|조임?).{0,8}(?:적|없|크지\s*않)", text)
        or (
            re.search(r"(?:발볼|앞볼)", text)
            and _has_negated_term(
                text, ("좁", "끼", "압박", "조이", "타이트", "눌", "아프", "불편")
            )
        )
    )
    toe_pressure_is_low = bool(
        re.search(r"(?:앞코|발가락|엄지).{0,18}(?:압박|조임?).{0,8}(?:적|없|크지\s*않)", text)
        or (
            re.search(r"(?:앞코|발가락|엄지)", text)
            and _has_negated_term(
                text, ("좁", "끼", "압박", "조이", "타이트", "눌", "아프", "불편")
            )
        )
    )
    if (
        not width_pressure_is_low
        and re.search(r"(?:발볼|앞볼)", text)
        and _has_unnegated_term(
            text, ("좁", "끼", "압박", "조이", "타이트", "눌", "아프", "불편")
        )
    ):
        concepts.add("WIDTH_TIGHT")
    if (
        not toe_pressure_is_low
        and re.search(r"(?:앞코|발가락|엄지)", text)
        and _has_unnegated_term(
            text, ("좁", "끼", "압박", "조이", "타이트", "눌", "아프", "불편")
        )
    ):
        concepts.add("TOEBOX_TIGHT")
    if width_pressure_is_low:
        concepts.add("WIDTH_ROOMY")
    if toe_pressure_is_low:
        concepts.add("TOEBOX_ROOMY")
    if re.search(r"(?:발볼|앞볼).{0,18}(?:여유|넉넉|편하|편해|편안)", text):
        concepts.add("WIDTH_ROOMY")
    if re.search(r"(?:앞코|발가락).{0,18}(?:여유|넉넉|편하|편해|편안)", text):
        concepts.add("TOEBOX_ROOMY")
    # A positive outcome after sizing up still explains that the original shoe
    # runs tight. Keep it as caution evidence instead of mixed evidence.
    if size_adjustment:
        concepts.difference_update({"TRUE_SIZE", "FIT_COMFORTABLE"})
    return concepts


def _heel_concepts(clause: str) -> set[str]:
    text = normalize_review_text(clause)
    concepts = _point_signal_concepts("HEEL", clause)
    if (
        re.search(r"(?:바닥|밑창|솔|미드솔|아웃솔)", text)
        and _has_unnegated_term(text, ("딱딱", "단단", "푹신감 없"))
    ):
        concepts.add("CUSHION_FIRM")
    if (
        re.search(r"(?:쿠션|푹신감)", text)
        and _has_unnegated_term(text, ("부족", "없", "아쉽", "별로"))
    ):
        concepts.add("CUSHION_POOR")
    if (
        re.search(r"(?:충격|바닥)", text)
        and _has_unnegated_term(text, ("아프", "불편", "부담", "크"))
    ):
        concepts.add("IMPACT_BURDEN")
    has_duration = bool(re.search(r"(?:오래|장시간|종일|하루종일|잠깐.{0,12}(?:이상|밖에))", text))
    if has_duration and _has_unnegated_term(
        text, ("피로", "불편", "아프", "부담", "힘들", "못 신")
    ):
        concepts.add("LONG_WEAR_DISCOMFORT")
    if has_duration and _has_unnegated_term(
        text, ("편하", "편해", "편안", "부담이 적", "부담 없", "부담이 없")
    ):
        concepts.add("LONG_WEAR_COMFORT")
    explicit_low_burden = bool(
        re.search(r"부담(?:이|은|도)?\s*(?:적|없|크지\s*않)", text)
    )
    explicit_long_discomfort = bool(
        re.search(
            r"(?:오래|장시간|종일).{0,24}(?:피로|불편|아프|부담|힘들|못\s*신)",
            text,
        )
    )
    if explicit_low_burden:
        concepts.difference_update({"IMPACT_BURDEN", "LONG_WEAR_DISCOMFORT"})
        concepts.add("LONG_WEAR_COMFORT")
    if explicit_long_discomfort and not explicit_low_burden:
        concepts.discard("LONG_WEAR_COMFORT")
    if (
        not _NON_FIT_CONTEXT_PATTERN.search(text)
        and re.search(r"(?:착화감|안정감|안정적)", text)
        and _has_unnegated_term(text, ("좋", "편하", "편해", "편안", "안정적"))
    ):
        concepts.add("STABLE_COMFORT")
    if (
        re.search(r"(?:착화감|뒤꿈치|발목)", text)
        and _has_unnegated_term(text, ("불편", "아프", "쓸", "까짐", "미끄러"))
    ):
        concepts.add("WEAR_DISCOMFORT")
    return concepts


def _insole_concepts(clause: str) -> set[str]:
    text = normalize_review_text(clause)
    concepts: set[str] = set()
    has_breathability_subject = bool(re.search(r"(?:통풍|통기)", text))
    if has_breathability_subject and _has_unnegated_term(
        text, ("좋", "잘 되", "우수", "원활")
    ):
        concepts.add("BREATHABLE")
    if has_breathability_subject and _has_negated_term(
        text, ("나쁘", "부족", "별로", "답답")
    ):
        concepts.add("BREATHABLE")
    if (
        has_breathability_subject
        and (
            re.search(r"(?:통풍|통기).{0,14}안\s*되", text)
            or _has_unnegated_term(text, ("나쁘", "부족", "별로", "답답"))
        )
    ) or (
        "답답" in text and _has_unnegated_term(text, ("답답",))
    ):
        concepts.add("STUFFY")
    if re.search(r"(?:내부|어퍼|신발).{0,14}(?:시원|쾌적)", text) or re.search(
        r"(?:시원|쾌적).{0,14}(?:내부|어퍼|신발)", text
    ):
        concepts.add("COOL_INTERIOR")
    if _has_unnegated_term(text, ("열감", "후끈", "더워", "덥")):
        concepts.add("HOT_INTERIOR")
    if (
        re.search(r"(?:습기|축축|땀)", text)
        and _has_unnegated_term(text, ("차", "많", "심하", "젖", "축축"))
    ):
        concepts.add("MOIST_INTERIOR")
    if (
        re.search(r"(?:습기|땀)", text)
        and (
            re.search(r"(?:습기|땀).{0,12}(?:안\s*차|적|없)", text)
            or _has_negated_term(text, ("차", "많", "심하", "젖", "축축"))
        )
    ):
        concepts.add("DRY_INTERIOR")
    if (
        re.search(r"(?:깔창|인솔|내부)", text)
        and _has_unnegated_term(
            text, ("편하", "편해", "편안", "쾌적", "푹신", "폭신", "부드럽")
        )
    ):
        concepts.add("INSOLE_COMFORT")
    if re.search(
        r"(?:깔창|인솔|내부).{0,16}착용감.{0,8}(?:좋|만족)", text
    ):
        concepts.add("INSOLE_COMFORT")
    if (
        re.search(r"(?:깔창|인솔|내부)", text)
        and _has_unnegated_term(text, ("불편", "아프", "딱딱", "밀리", "미끄러"))
    ):
        concepts.add("INSOLE_DISCOMFORT")
    return concepts


def analyze_reason_review(reason_type: str, review_text: str) -> ReasonReviewAnalysis | None:
    if reason_type not in REASON_TYPES:
        raise ValueError(f"Unknown reason type: {reason_type}")
    if not normalize_review_text(review_text):
        return None
    if reason_type == "HEEL" and _HEEL_LOCK_PATTERN.search(review_text):
        return None

    selected_clauses: list[str] = []
    concepts: set[str] = set()
    all_domains = matched_reason_types(review_text)
    for clause in _clauses(review_text):
        domains = _domains_for_clause(clause)
        if reason_type not in domains:
            continue
        # A single unsplittable clause that itself mixes functional areas is not
        # safe evidence for any one card.
        if len(domains) != 1:
            continue
        clause_concepts = {
            "FOREFOOT": _forefoot_concepts,
            "HEEL": _heel_concepts,
            "INSOLE": _insole_concepts,
        }[reason_type](clause)
        if not clause_concepts:
            continue
        selected_clauses.append(clause)
        concepts.update(clause_concepts)

    if not concepts or not selected_clauses:
        return None
    directions = {_CONCEPT_DIRECTION[concept] for concept in concepts}
    polarity: Polarity = (
        "MIXED" if len(directions) > 1 else next(iter(directions))
    )
    ordered_concepts = tuple(
        concept for concept in _CONCEPT_PRIORITY if concept in concepts
    )
    return ReasonReviewAnalysis(
        excerpt=". ".join(dict.fromkeys(selected_clauses)),
        polarity=polarity,
        concepts=ordered_concepts,
        domains=all_domains,
    )


def _candidate_copy(candidate: CandidateT, excerpt: str) -> CandidateT:
    try:
        return replace(candidate, sentence=excerpt)
    except TypeError:
        # Production and public tests use frozen dataclasses. This fallback keeps
        # the helper usable with lightweight Protocol implementations as well.
        return ReviewCandidate(
            review_id=candidate.review_id,
            sentence=excerpt,
            similarity=float(getattr(candidate, "similarity", 0.0)),
            source_text=getattr(candidate, "source_text", None),
        )  # type: ignore[return-value]


def select_review_evidence(
    reason_type: str,
    risk_level: str,
    candidates: Sequence[CandidateT],
    limit: int = 3,
) -> tuple[CandidateT, ...]:
    """Select reason-pure, risk-aligned review evidence without changing score.

    LOW accepts positive experience only; HIGH accepts caution/negative
    experience only; MEDIUM deliberately preserves both directions when both
    are available. At most three unique review IDs are returned.
    """

    if reason_type not in REASON_TYPES:
        raise ValueError(f"Unknown reason type: {reason_type}")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Unknown risk level: {risk_level}")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    limit = min(limit, 3)
    if limit == 0:
        return ()
    if risk_level == "MEDIUM":
        # MEDIUM must retain both directions when both exist. Do not let a
        # legacy per-reason limit of one silently turn it into one-sided copy.
        limit = max(limit, 2)

    eligible: list[tuple[int, CandidateT, ReasonReviewAnalysis]] = []
    seen_ids: set[int] = set()
    for index, candidate in enumerate(candidates):
        if candidate.review_id in seen_ids:
            continue
        analysis = analyze_reason_review(reason_type, candidate.sentence)
        if analysis is None:
            continue
        source_text = getattr(candidate, "source_text", None)
        if source_text:
            source_domains = matched_reason_types(source_text)
            if source_domains != {reason_type}:
                continue
        if risk_level == "LOW" and analysis.polarity != "POSITIVE":
            continue
        if risk_level == "HIGH" and analysis.polarity != "NEGATIVE":
            continue
        seen_ids.add(candidate.review_id)
        eligible.append((index, _candidate_copy(candidate, analysis.excerpt), analysis))

    if risk_level != "MEDIUM":
        return tuple(candidate for _index, candidate, _analysis in eligible[:limit])

    chosen: list[tuple[int, CandidateT, ReasonReviewAnalysis]] = []
    positive = next(
        (item for item in eligible if item[2].polarity in {"POSITIVE", "MIXED"}),
        None,
    )
    negative = next(
        (item for item in eligible if item[2].polarity in {"NEGATIVE", "MIXED"}),
        None,
    )
    for item in (positive, negative):
        if item is not None and all(existing[1].review_id != item[1].review_id for existing in chosen):
            chosen.append(item)
    for item in eligible:
        if len(chosen) >= limit:
            break
        if all(existing[1].review_id != item[1].review_id for existing in chosen):
            chosen.append(item)
    chosen.sort(key=lambda item: item[0])
    return tuple(candidate for _index, candidate, _analysis in chosen[:limit])


def _comparison_sentence(
    reason_type: str, risk_level: str, *, has_measurement: bool
) -> str:
    if has_measurement:
        subject = {
            "FOREFOOT": "발볼·앞코 공간 특성을 위 측정 결과와 비교한 결과",
            "HEEL": "충격 완화·반발력·쿠션 특성을 위 측정 결과와 비교한 결과",
            "INSOLE": "통기성 특성을 위 측정 결과와 비교한 결과",
        }[reason_type]
    else:
        subject = {
            "FOREFOOT": "발볼·앞코 공간 특성에 대한 계산 결과",
            "HEEL": "충격 완화·반발력·쿠션 특성에 대한 계산 결과",
            "INSOLE": "통기성 특성에 대한 계산 결과",
        }[reason_type]
    outcome = {
        "LOW": "좋은 수준으로 계산되었습니다.",
        "MEDIUM": "보통 수준으로 계산되었습니다.",
        "HIGH": "주의가 필요한 수준으로 계산되었습니다.",
    }[risk_level]
    return f"제품의 {subject} {outcome}"


def _review_sentence(reason_type: str, candidates: Sequence[CandidateLike]) -> str:
    analyses = [
        analysis
        for candidate in candidates
        if (analysis := analyze_reason_review(reason_type, candidate.sentence)) is not None
    ]
    concept_counts = Counter(
        concept for analysis in analyses for concept in analysis.concepts
    )
    ordered = sorted(
        concept_counts,
        key=lambda concept: (-concept_counts[concept], _CONCEPT_PRIORITY.index(concept)),
    )

    has_positive = any(
        _CONCEPT_DIRECTION[concept] == "POSITIVE" for concept in ordered
    )
    has_negative = any(
        _CONCEPT_DIRECTION[concept] == "NEGATIVE" for concept in ordered
    )
    if has_positive and has_negative:
        positive = next(
            (concept for concept in ordered if _CONCEPT_DIRECTION[concept] == "POSITIVE"),
            None,
        )
        negative = next(
            (concept for concept in ordered if _CONCEPT_DIRECTION[concept] == "NEGATIVE"),
            None,
        )
        selected = [concept for concept in (positive, negative) if concept is not None]
        selected.extend(concept for concept in ordered if concept not in selected)
        ordered = selected

    phrases = [_CONCEPT_COPY[concept] for concept in ordered[:3]]
    if not phrases:
        raise AssertionError("selected reviews must yield at least one grounded concept")
    if has_positive and has_negative:
        positive_phrase = next(
            _CONCEPT_COPY[concept]
            for concept in ordered
            if _CONCEPT_DIRECTION[concept] == "POSITIVE"
        )
        negative_phrase = next(
            _CONCEPT_COPY[concept]
            for concept in ordered
            if _CONCEPT_DIRECTION[concept] == "NEGATIVE"
        )
        return (
            "해당 제품의 실제 착용 리뷰에서는 "
            f"{positive_phrase} 의견과 {negative_phrase} 의견이 함께 확인되어 "
            "착용감에 대한 평가가 엇갈립니다."
        )
    if len(phrases) == 1:
        body = phrases[0]
    else:
        body = ", ".join(phrases[:-1]) + f", {phrases[-1]}"
    return f"해당 제품의 실제 착용 리뷰에서는 {body} 의견이 확인됩니다."


def validate_reason_output(
    reason_type: str,
    risk_level: str,
    candidates: Sequence[CandidateLike],
    summary: str,
) -> None:
    if not candidates:
        if summary != NO_REVIEW_SUMMARY:
            raise AssertionError("reviewless reason must use the exact no-review summary")
        return
    if len(candidates) > 3 or len({candidate.review_id for candidate in candidates}) != len(candidates):
        raise AssertionError("reason review evidence must contain at most three unique reviews")
    for candidate in candidates:
        analysis = analyze_reason_review(reason_type, candidate.sentence)
        if analysis is None:
            raise AssertionError("review evidence does not match its reason type")
        if risk_level == "LOW" and analysis.polarity != "POSITIVE":
            raise AssertionError("LOW reason contains non-positive review evidence")
        if risk_level == "HIGH" and analysis.polarity != "NEGATIVE":
            raise AssertionError("HIGH reason contains non-negative review evidence")
    if any(copy in summary for copy in _BANNED_UNGROUNDED_COPY):
        raise AssertionError("summary contains an ungrounded compatibility claim")
    if any(term in summary for term in _SUMMARY_FORBIDDEN_BY_REASON[reason_type]):
        raise AssertionError("summary mixes evidence from another reason area")
    if reason_type == "HEEL" and re.search(
        r"(?:고정감|고정되|잘\s*잡|힐\s*락|heel\s*[- ]?\s*lock)", summary, re.IGNORECASE
    ):
        raise AssertionError("HEEL summary over-interprets measurement as heel lock")


def render_reason_summary(
    reason_type: str,
    risk_level: str,
    candidates: Sequence[CandidateLike],
    *,
    measurement_sentence: str | None = None,
) -> str:
    if not candidates:
        return NO_REVIEW_SUMMARY
    summary = " ".join(
        part
        for part in (
            measurement_sentence,
            _comparison_sentence(
                reason_type,
                risk_level,
                has_measurement=measurement_sentence is not None,
            ),
            _review_sentence(reason_type, candidates),
        )
        if part
    )
    validate_reason_output(reason_type, risk_level, candidates, summary)
    return summary


__all__ = [
    "NO_REVIEW_SUMMARY",
    "REASON_TYPES",
    "ReasonReviewAnalysis",
    "ReviewCandidate",
    "analyze_reason_review",
    "matched_reason_types",
    "render_reason_summary",
    "select_review_evidence",
    "validate_reason_output",
]
