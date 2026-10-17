from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Literal

from app.schemas.shoe_server import ServerShoeCharacteristics
from app.services.shoe.shoe_feature_rules import (
    PointSignal,
    classify_point_evidence,
    extract_point_summary_evidence,
    normalize_review_text,
)


EvidenceMode = Literal["SINGLE", "CONSENSUS", "MIXED", "LAB", "CONFLICT"]
EvidenceSource = Literal["MUSINSA", "RUNREPEAT"]
EvidenceStrength = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class PointEvidenceFact:
    evidence_id: str
    signal: PointSignal
    source: EvidenceSource
    mode: EvidenceMode
    review_ids: tuple[int, ...]
    evidence_texts: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.review_ids) if self.source == "MUSINSA" else 1


@dataclass(frozen=True)
class PointSummaryClaim:
    text: str
    evidence_ids: tuple[str, ...]


_EXACT_LAB_CLAIM_TEXT: dict[tuple[str, str], str] = {
    ("LAB_WIDTH_SPACE", "TIGHT"): "발볼 공간은 낮은 편입니다",
    ("LAB_WIDTH_SPACE", "BALANCED"): "발볼 공간은 보통 수준입니다",
    ("LAB_WIDTH_SPACE", "ROOMY"): "발볼 공간은 높은 편입니다",
    ("LAB_TOEBOX_SPACE", "TIGHT"): "앞코 공간은 낮은 편입니다",
    ("LAB_TOEBOX_SPACE", "BALANCED"): "앞코 공간은 보통 수준입니다",
    ("LAB_TOEBOX_SPACE", "ROOMY"): "앞코 공간은 높은 편입니다",
    ("CUSHION_SOFTNESS", "FIRM"): "쿠션감은 낮은 편입니다",
    ("CUSHION_SOFTNESS", "BALANCED"): "쿠션감은 보통 수준입니다",
    ("CUSHION_SOFTNESS", "SOFT"): "쿠션감은 높은 편입니다",
    ("HEEL_STRUCTURE", "FLEXIBLE"): "뒤꿈치 구조 강성은 낮은 편입니다",
    ("HEEL_STRUCTURE", "BALANCED"): "뒤꿈치 구조 강성은 보통 수준입니다",
    ("HEEL_STRUCTURE", "FIRM"): "뒤꿈치 구조 강성은 높은 편입니다",
    ("BREATHABILITY", "LOW"): "통기성은 낮은 편입니다",
    ("BREATHABILITY", "MEDIUM"): "통기성은 보통 수준입니다",
    ("BREATHABILITY", "HIGH"): "통기성은 높은 편입니다",
    ("SHOCK_ABSORPTION", "LOW"): "충격 완화 수준은 낮은 편입니다",
    ("SHOCK_ABSORPTION", "MEDIUM"): "충격 완화 수준은 보통 수준입니다",
    ("SHOCK_ABSORPTION", "HIGH"): "충격 완화 수준은 높은 편입니다",
    ("ENERGY_RETURN", "LOW"): "반발력은 낮은 편입니다",
    ("ENERGY_RETURN", "MEDIUM"): "반발력은 보통 수준입니다",
    ("ENERGY_RETURN", "HIGH"): "반발력은 높은 편입니다",
}


def render_exact_lab_claim(fact: PointEvidenceFact) -> str | None:
    """Render a RunRepeat metric without adding an inferred benefit or effect."""

    if fact.source != "RUNREPEAT" or fact.mode not in {"LAB", "CONFLICT"}:
        return None
    return _EXACT_LAB_CLAIM_TEXT.get((fact.signal.subject, fact.signal.value))


_STRENGTH_RANK: dict[EvidenceStrength, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}
_SIZE_UNIT_STEP_PATTERN = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*"
    r"(mm|cm|미리|밀리미터|센티|센티미터)\s*(?:정도\s*)?"
    r"(?:사이즈\s*)?업"
)
_SIZE_UNIT_ALIASES = {
    "미리": "mm",
    "밀리미터": "mm",
    "센티": "cm",
    "센티미터": "cm",
}


def consensus_size_unit_step(
    fact: PointEvidenceFact,
) -> tuple[str, str] | None:
    """Return an exact metric size step repeated by independent reviews."""

    if fact.signal.subject != "SIZE_OPTION" or fact.signal.value != "GENERIC_UP":
        return None
    exact_steps: dict[tuple[str, str], set[int]] = defaultdict(set)
    for review_id, evidence_text in zip(fact.review_ids, fact.evidence_texts):
        for value, raw_unit in _SIZE_UNIT_STEP_PATTERN.findall(
            normalize_review_text(evidence_text)
        ):
            exact_steps[(value, _SIZE_UNIT_ALIASES.get(raw_unit, raw_unit))].add(
                review_id
            )
    repeated_steps = [
        step for step, review_ids in exact_steps.items() if len(review_ids) >= 2
    ]
    return repeated_steps[0] if len(repeated_steps) == 1 else None


def underfoot_thickness_label(fact: PointEvidenceFact) -> str:
    """Preserve whether review evidence names an insole or the shoe bottom."""

    insole_mentions = sum(
        "깔창" in normalize_review_text(text) for text in fact.evidence_texts
    )
    sole_mentions = sum(
        bool(
            re.search(
                r"(?<!발)바닥|밑창|아웃솔",
                normalize_review_text(text),
            )
        )
        for text in fact.evidence_texts
    )
    return "깔창" if insole_mentions > sole_mentions else "바닥"


_OPPOSITE_VALUES = {
    "WIDTH_SPACE": {frozenset(("TIGHT", "ROOMY"))},
    "TOEBOX_SPACE": {frozenset(("TIGHT", "ROOMY"))},
    "HEEL_HOLD": {frozenset(("STABLE", "LOOSE"))},
    "HEEL_COMFORT": {
        frozenset(("COMFORTABLE", "RUBBING")),
        frozenset(("COMFORTABLE", "DISCOMFORT")),
    },
    "CUSHION_SOFTNESS": {frozenset(("SOFT", "FIRM"))},
    "CUSHION_QUALITY": {frozenset(("POSITIVE", "NEGATIVE"))},
    "SOLE_THICKNESS": {frozenset(("THIN", "THICK"))},
    "LONG_WEAR_COMFORT": {frozenset(("COMFORTABLE", "DISCOMFORT"))},
    "WEIGHT": {frozenset(("LIGHT", "HEAVY"))},
    "APPEARANCE": {frozenset(("POSITIVE", "NEGATIVE"))},
    "BREATHABILITY": {frozenset(("HIGH", "LOW"))},
    "SHOCK_ABSORPTION": {frozenset(("HIGH", "LOW"))},
}


def _comparable_subject(subject: str) -> str:
    # RunRepeat WIDTH_SPACE and review width wording describe the same bounded
    # product attribute.  Other lab concepts remain separate unless their
    # typed subject is already identical (for example CUSHION_SOFTNESS).
    return {
        "LAB_WIDTH_SPACE": "WIDTH_SPACE",
        "LAB_TOEBOX_SPACE": "TOEBOX_SPACE",
    }.get(subject, subject)


def _signals_conflict(left: PointSignal, right: PointSignal) -> bool:
    left_subject = _comparable_subject(left.subject)
    right_subject = _comparable_subject(right.subject)
    if left.category != right.category or left_subject != right_subject:
        return False
    if left.value == right.value:
        return {left.stance, right.stance} == {"POSITIVE", "NEGATIVE"}
    return frozenset((left.value, right.value)) in _OPPOSITE_VALUES.get(
        left_subject, set()
    )


def _canonical_review_pairs(
    review_pairs: Iterable[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Deduplicate repeated slots and legacy duplicate bodies without fake consensus."""

    text_by_id: dict[int, str] = {}
    rank_by_body: dict[str, int] = {}
    ids_by_body: dict[str, list[int]] = defaultdict(list)
    raw_text_by_body: dict[str, str] = {}
    for rank, (review_id, review_text) in enumerate(review_pairs):
        body = normalize_review_text(review_text)
        if not body:
            continue
        previous = text_by_id.get(review_id)
        if previous is not None and previous != body:
            raise ValueError(
                f"reviewId={review_id} is associated with more than one review body"
            )
        text_by_id[review_id] = body
        if review_id not in ids_by_body[body]:
            ids_by_body[body].append(review_id)
        rank_by_body.setdefault(body, rank)
        raw_text_by_body.setdefault(body, review_text)

    return [
        (min(ids_by_body[body]), raw_text_by_body[body])
        for body in sorted(rank_by_body, key=rank_by_body.__getitem__)
    ]


def _aggregate_review_facts(
    review_pairs: Iterable[tuple[int, str]],
) -> list[PointEvidenceFact]:
    evidence_by_signal: dict[PointSignal, dict[int, str]] = defaultdict(dict)
    for review_id, review_text in _canonical_review_pairs(review_pairs):
        seen_for_review: set[PointSignal] = set()
        for clause in extract_point_summary_evidence(review_text):
            for signal in classify_point_evidence(clause):
                if signal in seen_for_review:
                    continue
                seen_for_review.add(signal)
                evidence_by_signal[signal][review_id] = clause

    facts: list[PointEvidenceFact] = []
    for signal, evidence_by_review in sorted(
        evidence_by_signal.items(), key=lambda item: item[0]
    ):
        review_ids = tuple(evidence_by_review)
        mode: EvidenceMode = "CONSENSUS" if len(review_ids) >= 2 else "SINGLE"
        facts.append(
            PointEvidenceFact(
                evidence_id=(
                    f"MUSINSA:{signal.category}:{signal.subject}:"
                    f"{signal.value}:{signal.stance}"
                ),
                signal=signal,
                source="MUSINSA",
                mode=mode,
                review_ids=review_ids,
                evidence_texts=tuple(evidence_by_review.values()),
            )
        )

    resolved_facts = [
        replace(
            fact,
            mode="MIXED"
            if any(
                other is not fact
                and other.source == "MUSINSA"
                and _signals_conflict(fact.signal, other.signal)
                for other in facts
            )
            else fact.mode,
        )
        for fact in facts
    ]

    # If equally supported concrete size units disagree, neither is safe to
    # expose to Ollama as an independent product recommendation.  When one is
    # strictly stronger, keep only that unit and downgrade the weaker unit(s).
    concrete = [
        fact
        for fact in resolved_facts
        if fact.signal.subject == "SIZE_OPTION"
        and fact.signal.value in {"FULL_UP", "HALF_UP", "DOWN"}
        and fact.signal.stance == "POSITIVE"
        and fact.mode == "CONSENSUS"
    ]
    if len(concrete) >= 2:
        rank_by_id = {
            fact.evidence_id: (
                2 if fact.support_count >= 3 else 1,
                fact.support_count,
            )
            for fact in concrete
        }
        top_rank = max(rank_by_id.values())
        winners = {
            evidence_id
            for evidence_id, rank in rank_by_id.items()
            if rank == top_rank
        }
        trusted_id = next(iter(winners)) if len(winners) == 1 else None
        resolved_facts = [
            replace(fact, mode="MIXED")
            if fact.evidence_id in rank_by_id and fact.evidence_id != trusted_id
            else fact
            for fact in resolved_facts
        ]
    return resolved_facts


def _runrepeat_signal(characteristic_type: str, level: str) -> PointSignal:
    level_value = level.upper()
    mappings: dict[str, tuple[str, str, str]] = {
        "WIDTH_SPACE": (
            "WIDTH_FIT",
            "LAB_WIDTH_SPACE",
            {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"}[level_value],
        ),
        "TOEBOX_SPACE": (
            "WIDTH_FIT",
            "LAB_TOEBOX_SPACE",
            {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"}[level_value],
        ),
        "CUSHION": (
            "CUSHION_FEEL",
            "CUSHION_SOFTNESS",
            {"LOW": "FIRM", "MEDIUM": "BALANCED", "HIGH": "SOFT"}[level_value],
        ),
        "HEEL_HOLD": (
            "HEEL_FEEL",
            "HEEL_STRUCTURE",
            {"LOW": "FLEXIBLE", "MEDIUM": "BALANCED", "HIGH": "FIRM"}[level_value],
        ),
        "BREATHABILITY": ("OTHER", "BREATHABILITY", level_value),
        "SHOCK_ABSORPTION": ("CUSHION_FEEL", "SHOCK_ABSORPTION", level_value),
        "ENERGY_RETURN": ("OTHER", "ENERGY_RETURN", level_value),
    }
    category, subject, value = mappings[characteristic_type]
    return PointSignal(category, subject, value, "OBJECTIVE")


def _runrepeat_text(characteristic_type: str, level: str) -> str:
    level_text = {"LOW": "낮은 편", "MEDIUM": "보통 수준", "HIGH": "높은 편"}[level]
    names = {
        "CUSHION": "쿠션감",
        "SHOCK_ABSORPTION": "충격 완화",
        "ENERGY_RETURN": "반발력",
        "WIDTH_SPACE": "발볼 공간",
        "TOEBOX_SPACE": "앞코 공간",
        "HEEL_HOLD": "뒤꿈치 구조 강성",
        "BREATHABILITY": "통기성",
    }
    return f"RunRepeat 비교 특성에서 {names[characteristic_type]}은 {level_text}입니다."


def _runrepeat_facts(
    characteristics: ServerShoeCharacteristics | None,
) -> list[PointEvidenceFact]:
    if characteristics is None:
        return []
    facts: list[PointEvidenceFact] = []
    for item in characteristics.characteristics:
        if item.level is None:
            continue
        signal = _runrepeat_signal(item.type, item.level)
        facts.append(
            PointEvidenceFact(
                evidence_id=f"RUNREPEAT:{item.type}:{item.level}",
                signal=signal,
                source="RUNREPEAT",
                mode="LAB",
                review_ids=(),
                evidence_texts=(_runrepeat_text(item.type, item.level),),
            )
        )
    return facts


def prepare_point_evidence_facts(
    review_pairs: Iterable[tuple[int, str]],
    characteristics: ServerShoeCharacteristics | None = None,
) -> list[PointEvidenceFact]:
    review_facts = _aggregate_review_facts(review_pairs)
    lab_facts = _runrepeat_facts(characteristics)
    all_facts = [*review_facts, *lab_facts]
    return [
        replace(
            fact,
            mode="CONFLICT"
            if any(
                other.source != fact.source
                and _signals_conflict(fact.signal, other.signal)
                for other in all_facts
            )
            else fact.mode,
        )
        for fact in all_facts
    ]


def point_evidence_strength(
    fact: PointEvidenceFact,
    facts: Iterable[PointEvidenceFact],
) -> EvidenceStrength:
    """Return the claim strength without broadening the fact's exact concept.

    A single or contradictory review is never promoted.  Two independent
    MUSINSA bodies establish MEDIUM consensus, while three establish HIGH.
    Exact RunRepeat facts are MEDIUM on their own and become HIGH only when at
    least two independent MUSINSA reviews support the same subject and value.
    """

    if fact.signal.subject in {
        "PRODUCT_DETAIL",
        "MATERIAL_DETAIL",
        "HEEL_HEIGHT_DETAIL",
    }:
        return "LOW"
    if fact.signal.subject == "APPEARANCE" and (
        fact.signal.value != "POSITIVE" or fact.signal.stance != "POSITIVE"
    ):
        return "LOW"
    if fact.signal.subject == "OUTFIT" and fact.signal.stance != "POSITIVE":
        return "LOW"
    if fact.signal.subject == "SIZE_OPTION" and fact.signal.stance == "NEUTRAL":
        return "LOW"
    if fact.mode in {"SINGLE", "MIXED", "CONFLICT"}:
        return "LOW"

    all_facts = tuple(facts)
    corroborating_review = next(
        (
            other
            for other in all_facts
            if other.source == "MUSINSA"
            and other.mode == "CONSENSUS"
            and _comparable_subject(other.signal.subject)
            == _comparable_subject(fact.signal.subject)
            and other.signal.value == fact.signal.value
        ),
        None,
    )
    corroborating_lab = any(
        other.source == "RUNREPEAT"
        and other.mode == "LAB"
        and _comparable_subject(other.signal.subject)
        == _comparable_subject(fact.signal.subject)
        and other.signal.value == fact.signal.value
        for other in all_facts
    )

    if fact.mode == "LAB":
        return "HIGH" if corroborating_review is not None else "MEDIUM"
    if fact.mode == "CONSENSUS":
        if fact.support_count >= 3 or corroborating_lab:
            return "HIGH"
        return "MEDIUM"
    return "LOW"


def _has_minimum_strength(
    fact: PointEvidenceFact,
    facts: Iterable[PointEvidenceFact],
    minimum: EvidenceStrength = "MEDIUM",
) -> bool:
    return _STRENGTH_RANK[point_evidence_strength(fact, facts)] >= _STRENGTH_RANK[minimum]


def format_point_evidence_facts(facts: list[PointEvidenceFact]) -> str:
    if not facts:
        return "- 근거 없음"
    lines: list[str] = []
    for fact in facts:
        strength = point_evidence_strength(fact, facts)
        review_ids = ",".join(str(value) for value in fact.review_ids) or "-"
        expression = {
            "SINGLE": "LOW: 단일 근거이므로 pointSummary에서 제외",
            "CONSENSUS": "서로 다른 근거 리뷰 2개 이상: 제품 특성형 표현 가능",
            "MIXED": "LOW: 상반된 리뷰 근거이므로 pointSummary에서 제외",
            "LAB": "RunRepeat 객관 특성: 해당 특성명과 level 의미만 사용",
            "CONFLICT": "LOW: 리뷰와 RunRepeat 충돌이므로 pointSummary에서 제외",
        }[fact.mode]
        texts = " | ".join(f'"{text}"' for text in fact.evidence_texts[:3])
        lines.append(
            f"- evidenceId={fact.evidence_id}; category={fact.signal.category}; "
            f"signal={fact.signal.subject}:{fact.signal.value}:{fact.signal.stance}; "
            f"source={fact.source}; mode={fact.mode}; strength={strength}; "
            f"supportCount={fact.support_count}; reviewIds={review_ids}; "
            f"rule={expression}; evidence={texts}"
        )
    return "\n".join(lines)


def _fact(
    facts: list[PointEvidenceFact],
    *,
    subject: str,
    value: str | None = None,
    source: EvidenceSource | None = None,
    minimum_strength: EvidenceStrength = "MEDIUM",
) -> PointEvidenceFact | None:
    candidates = [
        fact
        for fact in facts
        if fact.signal.subject == subject
        and (value is None or fact.signal.value == value)
        and (source is None or fact.source == source)
        and _has_minimum_strength(fact, facts, minimum_strength)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            _STRENGTH_RANK[point_evidence_strength(item, facts)],
            item.mode == "CONSENSUS",
            item.support_count,
        ),
    )


def render_product_point_summary(
    shoe_name: str,
    facts: list[PointEvidenceFact],
    *,
    max_claims: int | None = 4,
    include_attributed_single_evidence: bool = False,
) -> list[PointSummaryClaim]:
    """Render fit-first MD-style copy from typed evidence.

    Product-level claims still require MEDIUM/HIGH evidence.  Sparse summaries
    may additionally expose a small number of non-conflicting SINGLE review
    observations when ``include_attributed_single_evidence`` is enabled.  Those
    observations are explicitly labelled as individual-review opinions, so
    they cannot be mistaken for product-wide consensus.
    """

    del shoe_name  # The UI already presents it; product copy starts with the fit.
    claims: list[PointSummaryClaim] = []

    def add_claim(text: str, evidence: Iterable[str]) -> None:
        if max_claims is not None and len(claims) >= max_claims:
            return
        claims.append(
            PointSummaryClaim(
                text=text.rstrip(". "),
                evidence_ids=tuple(dict.fromkeys(evidence)),
            )
        )

    def join_descriptions(parts: list[str]) -> str:
        if not parts:
            return ""
        if len(parts) == 1:
            return f"{parts[0]}입니다"
        connected: list[str] = []
        for part in parts[:-1]:
            if part.endswith(("편", "핏", "수준", "특징")):
                connected.append(f"{part}이며")
            else:
                connected.append(f"{part}이고")
        return f"{', '.join(connected)}, {parts[-1]}입니다"

    def connect_sentences(first: str, second: str) -> str:
        """Join two complete Korean sentences without producing ``합니다고``."""

        if first.endswith("입니다"):
            return f"{first[:-3]}이며, {second}"
        if first.endswith("합니다"):
            return f"{first[:-3]}하고, {second}"
        if first.endswith("습니다"):
            return f"{first[:-3]}고, {second}"
        return f"{first}, {second}"

    # 1) FIT: WIDTH -> SIZE -> INSTEP/condition.  Concrete sizing advice is
    # impossible here unless at least two independent reviews agree.
    width_tight = _fact(
        facts, subject="WIDTH_SPACE", value="TIGHT", source="MUSINSA"
    )
    width_roomy = _fact(
        facts, subject="WIDTH_SPACE", value="ROOMY", source="MUSINSA"
    )
    toebox_tight = _fact(
        facts, subject="TOEBOX_SPACE", value="TIGHT", source="MUSINSA"
    )
    toebox_roomy = _fact(
        facts, subject="TOEBOX_SPACE", value="ROOMY", source="MUSINSA"
    )

    def choose_size_fact(
        values: tuple[str, ...], stance: str
    ) -> PointEvidenceFact | None:
        candidates = [
            fact
            for fact in facts
            if fact.source == "MUSINSA"
            and fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value in values
            and fact.signal.stance == stance
            and _has_minimum_strength(fact, facts)
        ]
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda fact: (
                _STRENGTH_RANK[point_evidence_strength(fact, facts)],
                fact.support_count,
            ),
            reverse=True,
        )
        top_rank = (
            _STRENGTH_RANK[point_evidence_strength(ranked[0], facts)],
            ranked[0].support_count,
        )
        top = [
            fact
            for fact in ranked
            if (
                _STRENGTH_RANK[point_evidence_strength(fact, facts)],
                fact.support_count,
            )
            == top_rank
        ]
        return top[0] if len(top) == 1 else None

    exact_size_fact = choose_size_fact(("FULL_UP", "HALF_UP", "DOWN"), "POSITIVE")
    if exact_size_fact is None:
        exact_size_fact = choose_size_fact(("GENERIC_UP",), "POSITIVE")
    failed_size_fact = choose_size_fact(
        ("FULL_UP", "HALF_UP", "GENERIC_UP", "DOWN"), "NEGATIVE"
    )
    true_size = _fact(
        facts, subject="SIZE_OPTION", value="TRUE_SIZE", source="MUSINSA"
    )
    true_size_positive = (
        true_size if true_size and true_size.signal.stance == "POSITIVE" else None
    )
    true_size_failed = (
        true_size if true_size and true_size.signal.stance == "NEGATIVE" else None
    )
    exact_fit = _fact(
        facts, subject="OVERALL_FIT", value="EXACT", source="MUSINSA"
    )
    foot_condition = _fact(
        facts,
        subject="FOOT_CONDITION",
        value="WIDE_OR_HIGH_INSTEP",
        source="MUSINSA",
    )
    size_wording = {
        "FULL_UP": "한 사이즈 업",
        "HALF_UP": "반 사이즈 업",
        "GENERIC_UP": "사이즈 업",
        "DOWN": "사이즈 다운",
    }

    def size_wording_for_fact(fact: PointEvidenceFact) -> str:
        if fact.signal.value != "GENERIC_UP":
            return size_wording[fact.signal.value]
        if exact_step := consensus_size_unit_step(fact):
            value, unit = exact_step
            return f"{value}{unit} 사이즈 업"
        return size_wording[fact.signal.value]

    def negative_size_outcome(
        fact: PointEvidenceFact | None,
    ) -> Literal["SMALL", "LARGE"] | None:
        if fact is None:
            return None
        small_support = sum(
            any(
                marker in normalize_review_text(text)
                for marker in (
                    "여전히 작",
                    "그래도 작",
                    "작게 느껴",
                    "작아요",
                    "작습니다",
                    "작",
                    "좁",
                    "끼",
                    "눌",
                    "답답",
                    "까졌",
                )
            )
            for text in fact.evidence_texts
        )
        large_support = sum(
            any(
                marker in normalize_review_text(text)
                for marker in (
                    "너무 크",
                    "크게 느껴",
                    "사이즈가 크",
                    "사이즈는 크",
                    "헐렁",
                    "남아요",
                    "남습니다",
                )
            )
            for text in fact.evidence_texts
        )
        if small_support >= 2 and large_support < 2:
            return "SMALL"
        if large_support >= 2 and small_support < 2:
            return "LARGE"
        return None

    def failed_size_description(fact: PointEvidenceFact) -> str | None:
        prefix = {
            "FULL_UP": "한 사이즈 업에서도",
            "HALF_UP": "반 사이즈 업에서도",
            "DOWN": "사이즈 다운에서도",
            "TRUE_SIZE": "정사이즈는",
        }.get(fact.signal.value)
        if prefix is None:
            prefix = f"{size_wording_for_fact(fact)} 후에도"
        outcome = negative_size_outcome(fact)
        if outcome == "SMALL":
            return f"{prefix} 여유가 부족하게 느껴지는 편입니다"
        if outcome == "LARGE":
            return f"{prefix} 크게 느껴지는 편입니다"
        return None
    size_text = size_wording_for_fact(exact_size_fact) if exact_size_fact else None
    condition_text_by_id = (
        dict(zip(foot_condition.review_ids, foot_condition.evidence_texts))
        if foot_condition
        else {}
    )
    wide_condition_ids = {
        review_id
        for review_id, evidence_text in condition_text_by_id.items()
        if "발볼러" in normalize_review_text(evidence_text)
        or re.search(
            r"발볼.{0,10}(?:넓(?:으면|다면|은\s*(?:사람|분))|크(?:면|다면|는\s*(?:사람|분))|있(?:으면|다면|어서|는\s*(?:경우|편|사람|분)))",
            normalize_review_text(evidence_text),
        )
    }
    instep_condition_ids = {
        review_id
        for review_id, evidence_text in condition_text_by_id.items()
        if re.search(
            r"발등.{0,10}(?:높(?:으면|다면|은\s*(?:사람|분))|있(?:으면|다면|어서|는\s*(?:경우|편|사람|분)))",
            normalize_review_text(evidence_text),
        )
    }
    exact_size_ids = set(exact_size_fact.review_ids) if exact_size_fact else set()
    wide_size_support = exact_size_ids & wide_condition_ids
    instep_size_support = exact_size_ids & instep_condition_ids
    if len(wide_size_support) >= 2 and len(instep_size_support) >= 2:
        sizing_condition = "발볼이 넓거나 발등이 높은 편"
        condition_overlap = wide_size_support | instep_size_support
    elif len(wide_size_support) >= 2:
        sizing_condition = "발볼이 넓은 편"
        condition_overlap = wide_size_support
    elif len(instep_size_support) >= 2:
        sizing_condition = "발등이 높은 편"
        condition_overlap = instep_size_support
    else:
        sizing_condition = None
        condition_overlap = set()
    conditional_sizing = sizing_condition is not None
    fit_parts: list[str] = []
    fit_evidence: list[str] = []
    lab_width = _fact(facts, subject="LAB_WIDTH_SPACE", source="RUNREPEAT")
    lab_toebox = _fact(facts, subject="LAB_TOEBOX_SPACE", source="RUNREPEAT")
    def add_lab_fit_claim(fact: PointEvidenceFact | None) -> None:
        if fact and (text := render_exact_lab_claim(fact)):
            add_claim(
                text,
                (fact.evidence_id,),
            )

    review_width_present = width_tight is not None or width_roomy is not None
    if not review_width_present:
        add_lab_fit_claim(lab_width)
        if toebox_tight:
            add_claim(
                "앞코 공간은 비교적 좁게 느껴지는 편입니다",
                (toebox_tight.evidence_id,),
            )
        elif toebox_roomy:
            add_claim(
                "앞코 공간은 비교적 여유 있게 느껴지는 편입니다",
                (toebox_roomy.evidence_id,),
            )
        else:
            add_lab_fit_claim(lab_toebox)

    if width_tight:
        if exact_size_fact and size_text and conditional_sizing:
            fit_parts.append(
                "발볼을 비교적 슬림하게 잡는 핏이라 "
                f"{sizing_condition}이라면 {size_text}을 고려할 만합니다"
            )
            fit_evidence.extend(
                (
                    width_tight.evidence_id,
                    exact_size_fact.evidence_id,
                    foot_condition.evidence_id,
                )
            )
        elif exact_size_fact and size_text:
            fit_parts.append(
                "발볼은 비교적 슬림하게 맞는 편이며, "
                f"사이즈 선택에서는 {size_text}을 고려할 만합니다"
            )
            fit_evidence.extend((width_tight.evidence_id, exact_size_fact.evidence_id))
        else:
            fit_parts.append("발볼은 비교적 슬림하게 맞는 핏입니다")
            fit_evidence.append(width_tight.evidence_id)
    elif width_roomy:
        fit_parts.append("발볼은 비교적 여유 있게 나온 편입니다")
        fit_evidence.append(width_roomy.evidence_id)
    elif exact_size_fact and size_text:
        if conditional_sizing:
            fit_parts.append(
                f"{sizing_condition}이라면 {size_text}을 고려할 만합니다"
            )
            fit_evidence.extend(
                (exact_size_fact.evidence_id, foot_condition.evidence_id)
            )
        else:
            fit_parts.append(f"사이즈 선택에서는 {size_text}을 고려할 만합니다")
            fit_evidence.append(exact_size_fact.evidence_id)
    if exact_size_fact is None and failed_size_fact:
        failed_description = failed_size_description(failed_size_fact)
        if failed_description:
            fit_parts.append(failed_description)
            fit_evidence.append(failed_size_fact.evidence_id)
    if exact_size_fact is None and true_size_failed:
        failed_description = failed_size_description(true_size_failed)
        if failed_description:
            fit_parts.append(failed_description)
            fit_evidence.append(true_size_failed.evidence_id)

    if (
        true_size_positive
        and failed_size_fact is None
        and (exact_size_fact is None or conditional_sizing)
    ):
        exact_fit_overlap = (
            set(true_size_positive.review_ids) & set(exact_fit.review_ids)
            if exact_fit
            else set()
        )
        if exact_fit and len(exact_fit_overlap) >= 2:
            fit_parts.append("정사이즈는 전체적으로 딱 맞는 핏입니다")
            fit_evidence.extend(
                (true_size_positive.evidence_id, exact_fit.evidence_id)
            )
        else:
            fit_parts.append("정사이즈 선택이 잘 맞는 편입니다")
            fit_evidence.append(true_size_positive.evidence_id)
    elif exact_fit:
        fit_parts.append("전체적으로 딱 맞는 핏에 가깝습니다")
        fit_evidence.append(exact_fit.evidence_id)

    if fit_parts:
        fit_text = fit_parts[0]
        for fit_part in fit_parts[1:]:
            fit_text = connect_sentences(fit_text, fit_part)
        add_claim(fit_text, fit_evidence)
    if (
        exact_size_fact is not None
        and failed_size_fact is not None
        and failed_size_fact.signal.value != exact_size_fact.signal.value
    ):
        # Preserve a separately grounded outcome instead of letting one sizing
        # consensus silently hide another.  Keeping it in its own claim also
        # prevents positive and negative stances from borrowing each other's
        # predicates during validation.
        failed_description = failed_size_description(failed_size_fact)
        if failed_description:
            add_claim(failed_description, (failed_size_fact.evidence_id,))
    if review_width_present:
        if toebox_tight:
            add_claim(
                "앞코 공간은 비교적 좁게 느껴지는 편입니다",
                (toebox_tight.evidence_id,),
            )
        elif toebox_roomy:
            add_claim(
                "앞코 공간은 비교적 여유 있게 느껴지는 편입니다",
                (toebox_roomy.evidence_id,),
            )
        else:
            add_lab_fit_claim(lab_toebox)

    # 2) Structure and underfoot feel.  Softness, sole thickness and shock
    # absorption remain separate typed concepts.
    heel_stable = _fact(
        facts, subject="HEEL_HOLD", value="STABLE", source="MUSINSA"
    )
    heel_loose = _fact(
        facts, subject="HEEL_HOLD", value="LOOSE", source="MUSINSA"
    )
    heel_rubbing = _fact(
        facts, subject="HEEL_COMFORT", value="RUBBING", source="MUSINSA"
    )
    heel_discomfort = _fact(
        facts, subject="HEEL_COMFORT", value="DISCOMFORT", source="MUSINSA"
    )
    heel_comfortable = _fact(
        facts, subject="HEEL_COMFORT", value="COMFORTABLE", source="MUSINSA"
    )
    lab_cushion = _fact(
        facts, subject="CUSHION_SOFTNESS", source="RUNREPEAT"
    )
    review_soft = _fact(
        facts, subject="CUSHION_SOFTNESS", value="SOFT", source="MUSINSA"
    )
    review_firm = _fact(
        facts, subject="CUSHION_SOFTNESS", value="FIRM", source="MUSINSA"
    )
    cushion_positive = _fact(
        facts, subject="CUSHION_QUALITY", value="POSITIVE", source="MUSINSA"
    )
    cushion_negative = _fact(
        facts, subject="CUSHION_QUALITY", value="NEGATIVE", source="MUSINSA"
    )
    sole_thin = _fact(
        facts, subject="SOLE_THICKNESS", value="THIN", source="MUSINSA"
    )
    long_wear_discomfort = _fact(
        facts,
        subject="LONG_WEAR_COMFORT",
        value="DISCOMFORT",
        source="MUSINSA",
    )
    causal_thin_ids: set[int] = set()
    if sole_thin and long_wear_discomfort:
        thin_text_by_id = dict(zip(sole_thin.review_ids, sole_thin.evidence_texts))
        wear_text_by_id = dict(
            zip(long_wear_discomfort.review_ids, long_wear_discomfort.evidence_texts)
        )
        for review_id in set(thin_text_by_id) & set(wear_text_by_id):
            thin_text = normalize_review_text(thin_text_by_id[review_id])
            wear_text = normalize_review_text(wear_text_by_id[review_id])
            if thin_text != wear_text:
                continue
            if re.search(
                r"(?:바닥|밑창|깔창).{0,8}"
                r"(?:얇아서|얇기\s*때문|얇은\s*탓|얇아\s+)"
                r".{0,20}(?:오래|장시간|종일|발바닥)",
                thin_text,
            ):
                causal_thin_ids.add(review_id)
    thin_long_wear_consensus = len(causal_thin_ids) >= 2
    comfort_parts: list[str] = []
    comfort_evidence: list[str] = []
    if heel_stable:
        comfort_parts.append("뒤꿈치를 비교적 안정적으로 잡아주는 편")
        comfort_evidence.append(heel_stable.evidence_id)
    elif heel_loose:
        comfort_parts.append("뒤꿈치는 다소 헐렁하게 느껴지는 편")
        comfort_evidence.append(heel_loose.evidence_id)
    if heel_rubbing:
        comfort_parts.append("뒤꿈치 주변은 마찰이 느껴질 수 있는 편")
        comfort_evidence.append(heel_rubbing.evidence_id)
    elif heel_discomfort:
        comfort_parts.append("뒤꿈치 주변은 다소 불편하게 느껴질 수 있는 편")
        comfort_evidence.append(heel_discomfort.evidence_id)
    elif heel_comfortable:
        comfort_parts.append("뒤꿈치 주변은 비교적 편안하게 느껴지는 편")
        comfort_evidence.append(heel_comfortable.evidence_id)
    cushion_fact = review_soft or review_firm
    if cushion_fact:
        cushion_phrase = {
            "SOFT": "쿠션은 비교적 부드럽게 느껴지는 편",
            "BALANCED": "쿠션감은 보통 수준",
            "FIRM": "쿠션은 비교적 단단하게 느껴지는 편",
        }[cushion_fact.signal.value]
        comfort_parts.append(cushion_phrase)
        comfort_evidence.append(cushion_fact.evidence_id)
    elif cushion_positive:
        comfort_parts.append("쿠션감에 대한 만족도가 높은 편")
        comfort_evidence.append(cushion_positive.evidence_id)
    elif cushion_negative:
        comfort_parts.append("쿠션감은 다소 아쉽게 느껴지는 편")
        comfort_evidence.append(cushion_negative.evidence_id)
    if sole_thin and not thin_long_wear_consensus:
        comfort_parts.append(
            f"{underfoot_thickness_label(sole_thin)}은 얇게 느껴지는 편"
        )
        comfort_evidence.append(sole_thin.evidence_id)
    if comfort_parts:
        add_claim(join_descriptions(comfort_parts), comfort_evidence)

    # 3) Longer wear, overall comfort and physical weight.
    long_wear_comfort = _fact(
        facts,
        subject="LONG_WEAR_COMFORT",
        value="COMFORTABLE",
        source="MUSINSA",
    )
    wear_comfort = _fact(
        facts, subject="WEAR_COMFORT", value="COMFORTABLE", source="MUSINSA"
    )
    wear_discomfort = _fact(
        facts, subject="WEAR_COMFORT", value="DISCOMFORT", source="MUSINSA"
    )
    review_breathability_high = _fact(
        facts, subject="BREATHABILITY", value="HIGH", source="MUSINSA"
    )
    review_breathability_low = _fact(
        facts, subject="BREATHABILITY", value="LOW", source="MUSINSA"
    )
    heavy = _fact(facts, subject="WEIGHT", value="HEAVY", source="MUSINSA")
    light = _fact(facts, subject="WEIGHT", value="LIGHT", source="MUSINSA")
    use_parts: list[str] = []
    use_evidence: list[str] = []
    if long_wear_discomfort:
        if thin_long_wear_consensus:
            thickness_label = underfoot_thickness_label(sole_thin)
            use_parts.append(
                f"{thickness_label}이 얇게 느껴지는 편이라 오래 착용할 때 "
                "발바닥 부담이 생길 수 있습니다"
            )
            use_evidence.extend(
                (sole_thin.evidence_id, long_wear_discomfort.evidence_id)
            )
        else:
            use_parts.append("장시간 착용에서는 불편하게 느껴질 수 있습니다")
            use_evidence.append(long_wear_discomfort.evidence_id)
    elif long_wear_comfort:
        use_parts.append("장시간 착용에서도 비교적 편안하게 느껴지는 편입니다")
        use_evidence.append(long_wear_comfort.evidence_id)
    elif wear_comfort:
        use_parts.append("전반적인 착화감은 편하게 느껴지는 편입니다")
        use_evidence.append(wear_comfort.evidence_id)
    elif wear_discomfort:
        use_parts.append("전반적인 착화감은 다소 불편하게 느껴지는 편입니다")
        use_evidence.append(wear_discomfort.evidence_id)
    if review_breathability_high:
        use_parts.append("통풍과 통기성은 좋게 느껴지는 편입니다")
        use_evidence.append(review_breathability_high.evidence_id)
    elif review_breathability_low:
        use_parts.append("신발 내부는 다소 답답하게 느껴질 수 있습니다")
        use_evidence.append(review_breathability_low.evidence_id)
    if heavy:
        use_parts.append("착화 시 다소 무게감이 느껴지는 편입니다")
        use_evidence.append(heavy.evidence_id)
    elif light:
        use_parts.append("착화 시 가볍게 느껴지는 편입니다")
        use_evidence.append(light.evidence_id)
    if use_parts:
        use_text = use_parts[0]
        for use_part in use_parts[1:]:
            use_text = connect_sentences(use_text, use_part)
        add_claim(use_text, use_evidence)

    # RunRepeat facts are intentionally rendered as one exact metric-and-level
    # sentence.  Benefits such as joint protection, cool summer wear or faster
    # running are not implied by the metric and therefore cannot be appended.
    standalone_lab_facts = [
        fact
        for subject in (
            "CUSHION_SOFTNESS",
            "HEEL_STRUCTURE",
            "SHOCK_ABSORPTION",
            "BREATHABILITY",
            "ENERGY_RETURN",
        )
        if (fact := _fact(facts, subject=subject, source="RUNREPEAT")) is not None
        and not (
            subject == "CUSHION_SOFTNESS"
            and (review_soft is not None or review_firm is not None)
        )
        and not (
            subject == "BREATHABILITY"
            and (
                review_breathability_high is not None
                or review_breathability_low is not None
            )
        )
    ]
    for lab_fact in standalone_lab_facts:
        if text := render_exact_lab_claim(lab_fact):
            add_claim(text, (lab_fact.evidence_id,))

    # 4) Appearance and styling are allowed only with independent consensus.
    design = _fact(
        facts, subject="APPEARANCE", value="POSITIVE", source="MUSINSA"
    )
    silhouette = _fact(
        facts, subject="SILHOUETTE", value="SLIM", source="MUSINSA"
    )
    low_profile = _fact(
        facts, subject="PROFILE", value="LOW", source="MUSINSA"
    )
    eligible_outfits = [
        fact
        for fact in facts
        if fact.source == "MUSINSA"
        and fact.signal.subject == "OUTFIT"
        and fact.signal.stance == "POSITIVE"
        and _has_minimum_strength(fact, facts)
    ]
    outfit_by_value = {fact.signal.value: fact for fact in eligible_outfits}
    style_parts: list[str] = []
    style_evidence: list[str] = []
    if low_profile and silhouette:
        style_parts.append("로우하고 날렵한 실루엣이 특징")
        style_evidence.extend((low_profile.evidence_id, silhouette.evidence_id))
    elif low_profile:
        style_parts.append("로우 프로파일이 특징")
        style_evidence.append(low_profile.evidence_id)
    elif silhouette:
        style_parts.append("날렵한 실루엣이 특징")
        style_evidence.append(silhouette.evidence_id)

    if design:
        color_mentions = sum(
            any(marker in normalize_review_text(text) for marker in ("검정", "검은색", "블랙"))
            for text in design.evidence_texts
        )
        style_parts.append(
            "검정 색감과 디자인에 대한 만족도가 높은 편"
            if color_mentions >= 2
            else "디자인에 대한 만족도가 높은 편"
        )
        style_evidence.append(design.evidence_id)

    specific_names = {
        "JEANS": "청바지",
        "SLACKS": "슬랙스",
        "SKIRT": "스커트",
        "WIDE_PANTS": "와이드 팬츠",
    }
    specific_outfits = [
        (value, outfit_by_value[value])
        for value in specific_names
        if value in outfit_by_value
    ]
    general_outfit = outfit_by_value.get("GENERAL")
    daily_outfit = outfit_by_value.get("DAILY")
    if general_outfit:
        style_parts.append("다양한 스타일과 잘 어울리는 편")
        style_evidence.append(general_outfit.evidence_id)
    elif len(specific_outfits) >= 2:
        (first_value, first_fact), (second_value, second_fact) = specific_outfits[:2]
        style_parts.append(
            f"{specific_names[first_value]}와 {specific_names[second_value]} 등 "
            "다양한 스타일과 잘 어울리는 편"
        )
        style_evidence.extend((first_fact.evidence_id, second_fact.evidence_id))
    elif specific_outfits:
        value, outfit = specific_outfits[0]
        style_parts.append(f"{specific_names[value]}와 잘 어울리는 편")
        style_evidence.append(outfit.evidence_id)
    elif daily_outfit:
        style_parts.append("데일리로 활용하기 좋은 편")
        style_evidence.append(daily_outfit.evidence_id)
    if style_parts:
        add_claim(join_descriptions(style_parts), style_evidence)

    if include_attributed_single_evidence and claims and len(claims) < 3:
        # Keep sparse product copy useful without promoting one review into a
        # general product property.  Mixed/conflicting facts and sizing advice
        # are deliberately excluded because even attributed wording would be
        # too easy to misread as a purchase recommendation.
        single_observation_wording = {
            ("LONG_WEAR_COMFORT", "COMFORTABLE"): "하루 종일 착용해도 피로가 적었다는 의견",
            ("LONG_WEAR_COMFORT", "DISCOMFORT"): "오래 착용했을 때 불편했다는 의견",
            ("WIDTH_SPACE", "ROOMY"): "발볼이 넓게 느껴졌다는 의견",
            ("WIDTH_SPACE", "TIGHT"): "발볼이 좁게 느껴졌다는 의견",
            ("TOEBOX_SPACE", "ROOMY"): "앞코 공간이 여유롭게 느껴졌다는 의견",
            ("TOEBOX_SPACE", "TIGHT"): "앞코 공간이 좁게 느껴졌다는 의견",
            ("WEIGHT", "LIGHT"): "무게가 가볍게 느껴졌다는 의견",
            ("WEIGHT", "HEAVY"): "무게감이 느껴졌다는 의견",
            ("BREATHABILITY", "HIGH"): "통풍과 통기성이 좋았다는 의견",
            ("BREATHABILITY", "LOW"): "신발 내부가 답답하게 느껴졌다는 의견",
            ("CUSHION_QUALITY", "POSITIVE"): "쿠션감이 만족스러웠다는 의견",
            ("CUSHION_QUALITY", "NEGATIVE"): "쿠션감이 아쉬웠다는 의견",
        }
        observations: list[tuple[PointEvidenceFact, str]] = []
        for fact in facts:
            wording = single_observation_wording.get(
                (fact.signal.subject, fact.signal.value)
            )
            if fact.source != "MUSINSA" or fact.mode != "SINGLE" or wording is None:
                continue
            observations.append((fact, wording))
            if len(observations) == 3:
                break
        if observations:
            observation_text = ", ".join(wording for _, wording in observations)
            add_claim(
                "개별 리뷰에서는 "
                f"{observation_text}이 각각 확인되지만, 반복 근거가 아니므로 "
                "착용자에 따라 다르게 느껴질 수 있습니다",
                (fact.evidence_id for fact, _ in observations),
            )

    if not claims:
        claims.append(
            PointSummaryClaim(
                text="핏과 착화감을 상품 특성으로 일반화할 만큼 일관된 근거가 아직 부족합니다",
                evidence_ids=(),
            )
        )
    return claims if max_claims is None else claims[:max_claims]


def join_point_summary_claims(claims: list[PointSummaryClaim]) -> str:
    return " ".join(f"{claim.text.rstrip('.')}." for claim in claims)


_POINT_CLAIM_DOMAIN_BY_CATEGORY = {
    "SIZE_FIT": "FIT",
    "WIDTH_FIT": "FIT",
    "HEEL_FEEL": "COMFORT",
    "CUSHION_FEEL": "COMFORT",
    "LONG_WEAR": "COMFORT",
    "WEIGHT_FEEL": "COMFORT",
    "OTHER": "COMFORT",
    "DESIGN": "STYLE",
    "STYLING": "STYLE",
}


def _claim_domain(claim: PointSummaryClaim) -> str | None:
    domains: set[str] = set()
    for evidence_id in claim.evidence_ids:
        parts = evidence_id.split(":")
        if len(parts) < 2 or parts[0] != "MUSINSA":
            continue
        if domain := _POINT_CLAIM_DOMAIN_BY_CATEGORY.get(parts[1]):
            domains.add(domain)
    return next(iter(domains)) if len(domains) == 1 else None


def _connect_point_claim_text(first: str, second: str) -> str:
    first = first.rstrip(". ")
    second = second.rstrip(". ")
    natural_connector = {
        "쿠션은 비교적 부드럽게 느껴지는 편입니다": "쿠션이 비교적 부드럽고",
        "쿠션은 비교적 단단하게 느껴지는 편입니다": "쿠션이 비교적 단단하고",
        "쿠션감에 대한 만족도가 높은 편입니다": "쿠션감에 대한 만족도가 높고",
        "쿠션감은 다소 아쉽게 느껴지는 편입니다": "쿠션감이 다소 아쉽고",
        "장시간 착용에서도 비교적 편안하게 느껴지는 편입니다": "장시간 착용에도 비교적 편안하고",
        "전반적인 착화감은 편하게 느껴지는 편입니다": "전반적인 착화감이 편안하고",
    }.get(first)
    if natural_connector is not None:
        first = natural_connector
    elif first.endswith("입니다"):
        first = f"{first[:-3]}이며"
    elif first.endswith("합니다"):
        first = f"{first[:-3]}하고"
    elif first.endswith("습니다"):
        first = f"{first[:-3]}고"
    else:
        first = f"{first}이고"
    second = second.replace(
        "장시간 착용에서도 비교적 편안하게 느껴지는 편이며, ",
        "장시간 착용에도 편안하며, ",
    ).replace(
        "장시간 착용에서도 비교적 편안하게 느껴지는 편입니다",
        "장시간 착용에도 비교적 편안한 편입니다",
    )
    if "착화 시 가볍게 느껴지는 편" in second:
        second = second.replace("착화 시 가볍게", "무게도 가볍게")
    return f"{first}, {second}"


def _merge_adjacent_point_claims(
    claims: list[PointSummaryClaim], first_index: int
) -> list[PointSummaryClaim]:
    first = claims[first_index]
    second = claims[first_index + 1]
    merged = PointSummaryClaim(
        text=_connect_point_claim_text(first.text, second.text),
        evidence_ids=tuple(dict.fromkeys((*first.evidence_ids, *second.evidence_ids))),
    )
    return [*claims[:first_index], merged, *claims[first_index + 2 :]]


def compose_product_point_summary(claims: list[PointSummaryClaim]) -> str:
    """Compose review-only buyer guidance in two or three natural sentences.

    The deterministic recommendation batch uses this formatter after extracting
    consensus facts from the shoe's complete review set. It never fills missing
    product evidence with the current user's foot measurements.
    """

    grounded = [claim for claim in claims if claim.evidence_ids]
    if not grounded:
        return (
            "현재 수집된 리뷰만으로는 발볼과 사이즈감을 일관된 제품 특성으로 "
            "정리하기 어렵습니다. 쿠션·무게·장시간 착용감과 활용도도 반복 근거가 "
            "충분하지 않아 개별 착화 후기를 함께 확인해 주세요."
        )

    compacted = list(grounded)
    while len(compacted) > 3:
        merge_index = next(
            (
                index
                for index in range(len(compacted) - 1)
                if _claim_domain(compacted[index]) is not None
                and _claim_domain(compacted[index])
                == _claim_domain(compacted[index + 1])
            ),
            1,
        )
        compacted = _merge_adjacent_point_claims(compacted, merge_index)

    if len(compacted) == 3:
        same_domain_pair = next(
            (
                index
                for index in range(2)
                if _claim_domain(compacted[index]) is not None
                and _claim_domain(compacted[index])
                == _claim_domain(compacted[index + 1])
            ),
            None,
        )
        if same_domain_pair is not None:
            compacted = _merge_adjacent_point_claims(
                compacted, same_domain_pair
            )

    summary = join_point_summary_claims(compacted)
    return summary
