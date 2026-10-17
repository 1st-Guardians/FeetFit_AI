from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from app.core.config import settings
from app.schemas.shoe_server import ServerShoe, ServerUserFootState
from app.services.shoe.shoe_fit_policy import AreaScore
from app.services.shoe.shoe_review_evidence_policy import (
    render_reason_summary,
    select_review_evidence,
)


_REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")


class ReviewCandidateLike(Protocol):
    review_id: int
    sentence: str


def _maximum(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _width_ratio(state: ServerUserFootState) -> float | None:
    daily = state.daily_foot_analysis
    if daily is None:
        return None
    ratios = [
        float(width) / float(length)
        for width, length in (
            (daily.left_foot_width_mm, daily.measured_left_foot_size_mm),
            (daily.right_foot_width_mm, daily.measured_right_foot_size_mm),
        )
        if width is not None and length not in {None, 0.0}
    ]
    return sum(ratios) / len(ratios) if ratios else None


def _pressure_ratio(state: ServerUserFootState, attribute: str) -> float | None:
    values = [
        getattr(analysis, attribute)
        for analysis in state.static_pressure_analyses
        if getattr(analysis, attribute) is not None
    ]
    return _maximum(values)


def _measurement_sentence(
    reason_type: str,
    state: ServerUserFootState,
) -> str | None:
    daily = state.daily_foot_analysis
    hallux = state.hallux_valgus_analysis
    tinea = state.tina_pedis_analysis

    if reason_type == "FOREFOOT":
        clauses: list[str] = []
        ratio = _width_ratio(state)
        if ratio is not None and ratio >= settings.shoe_forefoot_width_ratio_neutral:
            clauses.append("발볼 여유가 더 필요합니다")
        max_hallux = _maximum(
            [
                hallux.left_toe_angle_degree if hallux else None,
                hallux.right_toe_angle_degree if hallux else None,
            ]
        )
        if (
            max_hallux is not None
            and max_hallux >= settings.shoe_hallux_angle_neutral_degree
        ):
            clauses.append("앞코 압박을 줄일 여유가 필요합니다")
        if clauses:
            return _join_measurement_clauses(clauses)
        if ratio is not None or max_hallux is not None:
            return "측정 결과 발볼과 앞코에 추가 여유가 필요한 신호는 크지 않습니다."
        return None

    if reason_type == "HEEL":
        clauses = []
        rearfoot_pressure = _pressure_ratio(state, "rearfoot_pressure_ratio")
        if (
            rearfoot_pressure is not None
            and rearfoot_pressure >= settings.shoe_rearfoot_pressure_neutral_percent
        ):
            clauses.append("뒤꿈치 쪽 하중이 높은 편입니다")
        pressure_imbalances: list[float] = []
        if (
            daily is not None
            and daily.left_pressure_percent is not None
            and daily.right_pressure_percent is not None
        ):
            pressure_imbalances.append(
                abs(daily.left_pressure_percent - daily.right_pressure_percent)
            )
        pressure_imbalances.extend(
            abs(analysis.left_pressure_ratio - analysis.right_pressure_ratio)
            for analysis in state.static_pressure_analyses
            if analysis.left_pressure_ratio is not None
            and analysis.right_pressure_ratio is not None
        )
        pressure_imbalance = max(pressure_imbalances) if pressure_imbalances else None
        if (
            pressure_imbalance is not None
            and pressure_imbalance >= settings.shoe_pressure_imbalance_neutral_percent
        ):
            clauses.append("좌우 압력 차이가 큰 편입니다")
        if (
            daily is not None
            and daily.balance_score is not None
            and daily.balance_score < settings.shoe_balance_neutral_score
        ):
            clauses.append("자세 균형 점수가 낮은 편입니다")
        static_balance_values = [
            analysis.balance_score
            for analysis in state.static_pressure_analyses
            if analysis.balance_score is not None
        ]
        static_balance = min(static_balance_values) if static_balance_values else None
        if (
            static_balance is not None
            and static_balance < settings.shoe_balance_neutral_score
            and not any("균형 점수" in clause for clause in clauses)
        ):
            clauses.append("족저 하중 균형 점수가 낮은 편입니다")
        if clauses:
            return _join_measurement_clauses(clauses)
        if rearfoot_pressure is not None and pressure_imbalance is not None:
            return (
                "측정 결과 뒤꿈치 쪽 하중 집중은 두드러지지 않고, "
                "좌우 압력 차이도 크지 않습니다."
            )
        if rearfoot_pressure is not None:
            return "측정 결과 뒤꿈치 쪽 하중 집중은 두드러지지 않습니다."
        if pressure_imbalance is not None:
            return "측정 결과 좌우 압력 차이는 크지 않습니다."
        return None

    if reason_type == "INSOLE":
        clauses = []
        if (
            daily is not None
            and daily.avg_humidity_percent is not None
            and daily.avg_humidity_percent >= settings.shoe_humidity_neutral_percent
        ):
            clauses.append("발 내부 습도가 높은 편입니다")
        if (
            tinea is not None
            and tinea.fungal_suspicion_safety_score
            < settings.shoe_fungal_safety_neutral_score
        ):
            clauses.append("통기성 관리가 필요합니다")
        if (
            tinea is not None
            and tinea.skin_reaction_safety_score
            < settings.shoe_skin_safety_neutral_score
        ):
            clauses.append("피부 반응 관리가 필요합니다")
        if clauses:
            return _join_measurement_clauses(clauses)
        if (
            (daily is not None and daily.avg_humidity_percent is not None)
            or tinea is not None
        ):
            return "측정 결과 발 내부 환경에서 통기성을 특별히 더 요구하는 신호는 크지 않습니다."
        return None

    raise ValueError(f"Unknown reason type: {reason_type}")


def _join_measurement_clauses(clauses: Sequence[str]) -> str:
    if not clauses:
        raise ValueError("measurement clauses cannot be empty")
    connected: list[str] = []
    for clause in clauses[:-1]:
        if clause.endswith("필요합니다"):
            connected.append(f"{clause.removesuffix('필요합니다')}필요하고")
        elif clause.endswith("높은 편입니다"):
            connected.append(f"{clause.removesuffix('높은 편입니다')}높고")
        elif clause.endswith("낮은 편입니다"):
            connected.append(f"{clause.removesuffix('낮은 편입니다')}낮으며")
        elif clause.endswith("큰 편입니다"):
            connected.append(f"{clause.removesuffix('큰 편입니다')}크고")
        elif clause.endswith("편입니다"):
            connected.append(f"{clause.removesuffix('편입니다')}편이고")
        elif clause.endswith("가능성이 있습니다"):
            connected.append(f"{clause.removesuffix('가능성이 있습니다')}가능성이 있으며")
        else:
            connected.append(f"{clause.rstrip('.')}이고")
    body = ", ".join([*connected, clauses[-1].rstrip(". ")])
    return f"측정 결과 {body}."


def build_personalized_reason_summaries(
    state: ServerUserFootState,
    areas: Sequence[AreaScore],
    shoe: ServerShoe,
    review_candidates_by_reason: Mapping[str, Sequence[ReviewCandidateLike]],
) -> dict[str, str]:
    """Build immediate, evidence-bounded copy for all three recommendation cards.

    The function intentionally runs during the recommendation batch, where the
    user's foot state and the exact compatibility components are still present.
    It never converts a compatibility score into an unsupported product property:
    product-specific wording is added only when independent review excerpts agree.
    """

    area_by_type = {area.reason_type: area for area in areas}
    if set(area_by_type) != set(_REASON_TYPES) or len(areas) != len(_REASON_TYPES):
        raise ValueError("areas must contain FOREFOOT, HEEL, and INSOLE exactly once")
    if not any(measurement.source.strip().upper() == "RUNREPEAT" for measurement in shoe.lab_measurements):
        raise ValueError("shoe must contain a RunRepeat measurement used by the area scores")

    result: dict[str, str] = {}
    for reason_type in _REASON_TYPES:
        area = area_by_type[reason_type]
        candidates = select_review_evidence(
            reason_type,
            area.risk_level,
            tuple(review_candidates_by_reason.get(reason_type, ())),
            limit=3,
        )
        result[reason_type] = render_reason_summary(
            reason_type,
            area.risk_level,
            candidates,
            measurement_sentence=_measurement_sentence(reason_type, state),
        )
    return result


def build_personalized_point_fallback(areas: Sequence[AreaScore]) -> str:
    """Return useful non-null copy when reviews cannot support product claims.

    This path deliberately says only what the user/product compatibility scores
    establish.  It prevents a reviewless shoe from triggering the slow detail
    generation path without inventing styling, sizing, or material properties.
    """

    area_by_type = {area.reason_type: area for area in areas}
    if set(area_by_type) != set(_REASON_TYPES) or len(areas) != len(_REASON_TYPES):
        raise ValueError("areas must contain FOREFOOT, HEEL, and INSOLE exactly once")

    forefoot = {
        "HIGH": (
            "현재 발 치수와 제품 실측값을 비교하면 발볼과 앞코 여유가 부족할 수 있어, "
            "착화 전 발볼과 발가락의 압박 여부를 확인하는 편이 좋습니다."
        ),
        "MEDIUM": (
            "현재 발 치수와 제품 실측값을 비교하면 발볼과 앞코 공간은 대체로 무난하지만, "
            "오래 신을 때의 압박감에는 개인차가 있을 수 있습니다."
        ),
        "LOW": (
            "현재 발 치수와 제품 실측값을 비교하면 발볼과 앞코 공간은 비교적 잘 맞는 편입니다."
        ),
    }[area_by_type["FOREFOOT"].risk_level]
    heel = {
        "HIGH": (
            "뒤꿈치 하중과 관련된 충격 완화·반발력·쿠션 특성은 장시간 착용 전에 부담 여부를 "
            "확인해 보는 것이 좋습니다."
        ),
        "MEDIUM": (
            "뒤꿈치 하중과 관련된 충격 완화·반발력·쿠션 특성은 보통 수준입니다."
        ),
        "LOW": "뒤꿈치 하중과 관련된 충격 완화·반발력·쿠션 특성 점수는 좋은 수준입니다.",
    }[area_by_type["HEEL"].risk_level]
    insole = {
        "HIGH": (
            "제품 통기성은 오래 신기 전에 내부 열감과 습기를 확인하는 편이 좋습니다."
        ),
        "MEDIUM": (
            "제품 통기성은 보통 수준이므로 장시간 착용 시 내부 환경을 확인해 보세요."
        ),
        "LOW": "제품 통기성 점수는 좋은 수준입니다.",
    }[area_by_type["INSOLE"].risk_level]
    return " ".join((forefoot, heel, insole))


__all__ = [
    "ReviewCandidateLike",
    "build_personalized_point_fallback",
    "build_personalized_reason_summaries",
]
