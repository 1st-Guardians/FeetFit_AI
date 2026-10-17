from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from app.schemas.shoe_server import ServerRecommendationContext
from app.services.shoe.recommendation_policy_version import RecommendationPolicyVersion


class ShoeRecommendationV2NotImplementedError(RuntimeError):
    """Raised until the V2 data pipeline and fit policy are implemented."""


@dataclass(frozen=True)
class V2EvidenceReview:
    review_id: int
    reason_type: str
    similarity: float | None = None


@dataclass(frozen=True)
class V2ShoeRecommendation:
    """V2-only result model; it is intentionally not a V1 API/DB payload."""

    shoe_id: int
    fit_score_v2: float | None = None
    confidence_score: float | None = None
    domain_scores: Mapping[str, float | None] = field(default_factory=dict)
    strengths: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    evidence_reviews: tuple[V2EvidenceReview, ...] = ()
    point_summary: str | None = None
    rank_reason: str | None = None
    recommendation_policy_version: RecommendationPolicyVersion = (
        RecommendationPolicyVersion.V2
    )

    def __post_init__(self) -> None:
        if self.recommendation_policy_version is not RecommendationPolicyVersion.V2:
            raise ValueError("V2 results must use recommendation policy version V2.")


@dataclass(frozen=True)
class ShoeRecommendationV2Batch:
    user_id: int
    measurement_session_id: int
    items: tuple[V2ShoeRecommendation, ...]
    recommendation_policy_version: RecommendationPolicyVersion = (
        RecommendationPolicyVersion.V2
    )

    def __post_init__(self) -> None:
        if self.recommendation_policy_version is not RecommendationPolicyVersion.V2:
            raise ValueError("V2 batches must use recommendation policy version V2.")


def compute_shoe_recommendations_v2(
    context: ServerRecommendationContext,
) -> ShoeRecommendationV2Batch:
    """Independent V2 entry point reserved for the next implementation phase.

    It must not delegate to V1 or synthesize placeholder scores: doing either
    would make V1/V2 comparison ambiguous and could accidentally persist V1
    values as V2 results.
    """

    raise ShoeRecommendationV2NotImplementedError(
        "Recommendation policy V2 is not implemented; "
        f"measurementSessionId={context.measurement_session_id} was not evaluated."
    )


__all__ = [
    "ShoeRecommendationV2Batch",
    "ShoeRecommendationV2NotImplementedError",
    "V2EvidenceReview",
    "V2ShoeRecommendation",
    "compute_shoe_recommendations_v2",
]
