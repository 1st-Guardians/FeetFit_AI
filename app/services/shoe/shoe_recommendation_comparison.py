from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.schemas.shoe_server import ServerRecommendationContext
from app.services.shoe.shoe_recommendation import (
    ShoeRecommendationBatch,
    compute_shoe_recommendations_v1,
)
from app.services.shoe.shoe_recommendation_v2 import (
    ShoeRecommendationV2Batch,
    ShoeRecommendationV2NotImplementedError,
    compute_shoe_recommendations_v2,
)


@dataclass(frozen=True)
class ShoeRecommendationComparisonItem:
    shoe_id: int
    v1_fit_score: float | None
    v2_fit_score: float | None
    score_delta: float | None
    v1_rank: int | None
    v2_rank: int | None
    rank_delta: int | None


@dataclass(frozen=True)
class ShoeRecommendationComparison:
    measurement_session_id: int
    v2_available: bool
    items: tuple[ShoeRecommendationComparisonItem, ...]
    v2_unavailable_reason: str | None = None


def _ranks(scores: dict[int, float]) -> dict[int, int]:
    # Mirrors the persisted recommendation ordering: score DESC, shoeId ASC.
    ordered = sorted(scores, key=lambda shoe_id: (-scores[shoe_id], shoe_id))
    return {shoe_id: index + 1 for index, shoe_id in enumerate(ordered)}


def compare_recommendation_batches(
    v1_batch: ShoeRecommendationBatch,
    v2_batch: ShoeRecommendationV2Batch | None,
    *,
    v2_unavailable_reason: str | None = None,
) -> ShoeRecommendationComparison:
    if v2_batch is not None:
        if v1_batch.measurement_session_id != v2_batch.measurement_session_id:
            raise ValueError("V1 and V2 batches must belong to the same measurement session.")
        if v1_batch.user_id != v2_batch.user_id:
            raise ValueError("V1 and V2 batches must belong to the same user.")

    v1_scores = {item.shoe_id: item.fit_score for item in v1_batch.items}
    v2_scores = {
        item.shoe_id: item.fit_score_v2
        for item in (() if v2_batch is None else v2_batch.items)
        if item.fit_score_v2 is not None
    }
    v1_ranks = _ranks(v1_scores)
    v2_ranks = _ranks(v2_scores)
    shoe_ids = sorted(set(v1_scores) | set(v2_scores))

    items = []
    for shoe_id in shoe_ids:
        v1_score = v1_scores.get(shoe_id)
        v2_score = v2_scores.get(shoe_id)
        v1_rank = v1_ranks.get(shoe_id)
        v2_rank = v2_ranks.get(shoe_id)
        items.append(
            ShoeRecommendationComparisonItem(
                shoe_id=shoe_id,
                v1_fit_score=v1_score,
                v2_fit_score=v2_score,
                score_delta=(
                    round(v2_score - v1_score, 2)
                    if v1_score is not None and v2_score is not None
                    else None
                ),
                v1_rank=v1_rank,
                v2_rank=v2_rank,
                rank_delta=(
                    v2_rank - v1_rank
                    if v1_rank is not None and v2_rank is not None
                    else None
                ),
            )
        )
    return ShoeRecommendationComparison(
        measurement_session_id=v1_batch.measurement_session_id,
        v2_available=v2_batch is not None,
        items=tuple(items),
        v2_unavailable_reason=v2_unavailable_reason,
    )


def compare_recommendations(
    context: ServerRecommendationContext,
    *,
    v1_runner: Callable[[ServerRecommendationContext], ShoeRecommendationBatch] = (
        compute_shoe_recommendations_v1
    ),
    v2_runner: Callable[[ServerRecommendationContext], ShoeRecommendationV2Batch] = (
        compute_shoe_recommendations_v2
    ),
) -> ShoeRecommendationComparison:
    """Development/test helper; it is not exposed through an operating API."""

    v1_batch = v1_runner(context)
    try:
        v2_batch = v2_runner(context)
    except ShoeRecommendationV2NotImplementedError as exc:
        return compare_recommendation_batches(
            v1_batch,
            None,
            v2_unavailable_reason=str(exc),
        )
    return compare_recommendation_batches(v1_batch, v2_batch)


__all__ = [
    "ShoeRecommendationComparison",
    "ShoeRecommendationComparisonItem",
    "compare_recommendation_batches",
    "compare_recommendations",
]
