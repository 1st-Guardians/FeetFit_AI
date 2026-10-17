from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

from app.core.config import settings
from app.schemas.shoe_server import ServerRecommendationContext, ServerShoe
from app.services.shoe.shoe_embedding import (
    embed_texts,
    flush_embedding_cache,
    get_or_embed_texts,
    rank_by_similarity,
    release_embedding_model,
)
from app.services.shoe.shoe_feature_rules import (
    REASON_TYPES,
    build_need_query,
    derive_foot_need,
    normalize_review_text,
    split_sentences,
)
from app.services.shoe.shoe_fit_policy import (
    AreaScore,
    ShoeFitPolicyError,
    overall_fit_score,
    risk_level_from_score,
    score_shoe_fit,
)
from app.services.shoe.shoe_point_summary import (
    compose_product_point_summary,
    prepare_point_evidence_facts,
    render_product_point_summary,
)
from app.services.shoe.shoe_personalized_summary import build_personalized_reason_summaries
from app.services.shoe.shoe_review_evidence_policy import (
    ReviewCandidate,
    select_review_evidence,
    validate_reason_output,
)
from app.services.shoe.recommendation_policy_version import (
    RecommendationPolicyVersion,
)


class ShoeRecommendationError(RuntimeError):
    pass


class ShoeRecommendationBusyError(ShoeRecommendationError):
    pass


logger = logging.getLogger(__name__)
_batch_lock = threading.Lock()


@dataclass(frozen=True)
class ReasonFacts:
    """A quantitative area result plus shoe-local MUSINSA evidence.

    ``score`` and ``risk_level`` come only from measurementSession + RunRepeat
    measurements. Review text never changes either value.
    """

    reason_type: str
    score: float
    title: str
    risk_level: str
    review_ids: list[int]
    review_candidates: tuple[ReviewCandidate, ...] = ()
    review_summary: str | None = None


@dataclass(frozen=True)
class ShoeFacts:
    shoe_id: int
    fit_score: float
    reasons: list[ReasonFacts]
    point_summary: str | None = None


@dataclass(frozen=True)
class ShoeRecommendationBatch:
    user_id: int
    measurement_session_id: int
    items: list[ShoeFacts]
    recommendation_policy_version: RecommendationPolicyVersion = (
        RecommendationPolicyVersion.V1
    )


def _sentence_candidates(shoe: ServerShoe) -> dict[str, tuple[int, str, str]]:
    candidates: dict[str, tuple[int, str, str]] = {}
    seen_review_bodies: set[str] = set()
    seen_sentences: set[str] = set()
    # The smallest reviewId is the deterministic canonical identity when
    # MUSINSA contains separate review records with the same body.
    for review in sorted(shoe.reviews, key=lambda item: item.id):
        body_fingerprint = normalize_review_text(review.review_text)
        if not body_fingerprint or body_fingerprint in seen_review_bodies:
            continue
        seen_review_bodies.add(body_fingerprint)
        for index, sentence in enumerate(split_sentences(review.review_text)):
            sentence_fingerprint = normalize_review_text(sentence)
            if not sentence_fingerprint or sentence_fingerprint in seen_sentences:
                continue
            seen_sentences.add(sentence_fingerprint)
            candidates[f"{shoe.id}:{review.id}:{index}"] = (
                review.id,
                sentence,
                review.review_text,
            )
    return candidates


def _semantic_shortlists(
    shoe: ServerShoe,
    query_vectors: dict[str, object],
    areas: tuple[AreaScore, ...],
) -> dict[str, tuple[ReviewCandidate, ...]]:
    sentences = _sentence_candidates(shoe)
    if not sentences:
        return {reason_type: () for reason_type in REASON_TYPES}

    embeddings = get_or_embed_texts(
        {key: sentence for key, (_review_id, sentence, _source) in sentences.items()}
    )
    risk_by_reason = {area.reason_type: area.risk_level for area in areas}
    results: dict[str, tuple[ReviewCandidate, ...]] = {}
    for reason_type in REASON_TYPES:
        ranked = rank_by_similarity(
            query_vectors[reason_type],
            embeddings,
            len(embeddings),
        )
        ranked_candidates: list[ReviewCandidate] = []
        for key, similarity in ranked:
            review_id, sentence, source_text = sentences[key]
            if similarity < settings.shoe_review_semantic_min_score:
                continue
            ranked_candidates.append(
                ReviewCandidate(
                    review_id=review_id,
                    sentence=sentence,
                    similarity=round(float(similarity), 6),
                    source_text=source_text,
                )
            )
        results[reason_type] = select_review_evidence(
            reason_type,
            risk_by_reason[reason_type],
            ranked_candidates,
            limit=min(settings.shoe_reviews_per_reason, 3),
        )
    return results


def _to_reason_facts(
    area: AreaScore,
    candidates: tuple[ReviewCandidate, ...],
    review_summary: str | None = None,
) -> ReasonFacts:
    return ReasonFacts(
        reason_type=area.reason_type,
        score=area.score,
        title=area.title,
        risk_level=area.risk_level,
        review_ids=[candidate.review_id for candidate in candidates],
        review_candidates=candidates,
        review_summary=review_summary,
    )


def _initial_point_summary(shoe: ServerShoe) -> str:
    """Build immediate product copy from every independent review for this shoe.

    BGE's three reason-specific candidates are intentionally optimized for the
    user's current need, so they can omit useful product-copy evidence such as
    sizing consensus, weight, comfort, or styling.  The recommendation context
    already contains this shoe's complete review set; the rule-based aggregation
    is cheap, deterministic, and keeps every claim tied to repeated evidence.
    """

    review_pairs = [
        (review.id, review.review_text)
        for review in shoe.reviews
    ]
    facts = prepare_point_evidence_facts(review_pairs)
    # Keep every repeated review claim until the final 2-3 sentence composer
    # groups fit, comfort, and styling. An early four-claim cap could otherwise
    # discard design/use evidence simply because it is rendered last.
    claims = render_product_point_summary(
        shoe.shoe_name,
        facts,
        max_claims=None,
        include_attributed_single_evidence=True,
    )
    return compose_product_point_summary(claims)


def _score_shoe_facts(
    shoe: ServerShoe,
    context: ServerRecommendationContext,
    query_vectors: dict[str, object] | None,
    areas: tuple[AreaScore, ...] | None = None,
) -> ShoeFacts:
    # Batch callers always supply the preflight result. Keeping the optional
    # fallback makes this focused helper usable in deterministic unit audits;
    # policy failures remain uncaught and cannot be confused with review errors.
    if areas is None:
        areas = score_shoe_fit(shoe, context.foot_state)
    raw_shortlists = (
        _semantic_shortlists(shoe, query_vectors, areas)
        if query_vectors is not None and shoe.reviews
        else {reason_type: () for reason_type in REASON_TYPES}
    )
    shortlists = {
        area.reason_type: select_review_evidence(
            area.reason_type,
            area.risk_level,
            tuple(raw_shortlists.get(area.reason_type, ())),
            limit=3,
        )
        for area in areas
    }
    review_summaries = build_personalized_reason_summaries(
        context.foot_state,
        areas,
        shoe,
        shortlists,
    )
    for area in areas:
        candidates = shortlists[area.reason_type]
        revalidated = select_review_evidence(
            area.reason_type,
            area.risk_level,
            candidates,
            limit=3,
        )
        if [candidate.review_id for candidate in revalidated] != [
            candidate.review_id for candidate in candidates
        ]:
            raise ShoeRecommendationError(
                f"Final review-evidence validation failed for {area.reason_type}."
            )
        validate_reason_output(
            area.reason_type,
            area.risk_level,
            candidates,
            review_summaries[area.reason_type],
        )
    reasons = [
        _to_reason_facts(
            area,
            shortlists[area.reason_type],
            review_summaries[area.reason_type],
        )
        for area in areas
    ]
    return ShoeFacts(
        shoe_id=shoe.id,
        fit_score=overall_fit_score(areas),
        reasons=reasons,
        point_summary=_initial_point_summary(shoe),
    )


def _compute_shoe_recommendations(
    context: ServerRecommendationContext,
) -> ShoeRecommendationBatch:
    """Compute an idempotent, session-scoped recommendation for every shoe.

    The function uses only the supplied completed-session context. Reviews are
    optional evidence and never gate or affect quantitative scoring. The Server
    is expected to send only quantitatively eligible shoes.  A full preflight
    verifies that contract before BGE is loaded, and fails the whole batch if a
    shoe slips through so a partial result can never violate expectedCount.
    """
    foot_state = context.foot_state
    if not any(
        [
            foot_state.daily_foot_analysis,
            foot_state.tina_pedis_analysis,
            foot_state.hallux_valgus_analysis,
            foot_state.static_pressure_analyses,
        ]
    ):
        raise ShoeRecommendationError(
            f"No measurement data found for measurement session {context.measurement_session_id}"
        )
    if not context.shoes:
        raise ShoeRecommendationError("Feetfit_Server returned an empty shoe catalog.")
    shoe_ids = [shoe.id for shoe in context.shoes]
    if len(shoe_ids) != len(set(shoe_ids)):
        raise ShoeRecommendationError("Recommendation context contains duplicate shoeId values.")

    scored_shoes: list[tuple[ServerShoe, tuple[AreaScore, ...]]] = []
    invalid_shoes: list[str] = []
    for shoe in context.shoes:
        try:
            areas = score_shoe_fit(shoe, foot_state)
        except ShoeFitPolicyError as exc:
            missing_domains = exc.reason_type or "FOREFOOT,HEEL,INSOLE"
            detail = (
                f"shoeId={shoe.id} modelCode={shoe.model_code!r} "
                f"shoeName={shoe.shoe_name!r} missingDomains={missing_domains} "
                f"error={exc}"
            )
            invalid_shoes.append(detail)
            logger.warning(
                "Recommendation quantitative preflight rejected a shoe. "
                "shoeId=%s modelCode=%r shoeName=%r missingDomains=%s error=%s",
                shoe.id,
                shoe.model_code,
                shoe.shoe_name,
                missing_domains,
                exc,
            )
            continue
        scored_shoes.append((shoe, areas))

    if invalid_shoes:
        logger.warning(
            "Recommendation quantitative preflight failed. "
            "catalogCount=%s eligibleCount=%s invalidCount=%s",
            len(context.shoes),
            len(scored_shoes),
            len(invalid_shoes),
        )
        raise ShoeRecommendationError(
            "Feetfit_Server recommendation context contains quantitatively "
            "ineligible shoes; refusing to forward a partial batch. "
            f"catalogCount={len(context.shoes)} eligibleCount={len(scored_shoes)} "
            f"invalidCount={len(invalid_shoes)}; "
            + " | ".join(invalid_shoes)
        )

    need = derive_foot_need(foot_state)
    query_vectors: dict[str, object] | None = None
    if any(shoe.reviews for shoe, _areas in scored_shoes):
        queries = [build_need_query(reason_type, need) for reason_type in REASON_TYPES]
        query_vectors = dict(zip(REASON_TYPES, embed_texts(queries), strict=True))

    try:
        items = [
            _score_shoe_facts(shoe, context, query_vectors, areas)
            for shoe, areas in scored_shoes
        ]
    finally:
        try:
            flush_embedding_cache()
        finally:
            release_embedding_model()
    if [item.shoe_id for item in items] != shoe_ids:
        raise ShoeRecommendationError("Recommendation output does not preserve full shoe coverage.")
    return ShoeRecommendationBatch(
        user_id=context.user_id,
        measurement_session_id=context.measurement_session_id,
        items=items,
    )


def compute_shoe_recommendations_v1(
    context: ServerRecommendationContext,
) -> ShoeRecommendationBatch:
    """Run the frozen production V1 recommendation pipeline."""

    acquired = _batch_lock.acquire(
        timeout=settings.shoe_recommendation_batch_lock_timeout_seconds
    )
    if not acquired:
        raise ShoeRecommendationBusyError(
            "Another Phase D recommendation batch is using the shared BGE-M3 runtime."
        )
    try:
        return _compute_shoe_recommendations(context)
    finally:
        _batch_lock.release()


def compute_shoe_recommendations(
    context: ServerRecommendationContext,
) -> ShoeRecommendationBatch:
    """Backward-compatible default entry point. Production defaults to V1."""

    return compute_shoe_recommendations_v1(context)


__all__ = [
    "ReasonFacts",
    "ReviewCandidate",
    "ShoeFacts",
    "ShoeRecommendationBatch",
    "ShoeRecommendationBusyError",
    "ShoeRecommendationError",
    "compute_shoe_recommendations",
    "compute_shoe_recommendations_v1",
    "risk_level_from_score",
    "select_review_evidence",
]
