from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import re
import threading
from time import perf_counter

from fastapi import APIRouter, BackgroundTasks, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.config import settings
from app.core.security import bearer_scheme, require_internal_api_key
from app.schemas.shoe_fit_comment import ShoeFitSummary
from app.schemas.shoe_server import ServerSavedRecommendation, ServerShoeCharacteristics
from app.schemas.shoes import (
    ShoeSummaryForwardRequest,
    ShoeSummaryGenerateRequest,
    ShoeSummaryReasonPayload,
    ShoeSummaryTriggerRequest,
    ShoeSummaryTriggerResponse,
)
from app.services.shoe.shoe_fit_comment_service import (
    ReasonFactsForPrompt,
    ShoeFitCommentError,
    build_immediate_shoe_summaries,
    generate_shoe_summaries,
)
from app.services.shoe.shoe_recommendation import risk_level_from_score
from app.services.shoe.shoe_server_client import (
    ShoeServerClient,
    ShoeServerClientError,
    ShoeServerConfigurationError,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_summary_inflight: set[tuple[int, int]] = set()
_summary_immediate_waiters: dict[
    tuple[int, int], Future["_ImmediateSummaryOutcome"]
] = {}
_direct_summary_waiters: dict[
    tuple[int, int], Future["_DirectSummaryOutcome"]
] = {}
_summary_inflight_lock = threading.Lock()
_IMMEDIATE_WAIT_TIMEOUT_SECONDS = 4.0
_DIRECT_DEDUPLICATED_WAIT_SLACK_SECONDS = 1.0
_LEGACY_MECHANICAL_SUMMARY_PATTERN = re.compile(
    r"정량 분석 기준|정량 분석 결과만 안내|관련 착화 리뷰\s*\d+개|"
    r"그 밖의 착용 특성은 여러 리뷰에서 같은 방향으로 반복된 정보가\s*"
    r"충분하지 않아 단정하기 어렵습니다|"
    r"쿠션감에 대한 만족도가 높고,\s*무게도 가볍게 느껴지는 편입니다|"
    r"(?:발볼 공간|앞코 공간|쿠션감)은\s*"
    r"(?:낮은 편|보통 수준|높은 편)입니다|"
    r"착화 시 가볍게 느껴지는 편입니다"
)


@dataclass(frozen=True)
class _ImmediateSummaryOutcome:
    saved: bool
    status_code: int
    detail: str


@dataclass(frozen=True)
class _DirectSummaryOutcome:
    payload: ShoeSummaryForwardRequest | None
    status_code: int
    detail: str


@dataclass(frozen=True)
class _SummaryWork:
    server_client: ShoeServerClient
    saved: ServerSavedRecommendation
    characteristics: ServerShoeCharacteristics
    reason_facts: list[ReasonFactsForPrompt]
    context_elapsed_ms: float
    total_started: float


class _MissingSavedRecommendationError(RuntimeError):
    pass


class _DirectSummaryContextTimeoutError(RuntimeError):
    pass


def _reserve_summary(
    key: tuple[int, int],
) -> tuple[bool, Future[_ImmediateSummaryOutcome]]:
    with _summary_inflight_lock:
        existing = _summary_immediate_waiters.get(key)
        if key in _summary_inflight and existing is not None:
            return False, existing
        future: Future[_ImmediateSummaryOutcome] = Future()
        _summary_inflight.add(key)
        _summary_immediate_waiters[key] = future
        return True, future


def _publish_immediate_outcome(
    key: tuple[int, int], outcome: _ImmediateSummaryOutcome
) -> None:
    with _summary_inflight_lock:
        future = _summary_immediate_waiters.get(key)
        if future is not None and not future.done():
            future.set_result(outcome)


def _release_summary(key: tuple[int, int]) -> None:
    with _summary_inflight_lock:
        future = _summary_immediate_waiters.pop(key, None)
        if future is not None and not future.done():
            future.set_result(
                _ImmediateSummaryOutcome(
                    saved=False,
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Shoe summary task ended before the immediate save completed.",
                )
            )
        _summary_inflight.discard(key)


def _reserve_direct_summary(
    key: tuple[int, int],
) -> tuple[bool, Future[_DirectSummaryOutcome]]:
    """Reserve natural-language generation independently from legacy saves."""

    with _summary_inflight_lock:
        existing = _direct_summary_waiters.get(key)
        if existing is not None:
            return False, existing
        future: Future[_DirectSummaryOutcome] = Future()
        _direct_summary_waiters[key] = future
        return True, future


def _publish_direct_summary(
    future: Future[_DirectSummaryOutcome], outcome: _DirectSummaryOutcome
) -> None:
    with _summary_inflight_lock:
        if not future.done():
            future.set_result(outcome)


def _release_direct_summary(
    key: tuple[int, int], future: Future[_DirectSummaryOutcome]
) -> None:
    with _summary_inflight_lock:
        if not future.done():
            future.set_result(
                _DirectSummaryOutcome(
                    payload=None,
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Direct shoe summary generation ended without a result.",
                )
            )
        if _direct_summary_waiters.get(key) is future:
            _direct_summary_waiters.pop(key, None)


def _failure_outcome(exc: Exception) -> _ImmediateSummaryOutcome:
    if isinstance(exc, ShoeServerConfigurationError):
        return _ImmediateSummaryOutcome(
            saved=False,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feetfit_Server summary integration is not configured.",
        )
    if isinstance(exc, _MissingSavedRecommendationError):
        return _ImmediateSummaryOutcome(
            saved=False,
            status_code=status.HTTP_409_CONFLICT,
            detail="A saved shoe recommendation is required before summary generation.",
        )
    if isinstance(exc, ShoeServerClientError):
        return _ImmediateSummaryOutcome(
            saved=False,
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Feetfit_Server summary context or save request failed.",
        )
    if isinstance(exc, ShoeFitCommentError):
        return _ImmediateSummaryOutcome(
            saved=False,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The immediate shoe summary could not be generated from saved evidence.",
        )
    return _ImmediateSummaryOutcome(
        saved=False,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Immediate shoe summary generation failed.",
    )


def _direct_failure_outcome(exc: Exception) -> _DirectSummaryOutcome:
    if isinstance(exc, ShoeServerConfigurationError):
        return _DirectSummaryOutcome(
            payload=None,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feetfit_Server summary integration is not configured.",
        )
    if isinstance(exc, _MissingSavedRecommendationError):
        return _DirectSummaryOutcome(
            payload=None,
            status_code=status.HTTP_409_CONFLICT,
            detail="A saved shoe recommendation is required before summary generation.",
        )
    if isinstance(exc, _DirectSummaryContextTimeoutError):
        return _DirectSummaryOutcome(
            payload=None,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Feetfit_Server summary context was not available within the generation deadline.",
        )
    if isinstance(exc, ShoeServerClientError):
        return _DirectSummaryOutcome(
            payload=None,
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Feetfit_Server summary context request failed.",
        )
    if isinstance(exc, ShoeFitCommentError):
        return _DirectSummaryOutcome(
            payload=None,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The shoe summary could not be generated from saved evidence.",
        )
    return _DirectSummaryOutcome(
        payload=None,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Direct shoe summary generation failed.",
    )


def _forward_request(
    measurement_session_id: int, summary: ShoeFitSummary
) -> ShoeSummaryForwardRequest:
    return ShoeSummaryForwardRequest(
        measurement_session_id=measurement_session_id,
        point_summary=summary.point_summary,
        reasons=[
            ShoeSummaryReasonPayload(
                reason_type="FOREFOOT",
                review_summary=summary.forefoot_summary,
                review_ids=summary.forefoot_review_ids,
            ),
            ShoeSummaryReasonPayload(
                reason_type="HEEL",
                review_summary=summary.heel_summary,
                review_ids=summary.heel_review_ids,
            ),
            ShoeSummaryReasonPayload(
                reason_type="INSOLE",
                review_summary=summary.insole_summary,
                review_ids=summary.insole_review_ids,
            ),
        ],
    )


@router.post(
    "/summaries/generate",
    status_code=status.HTTP_200_OK,
    response_model=ShoeSummaryForwardRequest,
    summary="신발 상세 응답용 자연어 요약 동기 생성",
    description=(
        "Feetfit_Server가 최초 신발 상세 조회 안에서 호출하는 동기 생성 API입니다. "
        "저장된 동일 측정 세션의 fitScore, riskLevel, 근거 리뷰와 신발 특성을 조회한 뒤 "
        "검증된 결정론적 문장 생성기로 pointSummary 및 FOREFOOT/HEEL/INSOLE "
        "reviewSummary를 즉시 만들고 Server 저장 DTO와 동일한 JSON을 반환합니다. "
        "이 경로는 Ollama와 Server 저장 callback을 호출하지 않으므로 모델 준비 상태와 무관하게 "
        "첫 조회 지연을 일정하게 유지합니다. Ollama 보강은 비동기 /summaries 경로에서만 수행합니다. "
        "같은 세션/신발의 동시 요청은 하나의 생성 결과를 공유합니다."
    ),
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "The recommendation batch has not saved this shoe yet."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Saved evidence cannot satisfy the summary contract."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The Server integration is not configured."
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "The Feetfit_Server context lookup failed."
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "The Server context lookup exceeded the direct-generation deadline."
        },
    },
)
async def generate_shoe_summary_for_detail(
    request: ShoeSummaryGenerateRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    _internal_api_key: None = Security(require_internal_api_key),
) -> ShoeSummaryForwardRequest:
    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    key = (request.measurement_session_id, request.shoe_id)
    reserved, future = _reserve_direct_summary(key)

    if not reserved:
        return await _wait_for_direct_summary(future)

    try:
        payload = await _generate_direct_summary_payload(
            request.shoe_id,
            request.measurement_session_id,
            authorization_header,
        )
    except asyncio.CancelledError:
        _publish_direct_summary(
            future,
            _DirectSummaryOutcome(
                payload=None,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Direct shoe summary generation was cancelled.",
            ),
        )
        raise
    except Exception as exc:
        outcome = _direct_failure_outcome(exc)
        _publish_direct_summary(future, outcome)
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 상세 조회용 요약 생성 실패",
            request.shoe_id,
            request.measurement_session_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=outcome.status_code, detail=outcome.detail
        ) from exc
    else:
        _publish_direct_summary(
            future,
            _DirectSummaryOutcome(
                payload=payload,
                status_code=status.HTTP_200_OK,
                detail="Direct shoe summary generated.",
            ),
        )
        return payload
    finally:
        _release_direct_summary(key, future)


async def _wait_for_direct_summary(
    future: Future[_DirectSummaryOutcome],
) -> ShoeSummaryForwardRequest:
    try:
        outcome = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=(
                settings.ollama_direct_summary_deadline_seconds
                + _DIRECT_DEDUPLICATED_WAIT_SLACK_SECONDS
            ),
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The in-flight direct shoe summary did not finish within the generation deadline.",
        ) from exc
    if outcome.payload is None:
        raise HTTPException(status_code=outcome.status_code, detail=outcome.detail)
    return outcome.payload


async def _generate_direct_summary_payload(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> ShoeSummaryForwardRequest:
    """Generate a natural result without invoking the Server save callback."""

    started = perf_counter()
    deadline_seconds = settings.ollama_direct_summary_deadline_seconds
    try:
        work = await asyncio.wait_for(
            _prepare_summary_work(
                shoe_id, measurement_session_id, authorization_header
            ),
            timeout=deadline_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise _DirectSummaryContextTimeoutError from exc

    summary = await _build_direct_fallback(work)

    payload = _forward_request(measurement_session_id, summary)
    logger.info(
        "Direct shoe summary generated. shoe_id=%s measurement_session_id=%s elapsed_ms=%.1f",
        shoe_id,
        measurement_session_id,
        (perf_counter() - started) * 1000,
    )
    return payload


async def _build_direct_fallback(work: _SummaryWork) -> ShoeFitSummary:
    immediate = await asyncio.to_thread(
        build_immediate_shoe_summaries,
        shoe_name=f"{work.saved.brand_name} {work.saved.shoe_name}",
        fit_score=work.saved.fit_score,
        overall_risk_level=risk_level_from_score(work.saved.fit_score),
        reasons=work.reason_facts,
        characteristics=work.characteristics,
        personal_fit_evidence=work.saved.personal_fit_evidence,
    )
    return _merge_saved_summary_copy(immediate, work.saved)


def _preservable_saved_summary(value: str | None) -> bool:
    return bool(value and value.strip()) and not _LEGACY_MECHANICAL_SUMMARY_PATTERN.search(
        value
    )


def _merge_saved_summary_copy(
    immediate: ShoeFitSummary, saved: ServerSavedRecommendation
) -> ShoeFitSummary:
    """Preserve only the batch-level product point summary.

    Per-reason summaries must always come from the current, domain/risk-validated
    review selection. Reusing an older saved reason copy could reintroduce a
    negative LOW review, cross-area text, or a judgment for an empty review set.
    """

    updates: dict[str, str] = {}
    if _preservable_saved_summary(saved.point_summary):
        updates["point_summary"] = saved.point_summary.strip()

    return immediate.model_copy(update=updates) if updates else immediate


@router.post(
    "/summaries",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ShoeSummaryTriggerResponse,
    response_model_exclude_none=True,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "The recommendation batch has not saved this shoe yet."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The request or deterministic summary evidence is invalid."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The Server integration is not configured or scheduling failed."
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "The Feetfit_Server context lookup or save callback failed."
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "The immediate summary was not saved within four seconds."
        },
    },
    summary="신발 상세 조회 시 착용 코멘트 생성 트리거",
    description=(
        "Feetfit_Server가 신발 상세 조회 응답의 pointSummary 또는 reviewSummary가 비어 있을 때 호출합니다. "
        "이미 배치(/reports/shoe-recommendations)로 계산/저장되어 있는 fitScore·riskLevel·근거 리뷰를 "
        "JWT 범위의 Server 내부 API에서 다시 계산하지 않고 읽어와서, Ollama로 pointSummary와 부위별 "
        "reviewSummary 문장만 생성합니다. "
        "waitForImmediate=false(기본)면 기존처럼 전체 작업을 백그라운드로 수행합니다. "
        "true면 최대 4초 동안 결정론적 즉시 요약의 Server 저장 완료를 기다린 뒤 "
        "immediateSaved=true를 반환하고 Ollama 보강만 백그라운드로 수행합니다."
    ),
)
async def trigger_shoe_summary_generation(
    request: ShoeSummaryTriggerRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    _internal_api_key: None = Security(require_internal_api_key),
) -> ShoeSummaryTriggerResponse:
    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    key = (request.measurement_session_id, request.shoe_id)
    reserved, immediate_future = _reserve_summary(key)
    if not reserved:
        if not request.wait_for_immediate:
            return ShoeSummaryTriggerResponse(accepted=True, deduplicated=True)
        return await _wait_for_immediate_outcome(
            immediate_future, deduplicated=True
        )

    if request.wait_for_immediate:
        try:
            work = await asyncio.wait_for(
                _prepare_and_save_immediate_summary(
                    request.shoe_id,
                    request.measurement_session_id,
                    authorization_header,
                ),
                timeout=_IMMEDIATE_WAIT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            outcome = _ImmediateSummaryOutcome(
                saved=False,
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "The immediate shoe summary was not saved within "
                    f"{_IMMEDIATE_WAIT_TIMEOUT_SECONDS:g} seconds."
                ),
            )
            _publish_immediate_outcome(key, outcome)
            _release_summary(key)
            logger.warning(
                "shoe_id=%s measurement_session_id=%s: 즉시 요약 동기 저장 시간 초과",
                request.shoe_id,
                request.measurement_session_id,
            )
            raise HTTPException(
                status_code=outcome.status_code, detail=outcome.detail
            ) from exc
        except asyncio.CancelledError:
            _publish_immediate_outcome(
                key,
                _ImmediateSummaryOutcome(
                    saved=False,
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The immediate shoe summary request was cancelled.",
                ),
            )
            _release_summary(key)
            raise
        except Exception as exc:
            outcome = _failure_outcome(exc)
            _publish_immediate_outcome(key, outcome)
            _release_summary(key)
            logger.warning(
                "shoe_id=%s measurement_session_id=%s: 즉시 요약 동기 저장 실패",
                request.shoe_id,
                request.measurement_session_id,
                exc_info=True,
            )
            raise HTTPException(
                status_code=outcome.status_code, detail=outcome.detail
            ) from exc

        try:
            background_tasks.add_task(
                _generate_and_save_refinement,
                request.shoe_id,
                request.measurement_session_id,
                work,
            )
        except Exception as exc:
            outcome = _ImmediateSummaryOutcome(
                saved=False,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The Ollama refinement task could not be scheduled.",
            )
            _publish_immediate_outcome(key, outcome)
            _release_summary(key)
            logger.warning(
                "shoe_id=%s measurement_session_id=%s: Ollama 보강 작업 등록 실패",
                request.shoe_id,
                request.measurement_session_id,
                exc_info=True,
            )
            raise HTTPException(
                status_code=outcome.status_code, detail=outcome.detail
            ) from exc
        _publish_immediate_outcome(
            key,
            _ImmediateSummaryOutcome(
                saved=True,
                status_code=status.HTTP_202_ACCEPTED,
                detail="Immediate shoe summary saved.",
            ),
        )
        return ShoeSummaryTriggerResponse(
            accepted=True, deduplicated=False, immediate_saved=True
        )

    background_tasks.add_task(
        _generate_and_save_summary,
        request.shoe_id,
        request.measurement_session_id,
        authorization_header,
    )
    return ShoeSummaryTriggerResponse(accepted=True, deduplicated=False)


async def _wait_for_immediate_outcome(
    future: Future[_ImmediateSummaryOutcome], *, deduplicated: bool
) -> ShoeSummaryTriggerResponse:
    try:
        outcome = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=_IMMEDIATE_WAIT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The in-flight immediate shoe summary was not saved within "
                f"{_IMMEDIATE_WAIT_TIMEOUT_SECONDS:g} seconds."
            ),
        ) from exc
    if not outcome.saved:
        raise HTTPException(status_code=outcome.status_code, detail=outcome.detail)
    return ShoeSummaryTriggerResponse(
        accepted=True, deduplicated=deduplicated, immediate_saved=True
    )


async def _generate_and_save_summary(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> None:
    """BackgroundTasks 경계에서는 모든 예외를 흡수한다.

    이 엔드포인트는 호출자에게 이미 202를 반환한 뒤 실행되므로, 예상하지 못한
    Ollama/계약 오류도 ASGI background-task 예외로 다시 전파하지 않는다.
    """
    try:
        await _generate_and_save_summary_inner(
            shoe_id, measurement_session_id, authorization_header
        )
    except Exception:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 요약 백그라운드 작업에서 예상하지 못한 오류",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
    finally:
        _release_summary((measurement_session_id, shoe_id))


async def _prepare_summary_work(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> _SummaryWork:
    total_started = perf_counter()
    context_started = perf_counter()
    server_client = ShoeServerClient(authorization_header)
    saved, characteristics = await asyncio.gather(
        server_client.fetch_saved_recommendation(measurement_session_id, shoe_id),
        server_client.fetch_shoe_characteristics(shoe_id),
    )
    if saved is None:
        raise _MissingSavedRecommendationError(
            "The recommendation batch must be saved before summary generation."
        )
    return _SummaryWork(
        server_client=server_client,
        saved=saved,
        characteristics=characteristics,
        reason_facts=[
            ReasonFactsForPrompt(
                reason_type=reason.reason_type,
                title=reason.title,
                risk_level=reason.risk_level,
                review_ids=[review.id for review in reason.reviews],
                review_texts=[review.review_text for review in reason.reviews],
            )
            for reason in saved.reasons
        ],
        context_elapsed_ms=(perf_counter() - context_started) * 1000,
        total_started=total_started,
    )


async def _save_immediate_summary(
    shoe_id: int, measurement_session_id: int, work: _SummaryWork
) -> None:
    immediate_started = perf_counter()
    immediate = await asyncio.to_thread(
        build_immediate_shoe_summaries,
        shoe_name=f"{work.saved.brand_name} {work.saved.shoe_name}",
        fit_score=work.saved.fit_score,
        overall_risk_level=risk_level_from_score(work.saved.fit_score),
        reasons=work.reason_facts,
        characteristics=work.characteristics,
        personal_fit_evidence=work.saved.personal_fit_evidence,
    )
    immediate = _merge_saved_summary_copy(immediate, work.saved)
    await work.server_client.save_summary(
        shoe_id, _forward_request(measurement_session_id, immediate)
    )
    logger.info(
        "Immediate typed-evidence shoe summary saved. shoe_id=%s "
        "measurement_session_id=%s elapsed_ms=%.1f",
        shoe_id,
        measurement_session_id,
        (perf_counter() - immediate_started) * 1000,
    )


async def _prepare_and_save_immediate_summary(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> _SummaryWork:
    work = await _prepare_summary_work(
        shoe_id, measurement_session_id, authorization_header
    )
    await _save_immediate_summary(shoe_id, measurement_session_id, work)
    return work


async def _generate_and_save_refinement(
    shoe_id: int, measurement_session_id: int, work: _SummaryWork
) -> None:
    """Run only the Ollama refinement after a synchronous immediate save."""
    try:
        await _generate_and_save_refinement_inner(
            shoe_id, measurement_session_id, work
        )
    except Exception:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: Ollama 보강 백그라운드 "
            "작업에서 예상하지 못한 오류",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
    finally:
        _release_summary((measurement_session_id, shoe_id))


async def _generate_and_save_refinement_inner(
    shoe_id: int, measurement_session_id: int, work: _SummaryWork
) -> None:
    generation_started = perf_counter()
    try:
        summary = await generate_shoe_summaries(
            shoe_name=f"{work.saved.brand_name} {work.saved.shoe_name}",
            fit_score=work.saved.fit_score,
            overall_risk_level=risk_level_from_score(work.saved.fit_score),
            reasons=work.reason_facts,
            characteristics=work.characteristics,
            personal_fit_evidence=work.saved.personal_fit_evidence,
        )
    except ShoeFitCommentError:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: Ollama 코멘트 생성 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
        return
    generation_elapsed_ms = (perf_counter() - generation_started) * 1000

    save_started = perf_counter()
    try:
        await work.server_client.save_summary(
            shoe_id, _forward_request(measurement_session_id, summary)
        )
    except ShoeServerClientError:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 요약 저장 콜백 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
    else:
        save_elapsed_ms = (perf_counter() - save_started) * 1000
        total_elapsed_ms = (perf_counter() - work.total_started) * 1000
        logger.info(
            "Shoe summary saved to Feetfit_Server. shoe_id=%s "
            "measurement_session_id=%s context_ms=%.1f generation_ms=%.1f "
            "save_ms=%.1f total_ms=%.1f",
            shoe_id,
            measurement_session_id,
            work.context_elapsed_ms,
            generation_elapsed_ms,
            save_elapsed_ms,
            total_elapsed_ms,
        )


async def _generate_and_save_summary_inner(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> None:
    key = (measurement_session_id, shoe_id)
    logger.info(
        "Shoe summary background generation started. shoe_id=%s "
        "measurement_session_id=%s",
        shoe_id,
        measurement_session_id,
    )
    try:
        work = await _prepare_summary_work(
            shoe_id, measurement_session_id, authorization_header
        )
    except _MissingSavedRecommendationError as exc:
        _publish_immediate_outcome(key, _failure_outcome(exc))
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 저장된 신발 적합도 "
            "데이터가 없습니다 (배치가 먼저 실행돼야 합니다).",
            shoe_id,
            measurement_session_id,
        )
        return
    except ShoeServerClientError as exc:
        _publish_immediate_outcome(key, _failure_outcome(exc))
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: Server 요약 context 조회 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
        return

    try:
        await _save_immediate_summary(shoe_id, measurement_session_id, work)
    except (ShoeFitCommentError, ShoeServerClientError) as exc:
        _publish_immediate_outcome(key, _failure_outcome(exc))
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 즉시 요약 저장 실패; "
            "Ollama 보강은 계속 진행합니다.",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
    else:
        _publish_immediate_outcome(
            key,
            _ImmediateSummaryOutcome(
                saved=True,
                status_code=status.HTTP_202_ACCEPTED,
                detail="Immediate shoe summary saved.",
            ),
        )
    await _generate_and_save_refinement_inner(shoe_id, measurement_session_id, work)
