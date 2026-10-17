import asyncio
import json
import logging
import time
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, Security, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme, require_internal_api_key
from app.schemas.reports import (
    FootMetricsReportRequest,
    FootTypeTextGenerationRequest,
    FootTypeTextGenerationResponse,
    HalluxValgusReportRequest,
)
from app.schemas.shoes import (
    ShoeRecommendationForwardRequest,
    ShoeRecommendationItemPayload,
    ShoeRecommendationReasonPayload,
    ShoeRecommendationTriggerRequest,
)
from app.services.hallux_valgus_analysis import (
    analyze_hallux_valgus_image,
    score_analysis_text,
)
from app.services.integrated_foot_analysis import (
    IntegratedFootAnalysisResult,
    analyze_integrated_foot_photo,
)
from app.services.report_text_generation import (
    generate_foot_type_text,
    generate_hallux_score_analysis_text,
)
from app.services.shoe.shoe_recommendation import (
    ShoeRecommendationBusyError,
    ShoeRecommendationError,
    compute_shoe_recommendations,
)
from app.services.shoe.shoe_server_client import (
    ShoeServerClient,
    ShoeServerClientError,
    ShoeServerConfigurationError,
)
from app.services.tinea_analysis import AnalysisError, analyze_foot_image, combine_png_images_side_by_side
from app.services.tinea_session_handoff import stage_dorsal_tinea


router = APIRouter()
logger = logging.getLogger(__name__)
shoe_progress_logger = logging.getLogger("uvicorn.error")
UPSTREAM_ERROR_BODY_LIMIT = 2000
SHOE_RECOMMENDATION_PIPELINE = "shoe-recommendation-v1"


def _log_shoe_recommendation_progress(
    *,
    request_id: str,
    measurement_session_id: int,
    event: str,
    stage: str,
    message: str,
    level: str = "info",
    status_code: int | None = None,
    recommendation_count: int | None = None,
    duration_ms: int | None = None,
    error_type: str | None = None,
) -> None:
    """Emit one operator-friendly event without request or recommendation data."""

    details: dict[str, int | str] = {}
    if status_code is not None:
        details["statusCode"] = int(status_code)
    if recommendation_count is not None:
        details["recommendationCount"] = int(recommendation_count)
    if duration_ms is not None:
        details["durationMs"] = max(0, int(duration_ms))
    if error_type is not None:
        details["errorType"] = str(error_type)[:80]

    payload = {
        "pipeline": SHOE_RECOMMENDATION_PIPELINE,
        "requestId": request_id,
        "measurementSessionId": measurement_session_id,
        "event": event,
        "stage": stage,
        "message": message,
        "details": details,
    }
    log_method = (
        shoe_progress_logger.warning
        if level == "warning"
        else shoe_progress_logger.info
    )
    log_method(
        "[신발 추천][세션 %s][요청 %s] %s | pipeline_progress %s",
        measurement_session_id,
        request_id[:8],
        message,
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def log_upstream_error(route_name: str, endpoint: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    body = response.text.replace("\n", "\\n")
    if len(body) > UPSTREAM_ERROR_BODY_LIMIT:
        body = f"{body[:UPSTREAM_ERROR_BODY_LIMIT]}..."

    logger.warning(
        "%s upstream returned %s from %s. response_body=%s",
        route_name,
        response.status_code,
        endpoint,
        body,
    )


@router.post(
    "/tina-pedis",
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
    deprecated=True,
    summary="발등 무좀 분석 후 발바닥 입력 대기",
    description=(
        "왼발 사진과 오른발 사진을 각각 1장씩 업로드하면 서버 내부 AI가 무좀 의심 영역과 염증/자극 영역을 분석합니다. "
        "이 호환용 API는 더 이상 GPT 또는 리포트 백엔드를 즉시 호출하지 않습니다. 분석 결과를 측정 세션과 "
        "Bearer 토큰 기준으로 잠시 보관한 뒤, 같은 정보로 plantar-footprint API에 발바닥 사진 두 장이 들어오면 "
        "발등과 발바닥을 종합하여 최종 무좀 리포트를 저장합니다."
    ),
    responses={
        202: {"description": "발등 분석 결과를 보관했으며 발바닥 사진 입력을 기다립니다."},
        400: {"description": "업로드한 이미지가 비어 있습니다."},
        413: {"description": "업로드한 이미지 용량이 너무 큽니다."},
        422: {"description": "요청 폼 데이터가 올바르지 않습니다."},
        500: {"description": "AI 분석 중 오류가 발생했습니다."},
    },
)
async def create_tinea_pedis_report(
    measurementSessionId: int = Form(1, ge=1, description="측정 세션 ID입니다. 기본값은 1입니다."),
    leftFootImage: UploadFile = File(..., description="분석할 왼발 원본 사진 1장을 업로드하세요."),
    rightFootImage: UploadFile = File(..., description="분석할 오른발 원본 사진 1장을 업로드하세요."),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> Response:
    left_image_bytes = await read_upload_image(leftFootImage, "leftFootImage")
    right_image_bytes = await read_upload_image(rightFootImage, "rightFootImage")

    try:
        left_analysis = await run_in_threadpool(
            analyze_foot_image,
            left_image_bytes,
            leftFootImage.filename,
            "left",
        )
        right_analysis = await run_in_threadpool(
            analyze_foot_image,
            right_image_bytes,
            rightFootImage.filename,
            "right",
        )
        combined_suspicion_map = await run_in_threadpool(
            combine_png_images_side_by_side,
            left_analysis.suspicion_map_png,
            right_analysis.suspicion_map_png,
        )
        combined_photo_overlay = await run_in_threadpool(
            combine_png_images_side_by_side,
            left_analysis.photo_overlay_png,
            right_analysis.photo_overlay_png,
        )
        combined_clean_original = await run_in_threadpool(
            combine_png_images_side_by_side,
            left_analysis.clean_original_png,
            right_analysis.clean_original_png,
        )
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    stage_dorsal_tinea(
        measurementSessionId,
        f"{credentials.scheme} {credentials.credentials}",
        left_analysis,
        right_analysis,
        combined_suspicion_map,
        combined_photo_overlay,
        combined_clean_original,
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "measurementSessionId": measurementSessionId,
            "status": "waitingForPlantar",
            "message": (
                "발등 무좀 분석을 보관했습니다. 같은 measurementSessionId와 "
                "Authorization으로 발바닥 사진 두 장을 전송해 주세요."
            ),
            "nextEndpoint": "/api/reports/daily-foot-analysis/plantar-footprint",
            "expiresInSeconds": settings.tinea_handoff_ttl_seconds,
            "analysis": {
                "left": {
                    "fungalSuspicionSafetyScore": left_analysis.fungal_safety_score,
                    "skinReactionSafetyScore": left_analysis.skin_reaction_safety_score,
                },
                "right": {
                    "fungalSuspicionSafetyScore": right_analysis.fungal_safety_score,
                    "skinReactionSafetyScore": right_analysis.skin_reaction_safety_score,
                },
            },
        },
    )


@router.post(
    "/hallux-valgus",
    include_in_schema=False,
    summary="왼발/오른발 사진 분석 후 무지외반 리포트 저장 요청",
    description=(
        "왼발 사진과 오른발 사진을 각각 1장씩 업로드하면 서버 내부 AI가 발 외곽선을 추출하고 "
        "엄지 중심축과 내측 윤곽 기준축 사이의 가시적 편위각을 계산합니다. 분석 이미지는 투명 배경에 "
        "발 외곽선, 두 축과 교점을 표시하며, 무좀 리포트와 같은 방식으로 리포트 서버에 multipart "
        "요청을 전달합니다."
    ),
    responses={
        200: {"description": "리포트 서버에서 반환한 응답입니다."},
        201: {"description": "리포트 서버에서 저장 성공 후 반환한 응답입니다."},
        400: {"description": "업로드한 이미지가 비어 있습니다."},
        413: {"description": "업로드한 이미지 용량이 너무 큽니다."},
        500: {"description": "AI 분석 중 오류가 발생했습니다."},
        502: {"description": "리포트 서버 요청에 실패했습니다."},
    },
)
async def create_hallux_valgus_report(
    measurementSessionId: int = Form(1, ge=1, description="측정 세션 ID입니다. 기본값은 1입니다."),
    leftFootImage: UploadFile = File(..., description="분석할 왼발 원본 사진 1장을 업로드하세요."),
    rightFootImage: UploadFile = File(..., description="분석할 오른발 원본 사진 1장을 업로드하세요."),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> Response:
    left_image_bytes = await read_upload_image(leftFootImage, "leftFootImage")
    right_image_bytes = await read_upload_image(rightFootImage, "rightFootImage")

    try:
        left_analysis = await run_in_threadpool(
            analyze_hallux_valgus_image,
            left_image_bytes,
            leftFootImage.filename,
            "left",
        )
        right_analysis = await run_in_threadpool(
            analyze_hallux_valgus_image,
            right_image_bytes,
            rightFootImage.filename,
            "right",
        )
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    local_score_analysis_text = score_analysis_text(left_analysis.angle_degree, right_analysis.angle_degree)
    generated_score_analysis_text = await generate_hallux_score_analysis_text(
        left_analysis.angle_degree,
        right_analysis.angle_degree,
        local_score_analysis_text,
    )
    report_request = HalluxValgusReportRequest(
        measurement_session_id=measurementSessionId,
        left_toe_angle_degree=left_analysis.angle_degree,
        right_toe_angle_degree=right_analysis.angle_degree,
        score_analysis_text=generated_score_analysis_text,
    )
    forwarded_request = json.dumps(report_request.model_dump(by_alias=True), ensure_ascii=False)

    files = {
        "leftFootImage": (
            "hallux_valgus_left.png",
            left_analysis.analysis_png,
            "image/png",
        ),
        "rightFootImage": (
            "hallux_valgus_right.png",
            right_analysis.analysis_png,
            "image/png",
        ),
    }
    headers = {
        "accept": "*/*",
        "Authorization": f"{credentials.scheme} {credentials.credentials}",
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            upstream_response = await client.post(
                settings.hallux_valgus_report_endpoint,
                headers=headers,
                data={"request": forwarded_request},
                files=files,
            )
            log_upstream_error("hallux-valgus", settings.hallux_valgus_report_endpoint, upstream_response)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Report server request failed: {exc}",
        ) from exc

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type"),
    )


def upstream_response_summary(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "")
    try:
        body = response.json() if response.content else None
    except (json.JSONDecodeError, ValueError):
        body = response.text
    return {
        "statusCode": response.status_code,
        "contentType": content_type or None,
        "body": body,
    }


def integrated_analysis_summary(result: IntegratedFootAnalysisResult) -> dict:
    geometry = result.geometry
    feet = {}
    for side in ("left", "right"):
        prepared = geometry.feet[side]
        tinea = result.tinea[side]
        hallux = result.hallux_valgus[side]
        feet[side] = {
            "boardSide": prepared.board_side,
            "lengthMm": prepared.length_mm,
            "ballWidthMm": prepared.ball_width_mm,
            "segmentationConfidence": prepared.segmentation_confidence,
            "lengthDetails": prepared.length_details,
            "ballWidthDetails": prepared.ball_width_details,
            "tineaPedis": {
                "fungalSuspicionSafetyScore": tinea.fungal_safety_score,
                "skinReactionSafetyScore": tinea.skin_reaction_safety_score,
                "scoreReliability": tinea.metrics.get("score_reliability"),
                "fungalRegions": tinea.metrics.get("fungal_regions", []),
                "inflammationRegions": tinea.metrics.get(
                    "inflammation_regions", []
                ),
                "footMaskSource": tinea.metrics.get("foot_mask_source"),
            },
            "halluxValgus": {
                "angleDegree": hallux.angle_degree,
                "analysisText": hallux.analysis_text,
                "modelScore": hallux.score,
                "footMaskSource": hallux.metrics.get("foot_mask_source"),
            },
        }

    return {
        "inputImage": {
            "filename": geometry.original_filename,
            "width": geometry.input_width,
            "height": geometry.input_height,
        },
        "measurementValid": geometry.measurement_valid,
        "measurementStatus": geometry.measurement_status,
        "measurementInvalidReasons": list(geometry.measurement_invalid_reasons),
        "orientationTransform": geometry.orientation_transform,
        "detectedMarkerIds": list(geometry.detected_marker_ids),
        "missingMarkerIds": list(geometry.missing_marker_ids),
        "lensCorrection": geometry.lens_correction,
        "globalCalibration": geometry.global_calibration,
        "feet": feet,
    }


@router.post(
    "/integrated-foot-analysis",
    include_in_schema=False,
    summary="한 장의 양발 사진으로 치수·무좀·무지외반 통합 분석",
    description=(
        "1280x720 원본 양발 사진 한 장을 업로드하면 렌즈 왜곡 보정, "
        "ArUco 평면 보정, 발 경계/좌우 분리, 발길이·발볼, 무좀, "
        "무지외반 분석을 한 번의 공통 파이프라인으로 수행합니다. "
        "분석 후 기존 무좀/무지외반 리포트 서버에도 각각 저장 요청합니다."
    ),
    responses={
        200: {"description": "통합 분석 및 두 리포트 저장 성공"},
        207: {"description": "분석은 완료됐지만 리포트 서버 중 하나 이상이 오류 응답"},
        400: {"description": "업로드 이미지가 비어 있음"},
        413: {"description": "업로드 이미지 용량 초과"},
        422: {"description": "해상도, 마커, 보정 또는 발 분리 조건 불충족"},
        500: {"description": "질환 AI 분석 실패"},
        502: {"description": "리포트 서버 연결 실패"},
    },
)
async def create_integrated_foot_report(
    measurementSessionId: int = Form(1, ge=1, description="측정 세션 ID"),
    footImage: UploadFile = File(
        ...,
        description="ArUco 마커와 양발이 함께 보이는 원본 1280x720 사진 한 장",
    ),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> JSONResponse:
    image_bytes = await read_upload_image(footImage, "footImage")
    try:
        analysis = await run_in_threadpool(
            analyze_integrated_foot_photo,
            image_bytes,
            footImage.filename,
        )
    except AnalysisError as exc:
        detail = str(exc)
        client_input_error = any(
            marker in detail.lower()
            for marker in (
                "requires an unmodified",
                "marker",
                "calibration",
                "did not detect",
                "did not produce both feet",
                "segmentation returned an empty",
            )
        )
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
                if client_input_error
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=detail,
        ) from exc

    geometry = analysis.geometry
    if not geometry.measurement_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "ArUco foot measurement is not reliable enough to save.",
                "measurementStatus": geometry.measurement_status,
                "measurementInvalidReasons": list(
                    geometry.measurement_invalid_reasons
                ),
                "detectedMarkerIds": list(geometry.detected_marker_ids),
                "missingMarkerIds": list(geometry.missing_marker_ids),
            },
        )

    left_tinea = analysis.tinea["left"]
    right_tinea = analysis.tinea["right"]
    left_hallux = analysis.hallux_valgus["left"]
    right_hallux = analysis.hallux_valgus["right"]

    (
        combined_suspicion_map,
        combined_photo_overlay,
        combined_clean_original,
    ) = await asyncio.gather(
        run_in_threadpool(
            combine_png_images_side_by_side,
            left_tinea.suspicion_map_png,
            right_tinea.suspicion_map_png,
        ),
        run_in_threadpool(
            combine_png_images_side_by_side,
            left_tinea.photo_overlay_png,
            right_tinea.photo_overlay_png,
        ),
        run_in_threadpool(
            combine_png_images_side_by_side,
            left_tinea.clean_original_png,
            right_tinea.clean_original_png,
        ),
    )
    fallback_hallux_text = score_analysis_text(
        left_hallux.angle_degree,
        right_hallux.angle_degree,
    )
    hallux_report_text = await generate_hallux_score_analysis_text(
        left_hallux.angle_degree,
        right_hallux.angle_degree,
        fallback_hallux_text,
    )

    hallux_request = HalluxValgusReportRequest(
        measurement_session_id=measurementSessionId,
        left_toe_angle_degree=left_hallux.angle_degree,
        right_toe_angle_degree=right_hallux.angle_degree,
        score_analysis_text=hallux_report_text,
    )
    left_geometry = geometry.feet["left"]
    right_geometry = geometry.feet["right"]
    metrics_request = FootMetricsReportRequest(
        measurement_session_id=measurementSessionId,
        measured_left_foot_size_mm=left_geometry.length_mm,
        measured_right_foot_size_mm=right_geometry.length_mm,
        left_foot_width_mm=left_geometry.ball_width_mm,
        right_foot_width_mm=right_geometry.ball_width_mm,
    )
    headers = {
        "accept": "*/*",
        "Authorization": f"{credentials.scheme} {credentials.credentials}",
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            hallux_response, metrics_response = await asyncio.gather(
                client.post(
                    settings.hallux_valgus_report_endpoint,
                    headers=headers,
                    data={
                        "request": json.dumps(
                            hallux_request.model_dump(by_alias=True),
                            ensure_ascii=False,
                        )
                    },
                    files={
                        "leftFootImage": (
                            "hallux_valgus_left.png",
                            left_hallux.analysis_png,
                            "image/png",
                        ),
                        "rightFootImage": (
                            "hallux_valgus_right.png",
                            right_hallux.analysis_png,
                            "image/png",
                        ),
                    },
                ),
                client.post(
                    settings.daily_metrics_report_endpoint,
                    headers=headers,
                    json=metrics_request.model_dump(by_alias=True),
                ),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Report server request failed: {exc}",
        ) from exc

    log_upstream_error(
        "integrated-foot-analysis/hallux-valgus",
        settings.hallux_valgus_report_endpoint,
        hallux_response,
    )
    log_upstream_error(
        "integrated-foot-analysis/metrics",
        settings.daily_metrics_report_endpoint,
        metrics_response,
    )
    all_reports_ok = all(
        200 <= response.status_code < 300
        for response in (hallux_response, metrics_response)
    )
    if all_reports_ok:
        stage_dorsal_tinea(
            measurementSessionId,
            headers["Authorization"],
            left_tinea,
            right_tinea,
            combined_suspicion_map,
            combined_photo_overlay,
            combined_clean_original,
        )

    response_payload = {
        "measurementSessionId": measurementSessionId,
        "analysis": integrated_analysis_summary(analysis),
        "reports": {
            "tineaPedis": {
                "status": (
                    "waitingForPlantar"
                    if all_reports_ok
                    else "notStagedUpstreamFailure"
                )
            },
            "halluxValgus": upstream_response_summary(hallux_response),
            "metrics": upstream_response_summary(metrics_response),
        },
    }
    return JSONResponse(
        content=jsonable_encoder(response_payload),
        status_code=(
            status.HTTP_200_OK
            if all_reports_ok
            else status.HTTP_207_MULTI_STATUS
        ),
    )


@router.post(
    "/foot-type-text",
    response_model=FootTypeTextGenerationResponse,
    summary="완료된 발 분석 결과로 신발 목록용 발 타입 문구 생성",
    description=(
        "Feetfit_Server가 완료된 정확한 측정 세션의 분석값을 내부 인증으로 전달하면 "
        "GPT가 신발 선택에 가장 유용한 근거를 고릅니다. 최종 문구는 검증된 근거별 한국어 문장으로 "
        "생성해 반환합니다. DB 저장은 Feetfit_Server가 같은 세션을 다시 검증한 뒤 직접 수행하며, "
        "원시 길이/너비만으로 아치나 발볼 유형을 추정하지 않습니다."
    ),
    responses={
        200: {"description": "근거가 검증된 발 타입 문구를 생성했습니다."},
        403: {"description": "내부 API 키가 없거나 일치하지 않습니다."},
        422: {"description": "완료 세션 또는 분류된 발 분석 근거가 없습니다."},
    },
)
async def create_foot_type_text_report(
    request: FootTypeTextGenerationRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    _internal_api_key: None = Security(require_internal_api_key),
) -> FootTypeTextGenerationResponse:
    report_text = await generate_foot_type_text(request.analysis)
    return FootTypeTextGenerationResponse(
        measurement_session_id=request.measurement_session_id,
        facts_hash=request.facts_hash,
        type_text=report_text.type_text,
        evidence_id=report_text.evidence_id,
        source=report_text.source,
    )


@router.post(
    "/shoe-recommendations",
    summary="측정 세션 기준 신발 전체 발 적합도 산출 및 저장 요청",
    description=(
        "측정 세션 ID와 Bearer 토큰을 Feetfit_Server 내부 조회 API에 전달하여 해당 완료 세션의 발 분석, "
        "MUSINSA 리뷰, RunRepeat 원본 실측 context를 페이지 끝까지 조회합니다. 사용자 발 측정값과 RunRepeat "
        "실측값의 TEMPORARY_HEURISTIC(NOT_CLINICALLY_VALIDATED) 정책으로 모든 신발의 fitScore와 "
        "FOREFOOT/HEEL/INSOLE risk를 계산하고, BGE-M3로 해당 신발의 MUSINSA 리뷰만 선별합니다. "
        "계산 결과와 함께 MUSINSA 리뷰의 검증된 근거로 pointSummary를 만들고, 현재 세션의 "
        "발 압력·발 치수·발 환경 및 신발 적합도 근거로 부위별 reviewSummary도 즉시 생성해 "
        "한 번에 저장합니다. 내부 조회 실패 시 shared DB로 우회하지 않습니다. "
        "상단 Authorize 버튼에서 Bearer 토큰을 먼저 입력하세요."
    ),
    responses={
        200: {"description": "리포트 서버에서 반환한 응답입니다."},
        404: {"description": "측정 세션 또는 발 분석 데이터를 찾을 수 없습니다."},
        409: {"description": "다른 신발 추천 배치가 BGE-M3 런타임을 사용 중입니다."},
        422: {"description": "완료 세션 입력 또는 RunRepeat 정량 데이터가 불충분합니다."},
        500: {"description": "AI 계산 중 오류가 발생했습니다."},
        502: {"description": "리포트 서버 요청에 실패했습니다."},
    },
)
async def create_shoe_recommendations_report(
    request: ShoeRecommendationTriggerRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    _internal_api_key: None = Security(require_internal_api_key),
) -> Response:
    request_id = uuid4().hex
    measurement_session_id = request.measurement_session_id
    started_at = time.perf_counter()
    _log_shoe_recommendation_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        event="request_received",
        stage="request",
        message="신발 추천 요청이 들어왔습니다. 추천에 필요한 데이터를 불러와 분석을 시작합니다.",
    )

    def log_failure(
        *, stage: str, message: str, status_code: int, error_type: str
    ) -> None:
        _log_shoe_recommendation_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            event="recommendation_failed",
            stage=stage,
            message=message,
            level="warning",
            status_code=status_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
            error_type=error_type,
        )

    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    context_fetch_started_at = time.perf_counter()
    try:
        server_client = ShoeServerClient(authorization_header)
        context = await server_client.fetch_recommendation_context(measurement_session_id)
    except ShoeServerConfigurationError as exc:
        log_failure(
            stage="context-fetch",
            message="신발 추천 서버 설정 오류로 추천 데이터를 불러오지 못했습니다.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ShoeServerClientError as exc:
        upstream_status = exc.status_code
        if upstream_status is not None and 400 <= upstream_status < 500:
            log_failure(
                stage="context-fetch",
                message="신발 추천에 필요한 측정 결과와 신발 데이터를 불러오지 못했습니다.",
                status_code=upstream_status,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=upstream_status, detail=str(exc)) from exc
        log_failure(
            stage="context-fetch",
            message="신발 추천 데이터를 불러오는 중 서버 간 통신에 실패했습니다.",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        log_failure(
            stage="context-fetch",
            message="신발 추천 데이터를 준비하는 중 AI 서버 오류가 발생했습니다.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Shoe recommendation context preparation failed.",
        ) from exc

    _log_shoe_recommendation_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        event="stage_completed",
        stage="context-fetch",
        message="신발 추천에 필요한 측정 결과와 신발 데이터를 모두 불러왔습니다.",
        duration_ms=round((time.perf_counter() - context_fetch_started_at) * 1000),
    )

    recommendation_compute_started_at = time.perf_counter()
    try:
        batch = await run_in_threadpool(compute_shoe_recommendations, context)
    except ShoeRecommendationBusyError as exc:
        log_failure(
            stage="recommendation-compute",
            message="다른 신발 추천 작업이 실행 중이어서 이번 요청을 처리하지 못했습니다.",
            status_code=status.HTTP_409_CONFLICT,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ShoeRecommendationError as exc:
        log_failure(
            stage="recommendation-compute",
            message="신발 적합도와 추천 근거를 계산하지 못했습니다.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log_failure(
            stage="recommendation-compute",
            message="신발 적합도를 계산하는 중 AI 서버 오류가 발생했습니다.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Shoe recommendation calculation failed.",
        ) from exc

    _log_shoe_recommendation_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        event="stage_completed",
        stage="recommendation-compute",
        message="신발 적합도와 추천 근거 계산을 완료했습니다.",
        recommendation_count=len(batch.items),
        duration_ms=round((time.perf_counter() - recommendation_compute_started_at) * 1000),
    )

    payload_started_at = time.perf_counter()
    try:
        forward_request = ShoeRecommendationForwardRequest(
            measurement_session_id=batch.measurement_session_id,
            recommendations=[
                ShoeRecommendationItemPayload(
                    shoe_id=item.shoe_id,
                    fit_score=item.fit_score,
                    point_summary=item.point_summary,
                    reasons=[
                        ShoeRecommendationReasonPayload(
                            reason_type=reason.reason_type,
                            title=reason.title,
                            risk_level=reason.risk_level,
                            review_summary=reason.review_summary,
                            review_ids=reason.review_ids,
                        )
                        for reason in item.reasons
                    ],
                )
                for item in batch.items
            ],
        )
    except Exception as exc:
        log_failure(
            stage="recommendation-payload",
            message="신발 추천 결과를 백엔드 저장 형식으로 준비하지 못했습니다.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Shoe recommendation payload preparation failed.",
        ) from exc

    _log_shoe_recommendation_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        event="stage_completed",
        stage="recommendation-payload",
        message="신발 추천 결과를 백엔드 저장 형식으로 준비했습니다.",
        recommendation_count=len(batch.items),
        duration_ms=round((time.perf_counter() - payload_started_at) * 1000),
    )

    recommendation_save_started_at = time.perf_counter()
    try:
        upstream_response = await server_client.forward_recommendations(forward_request)
    except ShoeServerClientError as exc:
        log_failure(
            stage="recommendation-save",
            message="신발 추천 결과를 백엔드에 저장하지 못했습니다.",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log_failure(
            stage="recommendation-save",
            message="신발 추천 결과를 저장하는 중 AI 서버 오류가 발생했습니다.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Shoe recommendation save request failed.",
        ) from exc

    _log_shoe_recommendation_progress(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        event="stage_completed",
        stage="recommendation-save",
        message="백엔드의 신발 추천 저장 응답을 받았습니다.",
        status_code=upstream_response.status_code,
        recommendation_count=len(batch.items),
        duration_ms=round((time.perf_counter() - recommendation_save_started_at) * 1000),
    )

    duration_ms = round((time.perf_counter() - started_at) * 1000)
    if 200 <= upstream_response.status_code < 300:
        _log_shoe_recommendation_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            event="recommendation_completed",
            stage="recommendation-save",
            message="신발 추천 계산이 완료되었고 백엔드에서 저장 성공 응답을 받았습니다.",
            status_code=upstream_response.status_code,
            recommendation_count=len(batch.items),
            duration_ms=duration_ms,
        )
    else:
        _log_shoe_recommendation_progress(
            request_id=request_id,
            measurement_session_id=measurement_session_id,
            event="recommendation_failed",
            stage="recommendation-save",
            message="신발 추천 계산은 완료했지만 백엔드 저장에 실패했습니다.",
            level="warning",
            status_code=upstream_response.status_code,
            recommendation_count=len(batch.items),
            duration_ms=duration_ms,
            error_type="UpstreamHTTPError",
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type"),
        headers={
            "X-Feetfit-Pipeline": SHOE_RECOMMENDATION_PIPELINE,
            "X-Feetfit-Request-Id": request_id,
        },
    )


async def read_upload_image(upload: UploadFile, field_name: str) -> bytes:
    image_bytes = await upload.read()
    if not image_bytes:
        logger.warning(
            "%s upload is empty. filename=%s content_type=%s",
            field_name,
            upload.filename,
            upload.content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} 이미지가 비어 있습니다.",
        )
    if len(image_bytes) > settings.max_upload_size_bytes:
        logger.warning(
            "%s upload is too large. filename=%s content_type=%s size=%s max_size=%s",
            field_name,
            upload.filename,
            upload.content_type,
            len(image_bytes),
            settings.max_upload_size_bytes,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field_name} 이미지 용량이 {settings.max_upload_size_bytes} bytes를 초과했습니다.",
        )
    return image_bytes
