from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, Security, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme
from app.schemas.reports import TineaPedisReportRequest
from app.services.plantar_full_sam_analysis import (
    PIPELINE_VERSION as FOOTPRINT_PIPELINE_VERSION,
    PlantarFullSamAnalysisError,
    analyze_plantar_image,
    combine_transparent_visualizations,
    new_request_dir,
)
from app.services.report_text_generation import (
    build_fallback_plantar_footprint_analysis_text,
    generate_plantar_footprint_analysis_text,
    generate_tinea_report_text,
)
from app.services.tinea_analysis import (
    AnalysisError,
    build_plantar_skin_result,
    combine_png_images_side_by_side,
)
from app.services.tinea_policy import aggregate_tinea_safety_scores
from app.services.tinea_session_handoff import (
    delete_dorsal_tinea,
    get_dorsal_tinea,
)


router = APIRouter()
PIPELINE_VERSION = FOOTPRINT_PIPELINE_VERSION
logger = logging.getLogger(__name__)


def _contact_status(left_region_count: int, right_region_count: int) -> str:
    detected_sides = int(left_region_count > 0) + int(right_region_count > 0)
    if detected_sides == 2:
        return "detected"
    if detected_sides == 1:
        return "partial"
    return "inconclusive"


def _is_success_status(status_code: int) -> bool:
    return 200 <= int(status_code) < 300


async def _read_upload_image(upload: UploadFile, field_name: str) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is empty.",
        )
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field_name} exceeds {settings.max_upload_size_bytes} bytes.",
        )
    return data


def _read_mask_png(path: Path) -> np.ndarray:
    try:
        encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise PlantarFullSamAnalysisError(
            "Could not read the plantar sole mask for tinea analysis."
        ) from exc
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None or not bool(np.any(mask > 0)):
        raise PlantarFullSamAnalysisError(
            "The plantar sole mask for tinea analysis is empty or invalid."
        )
    return np.ascontiguousarray(mask)


def _read_clean_tinea_input(path: Path) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PlantarFullSamAnalysisError(
            "Could not read the clean plantar image for tinea analysis."
        ) from exc
    if not data:
        raise PlantarFullSamAnalysisError(
            "The clean plantar image for tinea analysis is empty."
        )
    return data


def _read_segmented_visualization(path: Path, side: str) -> bytes:
    """Read one side's generated transparent PNG for backend persistence."""

    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PlantarFullSamAnalysisError(
            f"Could not read the {side} plantar visualization."
        ) from exc
    if not data:
        raise PlantarFullSamAnalysisError(
            f"The {side} plantar visualization is empty."
        )
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 4:
        raise PlantarFullSamAnalysisError(
            f"The {side} plantar visualization is not a transparent PNG."
        )
    return data


def _segmentation_facts(result: object) -> dict[str, object]:
    """Return model-derived facts without making a diagnostic inference."""

    region_count = max(0, int(getattr(result, "region_count", 0)))
    contact_area_ratio = float(getattr(result, "contact_area_ratio", 0.0))
    if not np.isfinite(contact_area_ratio):
        contact_area_ratio = 0.0
    contact_area_ratio = min(1.0, max(0.0, contact_area_ratio))
    return {
        "region_count": region_count,
        "contact_area_ratio": contact_area_ratio,
        "status": "detected" if region_count > 0 else "inconclusive",
    }


def _plantar_text_fallback() -> str:
    return build_fallback_plantar_footprint_analysis_text(
        left_segmentation_facts={},
        right_segmentation_facts={},
    )


@router.post(
    "",
    include_in_schema=False,
    summary="Segment plantar contact regions for both feet.",
    responses={
        status.HTTP_200_OK: {
            "description": (
                "The left and right plantar visualizations and a non-diagnostic "
                "analysis summary were forwarded successfully."
            ),
            "headers": {
                "X-Feetfit-Contact-Status": {
                    "description": (
                        "Contact detection summary: detected for both feet, partial "
                        "for one foot, or inconclusive when neither foot has a reliable region."
                    ),
                    "schema": {
                        "type": "string",
                        "enum": ["detected", "partial", "inconclusive"],
                    },
                },
                "X-Feetfit-Tinea-Status": {
                    "description": (
                        "saved when matching dorsal and plantar results were "
                        "combined and stored; missing-dorsal when only the "
                        "plantar footprint could be stored and the plantar "
                        "photos must be resent after the top-photo request."
                    ),
                    "schema": {
                        "type": "string",
                        "enum": ["saved", "missing-dorsal"],
                    },
                }
            },
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": (
                "The local plantar analysis pipeline or report backend could not "
                "provide a valid response."
            )
        },
    },
)
async def segment_pressed_regions(
    measurementSessionId: int = Form(..., ge=1),
    leftFootImage: UploadFile = File(...),
    rightFootImage: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> Response:
    left_bytes = await _read_upload_image(leftFootImage, "leftFootImage")
    right_bytes = await _read_upload_image(rightFootImage, "rightFootImage")
    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    request_id, output_dir = new_request_dir()
    stage = "fungal-footprint-left-analysis"
    dorsal_handoff = None
    tinea_request_json: str | None = None
    sole_suspicion_map: bytes | None = None
    sole_photo_overlay: bytes | None = None
    sole_clean_original: bytes | None = None
    try:
        left_result = await run_in_threadpool(
            analyze_plantar_image,
            left_bytes,
            "left",
            output_dir,
        )
        stage = "fungal-footprint-right-analysis"
        right_result = await run_in_threadpool(
            analyze_plantar_image,
            right_bytes,
            "right",
            output_dir,
        )
        stage = "combine-transparent-png"
        await run_in_threadpool(
            combine_transparent_visualizations,
            left_result.visualization_path,
            right_result.visualization_path,
            output_dir / "plantar_footprint.png",
        )

        stage = "read-segmented-visualizations"
        left_visualization, right_visualization = await asyncio.gather(
            run_in_threadpool(
                _read_segmented_visualization,
                left_result.visualization_path,
                "left",
            ),
            run_in_threadpool(
                _read_segmented_visualization,
                right_result.visualization_path,
                "right",
            ),
        )
        left_segmentation_facts = _segmentation_facts(left_result)
        right_segmentation_facts = _segmentation_facts(right_result)

        stage = "plantar-footprint-analysis-text"
        try:
            plantar_analysis_text = await generate_plantar_footprint_analysis_text(
                left_raw_image_bytes=left_bytes,
                right_raw_image_bytes=right_bytes,
                left_segmentation_facts=left_segmentation_facts,
                right_segmentation_facts=right_segmentation_facts,
            )
        except Exception as exc:  # The report images must survive text-generation faults.
            logger.warning(
                "Plantar analysis text fallback request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            plantar_analysis_text = _plantar_text_fallback()

        dorsal_handoff = get_dorsal_tinea(
            measurementSessionId,
            authorization_header,
        )
        if dorsal_handoff is not None:
            stage = "unassessed-plantar-display"
            left_clean, right_clean = await asyncio.gather(
                run_in_threadpool(_read_clean_tinea_input, left_result.tinea_input_path),
                run_in_threadpool(_read_clean_tinea_input, right_result.tinea_input_path),
            )
            sole_clean_original = await run_in_threadpool(combine_png_images_side_by_side, left_clean, right_clean)
            sole_suspicion_map = sole_clean_original
            sole_photo_overlay = sole_clean_original
            stage = "tinea-combined-report-text"
            report_text = await generate_tinea_report_text(
                dorsal_handoff.left_result,
                dorsal_handoff.right_result,
                suspicion_map_png=dorsal_handoff.combined_suspicion_map_png,
                photo_overlay_png=dorsal_handoff.combined_photo_overlay_png,
            )
            all_tinea_results = (
                dorsal_handoff.left_result,
                dorsal_handoff.right_result,
            )
            tinea_request = TineaPedisReportRequest(
                measurement_session_id=measurementSessionId,
                fungal_suspicion_safety_score=aggregate_tinea_safety_scores(
                    result.fungal_safety_score for result in all_tinea_results
                ),
                skin_reaction_safety_score=aggregate_tinea_safety_scores(
                    result.skin_reaction_safety_score for result in all_tinea_results
                ),
                fungal_suspicion_safety_description=(
                    report_text.fungal_suspicion_safety_description
                ),
                skin_reaction_safety_description=(
                    report_text.skin_reaction_safety_description
                ),
                total_score_description=report_text.total_score_description,
            )
            tinea_request_json = json.dumps(
                tinea_request.model_dump(by_alias=True),
                ensure_ascii=False,
            )
    except (PlantarFullSamAnalysisError, AnalysisError) as exc:
        failure = {
            "pipeline": PIPELINE_VERSION,
            "stage": stage,
            "message": str(exc),
            "requestId": request_id,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.exception(
                "Could not save plantar failure diagnostics request_id=%s",
                request_id,
            )
        logger.error(
            "Plantar pipeline failed request_id=%s stage=%s error=%s",
            request_id,
            stage,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=failure,
            headers={"X-Feetfit-Pipeline": PIPELINE_VERSION},
        ) from exc

    headers = {
        "accept": "*/*",
        "Authorization": authorization_header,
    }
    network_stage = "plantar-backend-forward"
    try:
        async with httpx.AsyncClient(
            timeout=settings.report_proxy_timeout_seconds,
            trust_env=False,
        ) as client:
            upstream = await client.post(
                settings.plantar_footprint_report_endpoint,
                headers=headers,
                files={
                    "request": (
                        None,
                        json.dumps(
                            {
                                "measurementSessionId": measurementSessionId,
                                "plantarFootprintAnalysisText": plantar_analysis_text,
                            },
                            ensure_ascii=False,
                        ),
                        "application/json",
                    ),
                    "leftPlantarFootprintImage": (
                        f"left_plantar_footprint_{request_id}.png",
                        left_visualization,
                        "image/png",
                    ),
                    "rightPlantarFootprintImage": (
                        f"right_plantar_footprint_{request_id}.png",
                        right_visualization,
                        "image/png",
                    ),
                },
            )
            if _is_success_status(upstream.status_code) and dorsal_handoff is not None:
                assert tinea_request_json is not None
                assert sole_suspicion_map is not None
                assert sole_photo_overlay is not None
                assert sole_clean_original is not None
                network_stage = "tinea-backend-forward"
                tinea_upstream = await client.post(
                    settings.tinea_report_endpoint,
                    headers=headers,
                    files={
                        "request": (
                            None,
                            tinea_request_json,
                            "application/json",
                        ),
                        "suspiciousAreaMapImage": (
                            "dorsal_suspicion_map.png",
                            dorsal_handoff.combined_suspicion_map_png,
                            "image/png",
                        ),
                        # Despite the legacy field names, the report UI expects
                        # the annotated photo paired with each suspicion map.
                        "originalFootImage": (
                            "dorsal_analysis_overlay.png",
                            dorsal_handoff.combined_photo_overlay_png,
                            "image/png",
                        ),
                        "soleSuspiciousAreaMapImage": (
                            "sole_not_assessed_photo.png",
                            sole_suspicion_map,
                            "image/png",
                        ),
                        "soleOriginalFootImage": (
                            "sole_not_assessed_photo.png",
                            sole_photo_overlay,
                            "image/png",
                        ),
                    },
                )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": network_stage,
                "message": "Report server transport failed "
                f"({type(exc).__name__}).",
                "requestId": request_id,
            },
            headers={"X-Feetfit-Pipeline": PIPELINE_VERSION},
        ) from exc

    if not _is_success_status(upstream.status_code):
        try:
            upstream_detail = upstream.json()
        except ValueError:
            upstream_detail = upstream.text[:1000]
        raise HTTPException(
            status_code=upstream.status_code,
            detail={
                "pipeline": PIPELINE_VERSION,
                "stage": "backend-forward",
                "message": "The report backend returned an error.",
                "upstreamStatus": upstream.status_code,
                "upstreamDetail": upstream_detail,
                "requestId": request_id,
            },
            headers={"X-Feetfit-Pipeline": PIPELINE_VERSION},
        )

    if dorsal_handoff is not None:
        if not _is_success_status(tinea_upstream.status_code):
            try:
                upstream_detail = tinea_upstream.json()
            except ValueError:
                upstream_detail = tinea_upstream.text[:1000]
            raise HTTPException(
                status_code=tinea_upstream.status_code,
                detail={
                    "pipeline": PIPELINE_VERSION,
                    "stage": "tinea-backend-forward",
                    "message": "The tinea report backend returned an error.",
                    "upstreamStatus": tinea_upstream.status_code,
                    "upstreamDetail": upstream_detail,
                    "requestId": request_id,
                },
                headers={"X-Feetfit-Pipeline": PIPELINE_VERSION},
            )
        delete_dorsal_tinea(
            measurementSessionId,
            authorization_header,
            dorsal_handoff.generation_id,
        )

    contact_status = _contact_status(
        left_result.region_count,
        right_result.region_count,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers={
            "X-Feetfit-Pipeline": PIPELINE_VERSION,
            "X-Feetfit-Request-Id": request_id,
            "X-Feetfit-Contact-Status": contact_status,
            "X-Feetfit-Tinea-Status": (
                "saved" if dorsal_handoff is not None else "missing-dorsal"
            ),
        },
    )
