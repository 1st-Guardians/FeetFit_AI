"""Administrator-only API for launching the verified shoe catalog worker."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, Security, status

from app.core.ingestion_security import require_shoe_ingestion_admin_key
from app.schemas.shoe_ingestion import (
    ShoeIngestionJobResponse,
    ShoeIngestionJobStartRequest,
)
from app.services.shoe.verified_ingestion_jobs import (
    ShoeIngestionJobAlreadyRunningError,
    ShoeIngestionJobConfigurationError,
    ShoeIngestionJobConfirmationError,
    ShoeIngestionJobError,
    ShoeIngestionJobNotFoundError,
    ShoeIngestionJobPostLaunchError,
    ShoeIngestionJobService,
    get_shoe_ingestion_job_service,
)


router = APIRouter()


@router.post(
    "/jobs",
    response_model=ShoeIngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="검증된 신발 전체 DB 적재 작업 시작",
    description=(
        "고정된 검증 bundle의 신발 338개, MUSINSA 리뷰, RunRepeat 실측값을 "
        "Feetfit_Server 내부 API로 순차 적재하는 장기 작업을 시작합니다. "
        "연결 URL, 파일 경로, 비밀 키는 요청으로 받지 않습니다."
    ),
    responses={
        202: {"description": "작업이 등록됐습니다."},
        403: {"description": "신발 적재 관리자 키가 올바르지 않습니다."},
        409: {"description": "다른 적재 작업이 실행 중입니다."},
        422: {"description": "고정 action/target/manifest 확인값이 다릅니다."},
        503: {"description": "기능 또는 안전한 서버 연결 설정이 준비되지 않았습니다."},
    },
)
def start_verified_shoe_ingestion_job(
    request: ShoeIngestionJobStartRequest,
    response: Response,
    _admin: None = Security(require_shoe_ingestion_admin_key),
    service: Annotated[
        ShoeIngestionJobService, Depends(get_shoe_ingestion_job_service)
    ] = None,
) -> ShoeIngestionJobResponse:
    try:
        result = service.start_job(
            action=request.action,
            target=request.target,
            manifest_sha256=request.manifest_sha256,
        )
    except ShoeIngestionJobConfirmationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ShoeIngestionJobAlreadyRunningError as exc:
        detail: dict[str, str] = {"code": "SHOE_INGESTION_JOB_ACTIVE"}
        if exc.job_id:
            detail.update(
                {
                    "jobId": exc.job_id,
                    "statusUrl": f"/api/admin/shoe-ingestion/jobs/{exc.job_id}",
                }
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        ) from exc
    except ShoeIngestionJobConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ShoeIngestionJobPostLaunchError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SHOE_INGESTION_POST_LAUNCH_TRACKING_FAILED",
                "jobId": exc.job_id,
                "statusUrl": f"/api/admin/shoe-ingestion/jobs/{exc.job_id}",
            },
        ) from exc
    except ShoeIngestionJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    response.headers["Location"] = result.status_url
    return result


@router.get(
    "/jobs/{job_id}",
    response_model=ShoeIngestionJobResponse,
    summary="검증된 신발 DB 적재 작업 상태 조회",
    responses={
        200: {"description": "작업 상태와 안전한 개수 진행 정보입니다."},
        403: {"description": "신발 적재 관리자 키가 올바르지 않습니다."},
        404: {"description": "작업을 찾을 수 없습니다."},
        503: {"description": "신발 적재 기능이 비활성화됐습니다."},
    },
)
def get_verified_shoe_ingestion_job(
    job_id: str,
    _admin: None = Security(require_shoe_ingestion_admin_key),
    service: Annotated[
        ShoeIngestionJobService, Depends(get_shoe_ingestion_job_service)
    ] = None,
) -> ShoeIngestionJobResponse:
    try:
        return service.get_job(job_id)
    except ShoeIngestionJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job was not found.",
        ) from exc
    except ShoeIngestionJobConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ShoeIngestionJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
