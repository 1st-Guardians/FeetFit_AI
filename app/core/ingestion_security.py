"""Authentication boundary for catalog-wide shoe ingestion operations."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings


shoe_ingestion_admin_key_scheme = APIKeyHeader(
    name="X-Shoe-Ingestion-Admin-Key",
    scheme_name="신발 DB 적재 관리자 키",
    description=(
        "검증된 신발 카탈로그 적재 작업 전용 관리자 키입니다. "
        "Feetfit_Server 내부 API 키와 다른 값을 사용하세요."
    ),
    auto_error=False,
)


def require_shoe_ingestion_admin_key(
    provided: Annotated[
        str | None, Security(shoe_ingestion_admin_key_scheme)
    ] = None,
) -> None:
    """Fail closed without revealing whether a supplied key was close or missing."""

    if not settings.shoe_ingestion_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified shoe ingestion is disabled.",
        )

    expected = settings.shoe_ingestion_admin_api_key.strip()
    server_key = settings.feetfit_server_internal_api_key.strip()
    if (
        len(expected) < 32
        or not server_key
        or hmac.compare_digest(expected, server_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified shoe ingestion authentication is not safely configured.",
        )

    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden shoe ingestion request.",
        )


ShoeIngestionAdmin = Annotated[None, Security(require_shoe_ingestion_admin_key)]
