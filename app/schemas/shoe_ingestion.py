"""Contracts for the administrator-only verified shoe-ingestion job API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VERIFIED_INGESTION_ACTION = "RESTORE_VERIFIED_SHOE_CATALOG"
VERIFIED_INGESTION_MANIFEST_SHA256 = (
    "afd64dbea59828968e62c0a8e840994794f02e805a8d208f9e28439de14dc46e"
)


class ShoeIngestionJobStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    CANARY_RUNNING = "CANARY_RUNNING"
    FULL_RUNNING = "FULL_RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_MANUAL_COMMIT_CHECK = "NEEDS_MANUAL_COMMIT_CHECK"


class ShoeIngestionExpectedCounts(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    shoes: int = 338
    reviews: int = 8_517
    runrepeat_snapshots: int = Field(338, alias="runRepeatSnapshots")
    raw_metrics: int = Field(3_029, alias="rawMetrics")


class ShoeIngestionJobStartRequest(BaseModel):
    """A confirmation-only request; connection details are never client input."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
            }
        },
    )

    action: Literal["RESTORE_VERIFIED_SHOE_CATALOG"]
    target: str = Field(min_length=1, max_length=64)
    manifest_sha256: str = Field(
        alias="manifestSha256",
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class ShoeIngestionJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    status: ShoeIngestionJobStatus
    stage: str | None = None
    target: str
    manifest_sha256: str = Field(alias="manifestSha256")
    status_url: str = Field(alias="statusUrl")
    expected: ShoeIngestionExpectedCounts = Field(
        default_factory=ShoeIngestionExpectedCounts
    )
    progress: dict[str, int] = Field(default_factory=dict)
    created_at: str = Field(alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")
    failure_code: str | None = Field(default=None, alias="failureCode")
