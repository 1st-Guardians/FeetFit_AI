from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PhotoMetricType = Literal[
    "HALLUX_VALGUS",
    "ATHLETES_FOOT",
    "SKIN_IRRITATION",
]


class PhotoMetricSummaryRequest(BaseModel):
    """Exact JSON contract for one photo-derived daily summary metric."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    metric_type: PhotoMetricType = Field(alias="metricType")
    score: float = Field(ge=0, le=100)
    advice: tuple[str, str]


class PhotoAnalysisErrorDetail(BaseModel):
    """Safe, stage-aware failure contract for the photo-analysis pipeline."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    pipeline: str = Field(description="Stable AI pipeline version identifier.")
    stage: str = Field(description="The exact pipeline stage that failed.")
    completed_stages: list[str] = Field(
        alias="completedStages",
        description="Ordered stages completed before this failure.",
    )
    request_id: str = Field(
        alias="requestId",
        description="Safe correlation ID; it is never used as a cache key.",
    )
    photo_capture_attempt: int | None = Field(
        default=None, alias="photoCaptureAttempt", ge=0, le=3,
        description="Original backend-assigned capture attempt; never incremented by AI.",
    )
    error_type: str = Field(
        alias="errorType",
        description="Exception class or stable validation-error type only.",
    )
    diagnostic_code: str | None = Field(
        default=None,
        alias="diagnosticCode",
        description="Stable capture-preflight reason code without raw exception text.",
    )
    image_field: str | None = Field(
        default=None,
        alias="imageField",
        description="The hardware capture field rejected by preflight validation.",
    )
    expected_width: int | None = Field(default=None, alias="expectedWidth", ge=1)
    expected_height: int | None = Field(default=None, alias="expectedHeight", ge=1)
    actual_width: int | None = Field(default=None, alias="actualWidth", ge=0)
    actual_height: int | None = Field(default=None, alias="actualHeight", ge=0)
    detected_marker_count: int | None = Field(
        default=None,
        alias="detectedMarkerCount",
        ge=0,
    )
    required_marker_count: int | None = Field(
        default=None,
        alias="requiredMarkerCount",
        ge=1,
    )
    detected_foot_count: int | None = Field(
        default=None,
        alias="detectedFootCount",
        ge=0,
    )
    required_foot_count: int | None = Field(
        default=None,
        alias="requiredFootCount",
        ge=1,
    )
    recapture_required: bool | None = Field(
        default=None,
        alias="recaptureRequired",
        description=(
            "Whether new photos are required. ArUco rejection can be recovered "
            "in the same session only after READY_FOR_RECAPTURE is acknowledged."
        ),
    )
    same_session_recapture_allowed: bool | None = Field(
        default=None, alias="sameSessionRecaptureAllowed",
        description="True only after the backend confirms WAITING_FOR_RECAPTURE.",
    )
    status_callback_confirmed: bool | None = Field(
        default=None, alias="statusCallbackConfirmed",
        description="Whether the backend acknowledged the returned session state.",
    )
    next_measurement_status: Literal["READY_FOR_RECAPTURE"] | None = Field(
        default=None, alias="nextMeasurementStatus",
        description="Status the user explicitly requests with the retake button.",
    )
    should_disconnect: bool | None = Field(default=None, alias="shouldDisconnect")
    photo_recapture_count: int | None = Field(
        default=None, alias="photoRecaptureCount", ge=0,
    )
    max_photo_recaptures: int | None = Field(
        default=None, alias="maxPhotoRecaptures", ge=0,
    )
    failure_reason: Literal["INVALID_CAPTURE_DATA", "AI_SERVER_ERROR"] | None = Field(
        default=None, alias="failureReason",
    )
    marker_occlusion_status: Literal[
        "foot_occlusion", "no_foot_overlap", "unknown", "markers_visible",
    ] | None = Field(default=None, alias="markerOcclusionStatus")
    marker_occlusion_reason: str | None = Field(
        default=None, alias="markerOcclusionReason",
        description="Allowlisted visual diagnostic reason, never a raw model error.",
    )
    marker_occlusion_foot_detected: bool | None = Field(
        default=None, alias="markerOcclusionFootDetected",
        description="Whether usable foot components were found for the overlap check.",
    )
    marker_occlusion_method: Literal[
        "calibrated_single_marker_projection_yolo_overlap", "visible_partial_marker_corners_yolo_overlap",
    ] | None = Field(default=None, alias="markerOcclusionMethod")
    marker_visible_corner_count: int | None = Field(default=None, alias="markerVisibleCornerCount", ge=0, le=4)
    marker_covered_corner_count: int | None = Field(default=None, alias="markerCoveredCornerCount", ge=0, le=4)
    marker_pattern_score: float | None = Field(default=None, alias="markerPatternScore", ge=0, le=1)
    marker_projection_reason: str | None = Field(default=None, alias="markerProjectionReason")
    marker_partial_check_reason: str | None = Field(default=None, alias="markerPartialCheckReason")
    marker_partial_candidate_count: int | None = Field(default=None, alias="markerPartialCandidateCount", ge=0, le=1000000)
    marker_partial_edge_count: int | None = Field(default=None, alias="markerPartialEdgeCount", ge=0, le=1000000)
    marker_partial_covered_candidate_count: int | None = Field(default=None, alias="markerPartialCoveredCandidateCount", ge=0, le=1000000)
    marker_partial_pattern_candidate_count: int | None = Field(default=None, alias="markerPartialPatternCandidateCount", ge=0, le=1000000)
    marker_occlusion_detected: bool | None = Field(
        default=None, alias="markerOcclusionDetected",
        description="Visual evidence of foot overlap at a localized missing marker.",
    )
    occluded_marker_ids: list[int] | None = Field(default=None, alias="occludedMarkerIds")
    marker_occlusion_overlap_ratio: float | None = Field(
        default=None, alias="markerOcclusionOverlapRatio", ge=0, le=1,
    )
    may_start_pressure: bool | None = Field(
        default=None,
        alias="mayStartPressure",
        description="Whether hardware may proceed to pressure acquisition.",
    )
    validation_field: str | None = Field(
        default=None,
        alias="validationField",
        description="Safe generated-output field path when local validation fails.",
    )
    validation_rule: str | None = Field(
        default=None,
        alias="validationRule",
        description="Stable local output-validation rule code; never generated text.",
    )
    retryable: bool = Field(
        description="Whether retrying the same logical operation may succeed.",
    )
    partial_save: bool = Field(
        alias="partialSave",
        description="True when at least one backend report was saved before failure.",
    )
    reports: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe named save outcomes without upstream bodies or credentials.",
    )
    message: str = Field(description="Sanitized user-facing failure summary.")
    recommended_action: str | None = Field(
        default=None,
        alias="recommendedAction",
        description="Allowlisted Korean recapture guidance for the hardware or app.",
    )
    measurement_status: str | None = Field(
        default=None,
        alias="measurementStatus",
    )
    measurement_invalid_reasons: list[str] | None = Field(
        default=None,
        alias="measurementInvalidReasons",
    )
    detected_marker_ids: list[int] | None = Field(
        default=None,
        alias="detectedMarkerIds",
    )
    missing_marker_ids: list[int] | None = Field(
        default=None,
        alias="missingMarkerIds",
    )


class PhotoAnalysisErrorResponse(BaseModel):
    """FastAPI HTTPException envelope used by documented photo failures."""

    detail: PhotoAnalysisErrorDetail
