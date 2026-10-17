from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class PressureMeasurementIdentity(BaseModel):
    """Session and backend-issued attempt carried by a hardware pressure frame."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: StrictInt = Field(alias="measurementSessionId", ge=1)
    pressure_measurement_attempt: StrictInt = Field(
        alias="pressureMeasurementAttempt", ge=0, le=2_147_483_647,
        description="Required backend-issued attempt; initial measurements explicitly send 0.",
    )


PressureValue = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PressureHeatmapRequest(PressureMeasurementIdentity):
    """External hardware frame, separate from the immutable analysis work item.

    CamelCase aliases and the snake_case field names are both accepted.
    """

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid",
        json_schema_extra={"examples": [{
            "measurementSessionId": 34, "pressureMeasurementAttempt": 0,
            "leftPressureValues": [92.2, 1.0, 605.4, 8.6, 29.3, 8430.6, 7289.3, 34.5, 4151.6, 14356.8, 272.2, 15257.0],
            "rightPressureValues": [546.9, 752.2, 683.4, 101.4, 33.0, 1982.6, 3446.1, 64.3, 6026.4, 16111.5, 11520.5, 13116.1],
        }]},
    )

    left_pressure_values: list[PressureValue] = Field(
        alias="leftPressureValues", min_length=12, max_length=12,
        description="Exactly 12 finite, non-negative left-foot sensor values.",
    )
    right_pressure_values: list[PressureValue] = Field(
        alias="rightPressureValues", min_length=12, max_length=12,
        description=("Exactly 12 finite, non-negative right-foot hardware values in "
                     "heel-first order; the server reverses them into the shared "
                     "toe-first C0-C11 analysis layout."),
    )

    @field_validator("left_pressure_values", "right_pressure_values", mode="before")
    @classmethod
    def validate_pressure_values(cls, value: Any) -> Any:
        # Keep malformed and non-finite inputs JSON-safe in FastAPI's 422 body.
        # Numeric strings and booleans must not become valid sensor samples.
        if not isinstance(value, list):
            return None
        normalized: list[float | None] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                normalized.append(None)
                continue
            try:
                number = float(item)
            except (OverflowError, TypeError, ValueError):
                normalized.append(None)
                continue
            normalized.append(number if math.isfinite(number) else None)
        return normalized


class PressureBalanceSummaryRequest(BaseModel):
    """Exact backend summary contract for the pressure-balance metric."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    metric_type: Literal["PRESSURE_BALANCE"] = Field(alias="metricType")
    score: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    advice: tuple[str, str]


class PressureDistributionReportRequest(BaseModel):
    """Exact JSON part for backend POST /daily-foot-analysis/pressure."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    left_pressure_percent: float = Field(
        alias="leftPressurePercent",
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
    right_pressure_percent: float = Field(
        alias="rightPressurePercent",
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )


class PressureBalanceReportRequest(BaseModel):
    """Exact JSON body for backend POST /daily-foot-analysis/balance."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    balance_score: float = Field(
        alias="balanceScore",
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
    balance_comment: str = Field(alias="balanceComment", min_length=1)
