from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class FootEnvironmentAnalysisRequest(BaseModel):
    """Strict hardware readings accepted by the environment analysis route."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    measurement_session_id: StrictInt = Field(alias="measurementSessionId", ge=1)
    before_temperature_celsius: float = Field(
        alias="beforeTemperatureCelsius",
        ge=-20.0,
        le=80.0,
    )
    before_humidity_percent: float = Field(
        alias="beforeHumidityPercent",
        ge=0.0,
        le=100.0,
    )
    after_temperature_celsius: float = Field(
        alias="afterTemperatureCelsius",
        ge=-20.0,
        le=80.0,
    )
    after_humidity_percent: float = Field(
        alias="afterHumidityPercent",
        ge=0.0,
        le=100.0,
    )

    @field_validator(
        "before_temperature_celsius",
        "before_humidity_percent",
        "after_temperature_celsius",
        "after_humidity_percent",
        mode="before",
    )
    @classmethod
    def reject_non_finite_or_coerced_numbers(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        converted = float(value)
        if not math.isfinite(converted):
            # FastAPI serializes Pydantic's rejected input into its 422 detail.
            # Replace NaN/Infinity with a JSON-safe sentinel first.
            return None
        return converted

    @field_validator("measurement_session_id", mode="before")
    @classmethod
    def reject_coerced_session_ids(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value


class FootEnvironmentSummaryRequest(BaseModel):
    """Exact backend summary JSON for the photo-independent environment metric."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    metric_type: Literal["FOOT_ENVIRONMENT"] = Field(alias="metricType")
    score: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    advice: tuple[str, str]


class FootEnvironmentBackendRequest(BaseModel):
    """Exact raw-reading contract exposed by the daily-analysis backend.

    The backend derives and persists both averages from these four readings.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    before_temperature_celsius: float = Field(
        alias="beforeTemperatureCelsius",
        ge=-20.0,
        le=80.0,
        allow_inf_nan=False,
    )
    before_humidity_percent: float = Field(
        alias="beforeHumidityPercent",
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
    after_temperature_celsius: float = Field(
        alias="afterTemperatureCelsius",
        ge=-20.0,
        le=80.0,
        allow_inf_nan=False,
    )
    after_humidity_percent: float = Field(
        alias="afterHumidityPercent",
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
