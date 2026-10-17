from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.shoe_server import ReasonType, RiskLevel


class ShoeRecommendationTriggerRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)


class ShoeRecommendationReasonPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reason_type: ReasonType = Field(alias="reasonType")
    title: str
    risk_level: RiskLevel = Field(alias="riskLevel")
    # 신규 추천 배치는 개인화 문장을 즉시 채운다. null은 과거 클라이언트와의
    # 역호환 및 부분 실패 데이터 수용을 위해서만 허용한다.
    review_summary: str | None = Field(default=None, alias="reviewSummary")
    review_ids: list[int] = Field(alias="reviewIds", max_length=3)

    @model_validator(mode="after")
    def validate_unique_review_ids(self):
        if any(review_id < 1 for review_id in self.review_ids):
            raise ValueError("reviewIds must contain positive ids")
        if len(self.review_ids) != len(set(self.review_ids)):
            raise ValueError("reviewIds must not contain duplicates")
        return self


class ShoeRecommendationItemPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    shoe_id: int = Field(alias="shoeId", ge=1)
    fit_score: float = Field(alias="fitScore", ge=0, le=100)
    # BGE-M3 선별 리뷰에서 안전한 근거를 찾으면 첫 상세 조회용 결정적
    # pointSummary를 저장한다. null은 분류 가능한 상품 근거가 없다는 뜻이다.
    point_summary: str | None = Field(default=None, alias="pointSummary")
    reasons: list[ShoeRecommendationReasonPayload] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_reason_types(self):
        reason_types = [reason.reason_type for reason in self.reasons]
        if len(reason_types) != len(set(reason_types)):
            raise ValueError("reasons must contain unique reasonType values")
        if set(reason_types) != {"FOREFOOT", "HEEL", "INSOLE"}:
            raise ValueError("reasons must contain FOREFOOT, HEEL, and INSOLE")
        return self


class ShoeRecommendationForwardRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    recommendations: list[ShoeRecommendationItemPayload] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_shoe_ids(self):
        shoe_ids = [item.shoe_id for item in self.recommendations]
        if len(shoe_ids) != len(set(shoe_ids)):
            raise ValueError("recommendations must not contain duplicate shoeId values")
        return self


class ShoeSummaryTriggerRequest(BaseModel):
    """Feetfit_Server가 신발 상세 조회 시 pointSummary가 비어 있으면 호출하는 트리거 요청."""

    model_config = ConfigDict(populate_by_name=True)

    shoe_id: int = Field(alias="shoeId", ge=1)
    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    wait_for_immediate: bool = Field(default=False, alias="waitForImmediate")


class ShoeSummaryGenerateRequest(BaseModel):
    """Server가 상세 응답 안에서 자연어 요약을 동기 생성할 때 쓰는 요청."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    shoe_id: int = Field(alias="shoeId", ge=1)
    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)


class ShoeSummaryTriggerResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    accepted: bool
    deduplicated: bool
    immediate_saved: bool | None = Field(default=None, alias="immediateSaved")


class ShoeSummaryReasonPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reason_type: ReasonType = Field(alias="reasonType")
    review_summary: str = Field(alias="reviewSummary", min_length=1)
    review_ids: list[int] = Field(alias="reviewIds", max_length=3)

    @model_validator(mode="after")
    def validate_unique_review_ids(self):
        if any(review_id < 1 for review_id in self.review_ids):
            raise ValueError("reviewIds must contain positive ids")
        if len(self.review_ids) != len(set(self.review_ids)):
            raise ValueError("reviewIds must not contain duplicates")
        return self


class ShoeSummaryForwardRequest(BaseModel):
    """생성된 pointSummary/reviewSummary를 Feetfit_Server의
    POST /api/shoes/{shoeId}/summaries로 저장 요청할 때 쓰는 바디."""

    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    point_summary: str = Field(alias="pointSummary", min_length=1)
    reasons: list[ShoeSummaryReasonPayload] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_reason_types(self):
        reason_types = [reason.reason_type for reason in self.reasons]
        if set(reason_types) != {"FOREFOOT", "HEEL", "INSOLE"}:
            raise ValueError("reasons must contain FOREFOOT, HEEL, and INSOLE")
        return self
