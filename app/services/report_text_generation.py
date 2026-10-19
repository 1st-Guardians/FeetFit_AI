from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import random
import re
import time
from typing import Any, Literal

import httpx
from jsonschema import Draft202012Validator

from app.core.config import settings
from app.schemas.reports import FootTypeAnalysisContext
from app.services.plantar_photo_composite import (
    PLANTAR_COMPOSITE_LABEL,
    combine_raw_plantar_photos,
)
from app.services.pressure_sensor_layout import canonicalize_right_heel_first
from app.services.tinea_policy import (
    aggregate_tinea_safety_scores,
    safety_score_indicates_tinea_signal,
)


logger = logging.getLogger(__name__)
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class ReportGenerationError(RuntimeError):
    """Provider failure with safe metadata, never raw response or patient text."""

    def __init__(self, code: str, *, retryable: bool = False, status_code: int | None = None):
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(f"Report generation failed (code={code}, status={status_code}).")


@dataclass
class _ReportRequestBudget:
    """One generation's HTTP allowance, shared with output-validation repairs."""

    maximum_attempts: int
    deadline: float
    attempts: int = 0

    @classmethod
    def for_report(cls, *, full_photo: bool = False) -> _ReportRequestBudget:
        seconds = (
            settings.openai_report_full_photo_timeout_seconds
            if full_photo else settings.openai_report_timeout_seconds
        )
        return cls(
            maximum_attempts=int(settings.openai_report_max_attempts),
            deadline=time.monotonic() + float(seconds),
        )


@dataclass(frozen=True)
class TineaReportText:
    fungal_suspicion_safety_description: str
    skin_reaction_safety_description: str
    total_score_description: str
    generation_source: Literal["OPENAI", "DETERMINISTIC_FALLBACK"] = "OPENAI"
    inflammation_analysis_enabled: bool = True
    local_policy_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FullPhotoAnalysisTexts:
    hallux_score_analysis_text: str
    fungal_suspicion_safety_description: str
    skin_reaction_safety_description: str
    total_score_description: str
    plantar_footprint_analysis_text: str
    metric_advice: dict[str, tuple[str, str]]
    generation_source: Literal["OPENAI", "DETERMINISTIC_FALLBACK"] = "OPENAI"
    generation_failure_code: str | None = None
    inflammation_analysis_enabled: bool = True
    local_policy_fields: tuple[str, ...] = ()


class FullPhotoOutputValidationError(ValueError):
    """Safe, field-specific failure for GPT full-photo output validation.

    Generated text is deliberately excluded so logs and API error handling can
    identify the failed contract without exposing model output.
    """

    def __init__(self, *, field: str, rule_code: str) -> None:
        self.field = field
        self.rule_code = rule_code
        super().__init__(
            "OpenAI full photo analysis output validation failed "
            f"(field={field}, rule={rule_code})."
        )


@dataclass(frozen=True)
class FootEnvironmentReportText:
    score: float
    advice: tuple[str, str]


@dataclass(frozen=True)
class PressureBalanceReportText:
    score: float
    left_regional_observation: str
    right_regional_observation: str
    overall_balance_observation: str
    practical_tip: str
    generation_source: Literal["OPENAI", "DETERMINISTIC_FALLBACK"] = "OPENAI"
    generation_failure_code: str | None = None

    @property
    def advice(self) -> tuple[str, str]:
        """Exact two-item backend contract: combined evidence, then action tip."""

        combined_observation = "\n".join(
            (
                self.left_regional_observation,
                self.right_regional_observation,
                self.overall_balance_observation,
            )
        )
        return combined_observation, self.practical_tip

    @property
    def balance_comment(self) -> str:
        """Single comment used by the daily-foot-analysis balance endpoint."""

        return "\n".join(
            (
                self.left_regional_observation,
                self.right_regional_observation,
                self.overall_balance_observation,
                self.practical_tip,
            )
        )


@dataclass(frozen=True)
class FootTypeEvidence:
    evidence_id: str
    category: str
    canonical_fact: str
    type_text: str


@dataclass(frozen=True)
class FootTypeReportText:
    type_text: str
    evidence_id: str
    source: str


def _compact_text(value: Any, fallback: str, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        text = fallback
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _openai_enabled() -> bool:
    return bool(settings.openai_report_text_enabled and (settings.openai_api_key or "").strip())


def _image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    raise ValueError("Unsupported or invalid image bytes; expected PNG, JPEG, or WebP.")


def _image_data_url(image_bytes: bytes) -> str:
    mime_type = _image_mime_type(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _png_data_url(image_bytes: bytes) -> str:
    """Backward-compatible alias retained for callers outside this module."""
    return _image_data_url(image_bytes)


def _extract_response_text(data: dict) -> str:
    direct_text = data.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise ReportGenerationError("MODEL_REFUSAL")
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    if parts:
        return "".join(parts)
    raise ReportGenerationError("EMPTY_OUTPUT", retryable=True)


def _provider_error_code(data: Any) -> str:
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return ""
    return str(error.get("code") or error.get("type") or "")


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", {})
    raw = headers.get("retry-after")
    if not isinstance(raw, str):
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


async def _create_structured_response(
    *,
    system_prompt: str,
    user_payload: dict,
    schema_name: str,
    json_schema: dict,
    images: list[tuple[str, bytes]] | None = None,
    image_detail: Literal["low", "high", "auto", "original"] = "low",
    max_output_tokens: int = 600,
    wait_for_response_without_timeout: bool = False,
    request_budget: _ReportRequestBudget | None = None,
) -> dict:
    content: list[dict] = [
        {
            "type": "input_text",
            "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    ]
    if settings.openai_report_include_images:
        for label, image_bytes in images or []:
            if not image_bytes:
                continue
            content.append({"type": "input_text", "text": f"분석 이미지: {label}"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(image_bytes),
                    "detail": image_detail,
                }
            )

    request_body = {
        "model": settings.openai_report_model,
        # Foot-analysis inputs may be sensitive.  Responses used for report
        # copy are not retained by OpenAI when this request-level flag is set.
        "store": False,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    reasoning_model = settings.openai_report_model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    if reasoning_model:
        request_body["reasoning"] = {"effort": settings.openai_report_reasoning_effort}
        request_body["max_output_tokens"] = max(max_output_tokens, settings.openai_report_min_output_tokens)
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    connect_timeout = float(settings.openai_report_connect_timeout_seconds)
    pool_timeout = float(settings.openai_report_pool_timeout_seconds)
    if wait_for_response_without_timeout:
        # Give full-photo uploads and generation the overall deadline below,
        # without a separate read/write timeout; keep connect/pool waits short.
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=None,
            write=None,
            pool=pool_timeout,
        )
    else:
        total_timeout = float(settings.openai_report_timeout_seconds)
        timeout = httpx.Timeout(
            total_timeout,
            connect=min(connect_timeout, total_timeout),
            pool=min(pool_timeout, total_timeout),
        )
    if request_budget is None:
        request_budget = _ReportRequestBudget.for_report(
            full_photo=wait_for_response_without_timeout,
        )
    deadline = request_budget.deadline
    maximum_attempts = request_budget.maximum_attempts
    if request_budget.attempts >= maximum_attempts:
        raise ReportGenerationError("REQUEST_ATTEMPTS_EXHAUSTED")
    token_limit = max(int(request_body["max_output_tokens"]),
                      settings.openai_report_max_output_tokens if reasoning_model else 6_000)
    billing_codes = {
        "insufficient_quota", "credit_balance_exhausted",
        "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded", "billing_hard_limit_reached",
    }
    async with httpx.AsyncClient(timeout=timeout, trust_env=settings.openai_report_trust_env) as client:
        for attempt in range(request_budget.attempts + 1, maximum_attempts + 1):
            response = None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReportGenerationError("API_TIMEOUT")
            request_budget.attempts += 1
            try:
                response = await asyncio.wait_for(
                    client.post(OPENAI_RESPONSES_URL, headers=headers, json=request_body),
                    timeout=remaining,
                )
                try:
                    data = response.json()
                except (ValueError, TypeError):
                    data = None
                if response.status_code >= 300:
                    retryable = (
                        response.status_code in {408, 409, 429}
                        or response.status_code >= 500
                    ) and _provider_error_code(data) not in billing_codes
                    raise ReportGenerationError(
                        "API_HTTP_ERROR", retryable=retryable, status_code=response.status_code,
                    )
                if not isinstance(data, dict):
                    raise ReportGenerationError("INVALID_API_JSON", retryable=True)
                status = data.get("status")
                if status == "incomplete":
                    details = data.get("incomplete_details")
                    reason = details.get("reason") if isinstance(details, dict) else None
                    if reason == "max_output_tokens" and request_body["max_output_tokens"] < token_limit:
                        request_body = {
                            **request_body,
                            "max_output_tokens": min(token_limit, request_body["max_output_tokens"] * 2),
                        }
                        raise ReportGenerationError("OUTPUT_TOKEN_LIMIT", retryable=True)
                    raise ReportGenerationError("INCOMPLETE_OUTPUT")
                if status not in (None, "completed"):
                    raise ReportGenerationError(
                        "API_RESPONSE_FAILED",
                        retryable=status == "failed" and _provider_error_code(data) in {
                            "server_error", "internal_server_error", "rate_limit_exceeded",
                        },
                    )
                try:
                    parsed = json.loads(_extract_response_text(data))
                except (ValueError, TypeError):
                    raise ReportGenerationError("INVALID_OUTPUT_JSON", retryable=True) from None
                if not isinstance(parsed, dict):
                    raise ReportGenerationError("OUTPUT_OBJECT_REQUIRED", retryable=True)
                if next(Draft202012Validator(json_schema).iter_errors(parsed), None) is not None:
                    request_body = {
                        **request_body,
                        "input": [
                            {**request_body["input"][0], "content": system_prompt + "\n\n"
                             "The previous output did not match the JSON Schema. Regenerate the complete JSON "
                             "with every required field, no extra fields, and valid types, enums and lengths."},
                            *request_body["input"][1:],
                        ],
                    }
                    raise ReportGenerationError("OUTPUT_SCHEMA_MISMATCH", retryable=True)
                return parsed
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                error = ReportGenerationError(
                    "API_TIMEOUT" if isinstance(exc, httpx.TimeoutException) else "API_CONNECTION_ERROR",
                    retryable=True,
                )
            except asyncio.TimeoutError:
                raise ReportGenerationError("API_TIMEOUT") from None
            except ReportGenerationError as exc:
                error = exc

            if not error.retryable or attempt >= maximum_attempts:
                raise error from None
            retry_after = _retry_after_seconds(response) if response is not None else None
            delay = retry_after if retry_after is not None else 2 ** (attempt - 1) + random.uniform(0, .25)
            # Never retry earlier than the server requested. If that wait is
            # outside our budget, let the caller finish with its fallback.
            if (delay > float(settings.openai_report_retry_max_delay_seconds)
                    or delay >= deadline - time.monotonic()):
                raise error from None
            logger.warning(
                "Report API retry schema=%s attempt=%s max_attempts=%s code=%s status=%s delay=%.2f",
                schema_name, attempt, maximum_attempts, error.code, error.status_code, delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("Report API retry loop exited unexpectedly.")


_ARCH_EVIDENCE = {
    "LOW": FootTypeEvidence(
        evidence_id="ARCH_LOW",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 낮은 유형으로 분류됨",
        type_text=(
            "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요. "
            "오래 걷거나 서 있으면 피로가 커질 수 있어 아치를 잘 받쳐주는 신발이 더 편안할 수 있어요."
        ),
    ),
    "NORMAL": FootTypeEvidence(
        evidence_id="ARCH_NORMAL",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 보통 유형으로 분류됨",
        type_text=(
            "발의 아치와 발바닥 접촉이 비교적 균형적인 편이에요. "
            "발을 고르게 받쳐주고 전체적으로 편안하게 맞는 신발이 잘 맞을 수 있어요."
        ),
    ),
    "HIGH": FootTypeEvidence(
        evidence_id="ARCH_HIGH",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 높은 유형으로 분류됨",
        type_text=(
            "발의 아치가 높은 편이라 발바닥의 일부 부위에 압력이 집중될 수 있어요. "
            "충격을 부드럽게 분산하고 쿠션이 안정적인 신발이 더 편안할 수 있어요."
        ),
    ),
}

_WIDTH_EVIDENCE = {
    "NARROW": FootTypeEvidence(
        evidence_id="WIDTH_NARROW",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 좁은 유형으로 분류됨",
        type_text=(
            "발볼이 슬림한 편이라 너비가 넉넉한 신발에서는 발이 안에서 움직일 수 있어요. "
            "발볼을 안정적으로 잡아주는 신발이 더 잘 맞을 수 있어요."
        ),
    ),
    "NORMAL": FootTypeEvidence(
        evidence_id="WIDTH_NORMAL",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 보통 유형으로 분류됨",
        type_text=(
            "발볼 너비가 비교적 균형적인 편이에요. "
            "발길이와 발볼이 자연스럽게 맞고 발을 안정적으로 받쳐주는 신발이 편안할 수 있어요."
        ),
    ),
    "WIDE": FootTypeEvidence(
        evidence_id="WIDTH_WIDE",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 넓은 유형으로 분류됨",
        type_text=(
            "발볼이 넓은 편이라 앞쪽이 타이트한 신발에서는 압박을 느낄 수 있어요. "
            "발볼 공간이 여유로운 신발을 고르면 더 편안할 수 있어요."
        ),
    ),
}

_PRESSURE_EVIDENCE = {
    "LEFT_DOMINANT": FootTypeEvidence(
        evidence_id="PRESSURE_LEFT_DOMINANT",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 왼발 압력 비중이 더 높은 유형으로 분류됨",
        type_text=(
            "왼발에 압력이 조금 더 실리는 편이에요. "
            "양쪽 발을 고르게 받쳐주고 뒤꿈치가 안정적인 신발이 편안할 수 있어요."
        ),
    ),
    "BALANCED": FootTypeEvidence(
        evidence_id="PRESSURE_BALANCED",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 양발 압력 분포가 균형 유형으로 분류됨",
        type_text=(
            "양쪽 발의 압력 분포가 비교적 균형적인 편이에요. "
            "발바닥을 고르게 받쳐주는 신발이 편안함을 유지하는 데 도움이 될 수 있어요."
        ),
    ),
    "RIGHT_DOMINANT": FootTypeEvidence(
        evidence_id="PRESSURE_RIGHT_DOMINANT",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 오른발 압력 비중이 더 높은 유형으로 분류됨",
        type_text=(
            "오른발에 압력이 조금 더 실리는 편이에요. "
            "양쪽 발을 고르게 받쳐주고 뒤꿈치가 안정적인 신발이 편안할 수 있어요."
        ),
    ),
}

_PLANTAR_SHAPE_EVIDENCE = FootTypeEvidence(
    evidence_id="PLANTAR_SHAPE_TENDENCY",
    category="PLANTAR_SHAPE",
    canonical_fact="발바닥 이미지에서 요족·정상·평발 형태 경향 문구가 생성됨",
    type_text=(
        "발바닥 형태 분석 결과가 생성됐어요. "
        "요족·정상·평발 구분은 발바닥 형태 문구와 다른 발 측정 결과를 함께 확인해 주세요."
    ),
)


def _pressure_balance_type(context: FootTypeAnalysisContext) -> str:
    if context.pressure_balance_type != "UNKNOWN":
        return context.pressure_balance_type
    if (
        context.left_pressure_percent is None
        or context.right_pressure_percent is None
    ):
        return "UNKNOWN"

    difference = context.left_pressure_percent - context.right_pressure_percent
    tolerance = settings.foot_type_pressure_balance_tolerance_percent
    if abs(difference) <= tolerance:
        return "BALANCED"
    return "LEFT_DOMINANT" if difference > 0 else "RIGHT_DOMINANT"


def foot_type_evidence(context: FootTypeAnalysisContext) -> list[FootTypeEvidence]:
    """Return only evidence explicitly classified by the upstream analysis.

    Raw foot dimensions are intentionally not interpreted here.  In
    particular, width/length measurements must never be used to infer a low
    arch, flat foot, or a categorical width without a validated classifier.
    """

    evidence: list[FootTypeEvidence] = []
    if context.arch_type != "UNKNOWN":
        evidence.append(_ARCH_EVIDENCE[context.arch_type])
    if context.foot_width_type != "UNKNOWN":
        evidence.append(_WIDTH_EVIDENCE[context.foot_width_type])
    pressure_balance_type = _pressure_balance_type(context)
    if pressure_balance_type != "UNKNOWN":
        evidence.append(_PRESSURE_EVIDENCE[pressure_balance_type])
    if context.plantar_footprint_analysis_text is not None:
        evidence.append(_PLANTAR_SHAPE_EVIDENCE)
    return evidence


def build_fallback_foot_type_text(
    context: FootTypeAnalysisContext,
) -> FootTypeReportText:
    candidates = foot_type_evidence(context)
    if not candidates:  # The request schema normally rejects this first.
        raise ValueError("Foot type text requires at least one classified analysis result.")
    selected = candidates[0]
    return FootTypeReportText(
        type_text=selected.type_text,
        evidence_id=selected.evidence_id,
        source="FALLBACK",
    )


async def generate_foot_type_text(
    context: FootTypeAnalysisContext,
) -> FootTypeReportText:
    """Select one grounded shoe-list message and render audited Korean copy.

    GPT only chooses among evidence IDs supplied by the completed analysis.
    The final sentence is rendered from a vetted mapping so an API response
    cannot introduce a diagnosis or an unsupported foot characteristic.
    """

    fallback = build_fallback_foot_type_text(context)
    candidates = foot_type_evidence(context)
    if not (
        _openai_enabled()
        and settings.openai_foot_type_text_enabled
    ):
        return fallback

    candidate_ids = [candidate.evidence_id for candidate in candidates]
    schema = {
        "type": "object",
        "properties": {
            "selectedEvidenceId": {
                "type": "string",
                "enum": candidate_ids,
            }
        },
        "required": ["selectedEvidenceId"],
        "additionalProperties": False,
    }
    payload = {
        "task": "신발 목록 상단에 표시할 가장 유용한 발 분석 근거 한 개 선택",
        "rules": [
            "후보에 없는 발 특성이나 의학적 진단을 추론하지 않는다.",
            "신발 선택에 가장 직접적으로 도움이 되는 근거를 우선한다.",
            "ARCH, WIDTH, PRESSURE_BALANCE가 모두 있다면 ARCH를 우선 검토한다.",
            "문구가 '이번 측정에서는'으로 시작되지 않도록 한다.",
            "반드시 candidateEvidence의 evidenceId 중 하나만 선택한다.",
        ],
        "candidateEvidence": [
            {
                "evidenceId": candidate.evidence_id,
                "category": candidate.category,
                "fact": candidate.canonical_fact,
            }
            for candidate in candidates
        ],
    }

    try:
        data = await _create_structured_response(
            system_prompt=(
                "너는 FeetFit의 신발 선택 보조 문구를 위한 근거 선택기다. "
                "제공된 구조화 분석 근거 밖의 사실은 만들지 말고 JSON 스키마만 반환한다."
            ),
            user_payload=payload,
            schema_name="foot_type_text_evidence_selection",
            json_schema=schema,
            images=None,
            max_output_tokens=80,
        )
        selected_id = data.get("selectedEvidenceId")
        selected = next(
            candidate
            for candidate in candidates
            if candidate.evidence_id == selected_id
        )
    except Exception as exc:
        logger.warning(
            "GPT foot type evidence selection failed; using fallback. error=%s",
            exc,
        )
        return fallback

    return FootTypeReportText(
        type_text=selected.type_text,
        evidence_id=selected.evidence_id,
        source="OPENAI",
    )


def _metrics(result: Any) -> dict:
    metrics = getattr(result, "metrics", None)
    return metrics if isinstance(metrics, dict) else {}


_SKIN_IMAGE_REPORT_UNAVAILABLE = "피부 상태에 대한 이미지 설명을 생성하지 못했습니다."
_SKIN_IMAGE_FALLBACK_TIP = "피부 상태 설명을 다시 요청해 주세요."
_SKIN_IMAGE_REPORT_MODE = "image_based_with_detector_disabled"
_FUNGAL_WOUND_REPORT_INSTRUCTIONS = (
    "서버는 무좀과 상처의 겹침을 모델 신뢰도로 정리하며 동률은 무좀으로 처리한다. "
    "무좀 영역 안에 포함되거나 둘러싸인 상처는 무좀으로 통합한다. "
    "최종 facts의 영역과 wound_detected만 설명 근거로 사용하고, "
    "통합되어 제외된 상처를 원본 사진에서 다시 추가하거나 무좀 안에 염증·상처가 있다고 설명하지 않는다."
)
_SKIN_IMAGE_REPORT_INSTRUCTIONS = (
    "이번 요청에서는 염증 검출기가 꺼져 있어 피부 자극 측정값은 제공되지 않는다. "
    "피부 자극 설명과 SKIN_IRRITATION 조언은 inflammation_analysis_contract.image_labels에 "
    "지정된 검출 표시 없는 실제 사진을 직접 살펴보고 작성한다. "
    "피부 항목의 null 점수나 빈 표시를 정상 검출 결과 또는 건강한 상태의 증거로 해석하지 않는다. "
    "사진에서 눈에 띄는 변화가 없으면 관찰 범위 안에서 자연스럽게 설명하고, 보이는 변화나 "
    "판단의 한계가 있으면 그대로 설명한다. 특정 정상 결론이나 정해진 예문을 반복하지 않는다. "
    "발톱 색, 조명, 붉은색만으로 염증을 단정하지 않는다. 다른 지표의 색칠된 지도와 눌림 "
    "표시는 피부 자극 근거가 아니며 무좀·무지외반·발 형태의 해당 항목에만 사용한다. "
    "disabled_forced_clear 발바닥의 병변 모델 결과는 없지만 이 요청의 피부 자극 항목에는 "
    "표시 없는 사진의 외관을 설명할 수 있다. 사진의 외관을 질환의 확진이나 배제로 확대하지 않는다. "
    "총평과 관리 팁도 같은 실제 관찰을 바탕으로 직접 작성한다."
)


def _inflammation_reporting_disabled(*sources: Any) -> bool:
    """Respect request provenance, including results made before a setting change."""
    if not settings.tinea_inflammation_enabled:
        return True
    rows: list[dict] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            metrics = getattr(value, "metrics", None)
            if isinstance(metrics, dict):
                rows.append(metrics)
            return
        if any(key in value for key in (
            "inflammation_analysis_enabled", "inflammation_analysis_mode",
            "skin_reaction_safety_score", "analysis_mode",
        )):
            rows.append(value)
        for key in ("tinea_pedis", "surfaces", "dorsal", "plantar", "left", "right",
                    "left_foot", "right_foot"):
            if key in value:
                collect(value[key])

    for source in sources:
        collect(source)
    return any(row.get("inflammation_analysis_mode") in {
        "disabled_for_testing", "healthy_test_input", _SKIN_IMAGE_REPORT_MODE,
    } for row in rows)


def inflammation_report_uses_unannotated_images(*sources: Any) -> bool:
    """Share detector-off provenance between image selection and report prompts."""
    return _inflammation_reporting_disabled(*sources)


def _unavailable_skin_measurements(payload: dict[str, Any]) -> dict[str, Any]:
    """Omit disabled measurements from GPT evidence instead of inventing normality."""
    copied = dict(payload)
    copied.update({
        "skin_reaction_safety_score": None,
        "skin_reaction_regions": None,
        "inflammation_regions": None,
        "inflammation_pixel_ratio": None,
        "max_inflammation_prob": None,
        "inflammation_analysis_enabled": False,
        "inflammation_analysis_mode": _SKIN_IMAGE_REPORT_MODE,
        "skin_reaction_score_label": "unavailable_detector_disabled",
    })
    copied.pop("test_input", None)
    copied.pop("overall_health_label", None)
    return copied


def _image_based_skin_facts(facts: dict[str, Any]) -> dict[str, Any]:
    copied = _json_safe_facts(facts)
    tinea = copied.get("tinea_pedis")
    if not isinstance(tinea, dict):
        tinea = {}
        copied["tinea_pedis"] = tinea
    for surface in ("dorsal", "plantar"):
        sides = tinea.get(surface)
        if isinstance(sides, dict):
            for side, row in sides.items():
                if isinstance(row, dict):
                    sides[side] = _unavailable_skin_measurements(row)
    final_scores = tinea.get("final_scores")
    if not isinstance(final_scores, dict):
        final_scores = {}
        tinea["final_scores"] = final_scores
    final_scores["skin_reaction_safety_score"] = None
    tinea["inflammation_analysis_mode"] = _SKIN_IMAGE_REPORT_MODE
    tinea.pop("test_input", None)
    return copied


def _skin_photo_reporting_contract(image_labels: Sequence[str]) -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": _SKIN_IMAGE_REPORT_MODE,
        "assessment_source": "unannotated_actual_photos",
        "skin_measurements_available": False,
        "empty_overlay_is_healthy_evidence": False,
        "image_labels": list(image_labels),
    }


def _side_payload(result: Any, *, omit_skin_measurements: bool = False) -> dict:
    metrics = _metrics(result)
    payload = {
        "analysis_mode": metrics.get("analysis_mode", "model_inference"),
        "model_inference_executed": metrics.get(
            "model_inference_executed",
            True,
        ),
        "fungal_safety_score": getattr(result, "fungal_safety_score", None),
        "skin_reaction_safety_score": getattr(result, "skin_reaction_safety_score", None),
        "fungal_score_label": metrics.get("fungal_score_label"),
        "fungal_analysis_enabled": metrics.get("fungal_analysis_enabled", True),
        "fungal_assessment_source": metrics.get("fungal_model", "unknown"),
        "sam_inflammation_class_used": metrics.get("sam_inflammation_class_used", False),
        "skin_reaction_score_label": metrics.get("inflammation_score_label"),
        "inflammation_analysis_enabled": metrics.get("inflammation_analysis_enabled", True),
        "inflammation_analysis_mode": metrics.get("inflammation_analysis_mode", "enabled"),
        "skin_assessment_source": metrics.get("wound_model", "legacy_skin_model"),
        "wound_detected": metrics.get("wound_detected"),
        "wound_area_ratio": metrics.get("wound_area_ratio"),
        "wound_threshold": metrics.get("wound_threshold"),
        "lesion_overlap_policy": metrics.get("lesion_overlap_policy"),
        "contained_wound_pixels_reclassified_as_fungal": metrics.get(
            "contained_wound_pixels_reclassified_as_fungal"
        ),
        "overall_health_label": metrics.get("overall_health_label"),
        "fungal_pixel_ratio": metrics.get("fungal_pixel_ratio"),
        "inflammation_pixel_ratio": metrics.get("inflammation_pixel_ratio"),
        "max_fungal_prob": metrics.get("max_fungal_prob"),
        "max_inflammation_prob": metrics.get("max_inflammation_prob"),
        "fungal_regions": metrics.get("fungal_regions", []),
        "inflammation_regions": metrics.get("inflammation_regions", []),
        "score_reliability": metrics.get("score_reliability"),
        "foot_outline_found": metrics.get("foot_outline_found"),
    }
    if omit_skin_measurements or _inflammation_reporting_disabled(result):
        return _unavailable_skin_measurements(payload)
    return payload


def _region_locations(
    result: Any,
    key: str,
    *,
    surface: str | None = None,
) -> list[str]:
    locations = []
    for region in _metrics(result).get(key, [])[:3]:
        if not isinstance(region, dict):
            continue
        location = str(region.get("location") or "").strip()
        # Older tinea-analysis results used one shared location vocabulary and
        # can therefore contain a dorsal label even when the image is plantar.
        # Normalize only the report text; the model metrics remain untouched.
        if surface == "plantar":
            location = location.replace("발등", "발바닥")
        if location and location not in locations:
            locations.append(location)
    return locations


def _location_text(left_result: Any, right_result: Any, key: str) -> str:
    parts = []
    for side_label, result in (("왼발", left_result), ("오른발", right_result)):
        locations = _region_locations(result, key)
        if locations:
            parts.append(f"{side_label} {', '.join(locations)}")
    return "; ".join(parts)


def _surface_location_text(
    dorsal_left_result: Any,
    dorsal_right_result: Any,
    sole_left_result: Any,
    sole_right_result: Any,
    key: str,
) -> str:
    parts = []
    surfaces = (
        ("발등", "dorsal", dorsal_left_result, dorsal_right_result),
        ("발바닥", "plantar", sole_left_result, sole_right_result),
    )
    for surface_label, surface_key, left_result, right_result in surfaces:
        for side_label, result in (("왼발", left_result), ("오른발", right_result)):
            locations = _region_locations(result, key, surface=surface_key)
            if locations:
                parts.append(
                    f"{side_label} {surface_label} {', '.join(locations)}"
                )
    return "; ".join(parts)


def _complete_sole_pair(
    sole_left_result: Any | None,
    sole_right_result: Any | None,
) -> bool:
    if (sole_left_result is None) != (sole_right_result is None):
        raise ValueError(
            "sole_left_result and sole_right_result must be provided together."
        )
    return sole_left_result is not None


def _aggregate_score_from_results(results: tuple[Any, ...], attr_name: str) -> int:
    values = []
    for result in results:
        value = getattr(result, attr_name, None)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return aggregate_tinea_safety_scores(values or (100.0,))


def _aggregate_score(left_result: Any, right_result: Any, attr_name: str) -> int:
    return _aggregate_score_from_results((left_result, right_result), attr_name)


def _score_intensity(score: int) -> str:
    if score >= 90:
        return "낮은 수준으로"
    if score >= 70:
        return "약하게"
    if score >= 40:
        return "중간 수준으로"
    return "뚜렷하게"


def build_fallback_tinea_report_text(
    left_result: Any,
    right_result: Any,
    *,
    sole_left_result: Any | None = None,
    sole_right_result: Any | None = None,
) -> TineaReportText:
    # Legacy sole arguments are accepted but excluded from every score and report.
    sole_left_result = sole_right_result = None
    inflammation_disabled = _inflammation_reporting_disabled(
        left_result, right_result, sole_left_result, sole_right_result,
    )
    fungal_score = _aggregate_score(left_result, right_result, "fungal_safety_score")
    skin_score = _aggregate_score(left_result, right_result, "skin_reaction_safety_score")
    fungal_locations = _location_text(left_result, right_result, "fungal_regions")
    skin_locations = _location_text(left_result, right_result, "inflammation_regions")
    fungal_signal = bool(fungal_locations) or safety_score_indicates_tinea_signal(
        fungal_score
    )
    skin_signal = bool(skin_locations) or safety_score_indicates_tinea_signal(
        skin_score
    )

    if fungal_signal:
        location = (
            fungal_locations
            or ("발 일부 영역")
        )
        fungal_text = (
            f"{location}에서 무좀 의심 신호가 {_score_intensity(fungal_score)} 관찰됩니다. "
            "각질, 변색 또는 표면 손상 변화가 이어지는지 관찰과 관리가 필요합니다."
        )
    else:
        fungal_text = (
            "분석 이미지 기준으로 양발에서 뚜렷한 무좀 의심 영역은 크게 관찰되지 않습니다."
        )

    if inflammation_disabled:
        skin_text = _SKIN_IMAGE_REPORT_UNAVAILABLE
        skin_signal = False
    elif skin_signal:
        location = (
            skin_locations
            or ("발 일부 피부")
        )
        skin_text = (
            f"{location}에서 피부 발적 또는 자극 반응이 {_score_intensity(skin_score)} 확인됩니다. "
            "심한 부기나 짓무름은 함께 관찰해 주세요."
        )
    else:
        skin_text = (
            "피부 발적과 자극 반응은 뚜렷하지 않아 전반적으로 안정적인 편입니다."
        )

    if inflammation_disabled:
        total_text = (
            ("무좀 의심 신호가 관찰되어 발을 청결하고 건조하게 관리해 주세요. "
             if fungal_signal else "이번 무좀 분석에서는 뚜렷한 의심 신호가 관찰되지 않습니다. ")
            + _SKIN_IMAGE_REPORT_UNAVAILABLE
        )
    elif fungal_signal and skin_signal:
        total_text = (
            "무좀 의심 신호와 피부 자극 신호가 함께 보여 청결, 건조, 마찰 관리가 필요합니다."
        )
    elif fungal_signal:
        total_text = (
            "무좀 의심 신호가 관찰되어 해당 부위의 변색, 각질, 손상 변화를 꾸준히 확인해 주세요."
        )
    elif skin_signal:
        total_text = (
            "무좀 의심은 크지 않지만 피부 자극 가능성이 있어 건조와 마찰 관리를 권장합니다."
        )
    else:
        total_text = (
            "전반적으로 큰 위험 신호는 두드러지지 않지만 발을 건조하고 청결하게 유지해 주세요."
        )

    return TineaReportText(
        fungal_suspicion_safety_description=_compact_text(fungal_text, fungal_text),
        skin_reaction_safety_description=_compact_text(skin_text, skin_text),
        total_score_description=_compact_text(total_text, total_text, max_chars=160),
        generation_source="DETERMINISTIC_FALLBACK",
        inflammation_analysis_enabled=not inflammation_disabled,
    )



async def generate_tinea_report_text(
    left_result: Any,
    right_result: Any,
    *,
    suspicion_map_png: bytes | None = None,
    photo_overlay_png: bytes | None = None,
    sole_left_result: Any | None = None,
    sole_right_result: Any | None = None,
    sole_suspicion_map_png: bytes | None = None,
    sole_photo_overlay_png: bytes | None = None,
) -> TineaReportText:
    # Legacy sole arguments are accepted but excluded from every score and report.
    sole_left_result = sole_right_result = None
    sole_suspicion_map_png = sole_photo_overlay_png = None
    fallback = build_fallback_tinea_report_text(
        left_result,
        right_result,
        sole_left_result=sole_left_result,
        sole_right_result=sole_right_result,
    )
    inflammation_disabled = not fallback.inflammation_analysis_enabled
    if not _openai_enabled():
        return fallback

    schema = {
        "type": "object",
        "properties": {
            "fungal_suspicion_safety_description": {"type": "string"},
            "skin_reaction_safety_description": {"type": "string"},
            "total_score_description": {"type": "string"},
        },
        "required": [
            "fungal_suspicion_safety_description",
            "skin_reaction_safety_description",
            "total_score_description",
        ],
        "additionalProperties": False,
    }
    common_guidelines = [
        "한국어로 작성한다.",
        "진단처럼 단정하지 말고 '의심', '가능성', '관찰됩니다' 표현을 사용한다.",
        "무좀 의심과 피부 발적/자극/염증 반응을 구분한다.",
        "skin_assessment_source가 wound_segformer_b0 또는 custom_swin_tiny_unet이면 상처 분할 모델의 결과다. 검출 부위와 상처 의심을 설명하고, 염증이나 감염을 확진하지 않는다. 안전 점수는 높을수록 검출 부담이 적은 모델 지표다.",
        _FUNGAL_WOUND_REPORT_INSTRUCTIONS,
        "위치가 불명확하면 특정 발가락을 억지로 쓰지 말고 '발가락 주변', '발 중앙부'처럼 완화한다.",
        "각 필드는 1~2문장으로 짧게 작성한다.",
    ]
    fallback_payload = {
        "fungal_suspicion_safety_description": fallback.fungal_suspicion_safety_description,
        "skin_reaction_safety_description": fallback.skin_reaction_safety_description,
        "total_score_description": fallback.total_score_description,
    }
    color_legend = {
        "photo_overlay": "파란색 계열은 무좀 의심, 빨간색 계열은 피부 자극/염증 의심",
        "suspicion_map": "하늘색 계열은 무좀 의심, 분홍색 계열은 피부 자극/염증 의심",
    }
    payload = {
        "task": "양발 발등 무좀/피부반응 리포트 문구 생성",
        "analysis_scope": "dorsal_only",
        "guidelines": common_guidelines,
        "color_legend": color_legend,
        "left_foot": _side_payload(left_result, omit_skin_measurements=inflammation_disabled),
        "right_foot": _side_payload(right_result, omit_skin_measurements=inflammation_disabled),
        "fallback_text": fallback_payload,
    }
    images = [
        ("왼쪽이 왼발, 오른쪽이 오른발인 의심 영역 지도", suspicion_map_png or b""),
        ("왼쪽이 왼발, 오른쪽이 오른발인 원본 오버레이", photo_overlay_png or b""),
    ]
    system_prompt = (
        "너는 양발 발등 이미지 AI 분석 결과를 사용자에게 설명하는 한국어 리포트 작성자다. 발바닥 피부는 판정하지 않는다. "
        "제공된 점수, 위치 요약, 분석 이미지만 근거로 짧고 구체적인 문구를 작성한다."
    )

    if inflammation_disabled:
        if not settings.openai_report_include_images:
            return fallback
        clean_sources = [
            ("DORSAL LEFT CLEAN UNANNOTATED PHOTO", left_result),
            ("DORSAL RIGHT CLEAN UNANNOTATED PHOTO", right_result),
        ]
        images = []
        for label, source in clean_sources:
            clean = getattr(source, "clean_original_png", None)
            if not isinstance(clean, bytes) or not clean:
                return fallback
            try:
                _image_mime_type(clean)
            except ValueError:
                return fallback
            images.append((label, clean))
        system_prompt += " " + _SKIN_IMAGE_REPORT_INSTRUCTIONS
        payload["inflammation_analysis_contract"] = _skin_photo_reporting_contract(
            [label for label, _ in images],
        )
        payload.pop("fallback_text", None)
        payload.pop("color_legend", None)
        if "final_scores" in payload:
            payload["final_scores"]["skin_reaction_safety_score"] = None

    try:
        request_budget = _ReportRequestBudget.for_report() if inflammation_disabled else None
        maximum_attempts = request_budget.maximum_attempts if request_budget else 1
        for validation_attempt in range(1, maximum_attempts + 1):
            data = await _create_structured_response(
                system_prompt=system_prompt,
                user_payload=payload,
                schema_name="tinea_report_text",
                json_schema=schema,
                images=images,
                image_detail="high",
                max_output_tokens=700,
                request_budget=request_budget,
            )
            if not inflammation_disabled:
                if any(re.search(r"발바닥|족저|plantar|\bsole\b", str(value), re.IGNORECASE) for value in data.values()):
                    raise ReportGenerationError("DORSAL_ONLY_SKIN_ANALYSIS")
                break
            try:
                if not isinstance(data, dict):
                    raise FullPhotoOutputValidationError(field="response", rule_code="OBJECT_REQUIRED")
                descriptions = {
                    field: _require_full_report_text(data.get(field), field=field)
                    for field in schema["required"]
                }
                return TineaReportText(**descriptions, inflammation_analysis_enabled=False)
            except FullPhotoOutputValidationError as exc:
                if (validation_attempt >= maximum_attempts
                        or request_budget.attempts >= maximum_attempts):
                    raise
                payload = {
                    **payload,
                    "validation_retry": {
                        "field": exc.field,
                        "rule_code": exc.rule_code,
                        "instruction": "사진을 다시 확인하고 누락 없이 유효한 한국어 설명을 작성한다.",
                    },
                }
    except Exception as exc:
        logger.warning(
            "GPT tinea report text generation failed; using fallback. error_type=%s",
            type(exc).__name__,
        )
        return fallback

    result = TineaReportText(
        fungal_suspicion_safety_description=_compact_text(
            data.get("fungal_suspicion_safety_description"),
            fallback.fungal_suspicion_safety_description,
        ),
        skin_reaction_safety_description=_compact_text(
            data.get("skin_reaction_safety_description"),
            fallback.skin_reaction_safety_description,
        ),
        total_score_description=_compact_text(
            data.get("total_score_description"),
            fallback.total_score_description,
            max_chars=180,
        ),
    )
    return replace(result, inflammation_analysis_enabled=not inflammation_disabled)



def _normalize_plantar_segmentation_facts(facts: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(facts, dict):
        facts = {}

    region_count: int | None = None
    raw_count = facts.get("region_count")
    if not isinstance(raw_count, bool):
        try:
            parsed_count = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if parsed_count >= 0:
                region_count = min(parsed_count, 10_000)

    contact_area_ratio: float | None = None
    raw_ratio = facts.get("contact_area_ratio")
    if not isinstance(raw_ratio, bool):
        try:
            parsed_ratio = float(raw_ratio)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isfinite(parsed_ratio):
                # Accept either a 0..1 ratio or a percentage supplied as 0..100.
                if 1.0 < parsed_ratio <= 100.0:
                    parsed_ratio /= 100.0
                if 0.0 <= parsed_ratio <= 1.0:
                    contact_area_ratio = round(parsed_ratio, 5)

    status = " ".join(str(facts.get("status") or "unknown").split())[:80]
    return {
        "region_count": region_count,
        "contact_area_ratio": contact_area_ratio,
        "status": status or "unknown",
    }


def _plantar_facts_text(
    left_facts: dict[str, Any],
    right_facts: dict[str, Any],
) -> str:
    def line(side: str, facts: dict[str, Any]) -> str:
        ratio = facts["contact_area_ratio"]
        ratio_text = "unknown" if ratio is None else f"{ratio:.5f}"
        count = facts["region_count"]
        count_text = "unknown" if count is None else str(count)
        return (
            f"{side}: region_count={count_text}; "
            f"contact_area_ratio={ratio_text}; status={facts['status']}"
        )

    return f"{line('LEFT', left_facts)}\n{line('RIGHT', right_facts)}"


_PLANTAR_ABSTENTION_TEXT = (
    "발바닥 형태는 요족·정상·평발 중 어느 경향인지 판단하기 어렵습니다."
)

_PLANTAR_COMPOSITE_IMAGE_CONTRACT = (
    "발바닥 형태는 실제 세그먼테이션 전 원본 사진을 나란히 배치한 합본 한 장만 "
    "근거로 분석한다. 합본의 LEFT 패널은 왼발, RIGHT 패널은 오른발이다. "
    "사진은 크기·비율·촬영 방향을 유지하며 자르거나 좌우 반전하지 않았다. "
    "흰 여백과 사진 바깥의 라벨은 발 형태에 포함하지 않는다. "
    "발등 사진, 압력 히트맵, 세그먼테이션 마스크 또는 분석 오버레이로 "
    "요족·정상·평발 형태를 추론하지 않는다."
)

# Both photo-report paths use the same evidence policy. Similar segmentation
# ratios alone neither establish bilateral similarity nor classify arch shape.
_PLANTAR_SHAPE_ASSESSMENT_POLICY = {
    "bilateral_consistency": (
        "먼저 좌우 원본의 아치와 중족부 형태를 함께 비교한다. 좌우 형태가 유사하거나 "
        "차이가 경미하고 양쪽에 같은 범주를 지지하는 근거가 있으면 같은 범주로 설명한다. "
        "이때 '양발은 정상 형태입니다.' 또는 '양발은 평발 경향이 보입니다.'처럼 쓴다."
    ),
    "separate_categories": (
        "서로 다른 범주는 각 발에서 명확히 확인되는 형태 차이가 있을 때만 선택한다. "
        "촬영 각도, 좌우 반전, 밝기, 그림자, 가림, 경미한 윤곽 차이를 형태 차이로 "
        "과장하지 않고, 한쪽 근거가 부족하면 다른 쪽 결과를 복사하지 않는다."
    ),
    "high_arch_evidence": (
        "요족 경향은 원본에서 매우 좁은 중족부 형태와 깊은 안쪽 오목함 등 높은 아치를 지지하는 "
        "뚜렷한 형태 근거가 있을 때만 선택한다. 약간의 오목함이나 애매한 그림자만으로 "
        "요족을 선택하지 않고, 사진만으로 요족의 임상적 중증도를 단정하지 않는다."
    ),
    "low_arch_evidence": (
        "평발 경향은 원본에서 안쪽 아치의 오목함이 거의 사라지고 중족부가 넓고 평평하게 "
        "이어지는 등 낮은 아치를 지지하는 뚜렷한 형태 근거가 있을 때만 선택한다. "
        "경미한 오목함 감소나 약간 넓은 중족부만으로 평발 경향을 선택하지 않는다. "
        "요족이 아니라는 이유만으로 평발을 선택하지 않는다."
    ),
    "normal_or_insufficient_evidence": (
        "원본에서 아치와 중족부 형태를 충분히 확인할 수 있고 뚜렷한 형태 이상이 없으면 "
        "정상 형태로 설명한다. 경미한 아치 높낮이 차이, 약한 오목함 또는 경계 사례만으로 "
        "평발·요족 경향을 선택하지 않고 정상 형태를 유지한다. 원본에서 아치 형태를 충분히 "
        "확인할 수 없거나 촬영 품질·가림으로 판단 근거가 부족하면 지정된 판단 보류 문장을 쓴다. "
        "영역 수나 전체 면적 비율만으로 같은 범주를 강제하거나 아치를 분류하지 않는다."
    ),
}
_PLANTAR_SHAPE_ASSESSMENT_INSTRUCTIONS = " ".join(
    _PLANTAR_SHAPE_ASSESSMENT_POLICY.values()
)


def build_fallback_plantar_footprint_analysis_text(
    *,
    left_segmentation_facts: dict[str, Any],
    right_segmentation_facts: dict[str, Any],
) -> str:
    # These segmentation facts are not calibrated pressure-sensor values and
    # do not provide a deterministic arch-class threshold. Abstain rather than
    # inventing load balance or a confident foot-shape label.
    _normalize_plantar_segmentation_facts(left_segmentation_facts)
    _normalize_plantar_segmentation_facts(right_segmentation_facts)
    return _PLANTAR_ABSTENTION_TEXT


_PLANTAR_MARKUP_PATTERN = re.compile(
    r"https?://|www\.|<[^>]*>|`|\*\*|^\s*#{1,6}\s|^\s*[-+]\s",
    re.IGNORECASE | re.MULTILINE,
)
_PLANTAR_CERTAINTY_PATTERN = re.compile(
    r"진단됩니다|확진|판정됩니다|"
    r"(?:평발|요족|과내전|과외전)(?:\s*경향)?\s*(?:입니다|으로\s*진단|으로\s*판정)",
)
_PLANTAR_SHAPE_TERM_PATTERN = re.compile(r"요족|정상|평발")
_PLANTAR_HEDGE_LANGUAGE_PATTERN = re.compile(r"수\s*있|가능성")
_PLANTAR_EVIDENCE_STATEMENT_PATTERN = re.compile(
    r"정상\s*형태(?:이며|이고|입니다)|"
    r"(?:요족|평발)\s*경향(?:이며|이고|이\s*(?:보입니다|있습니다))"
)
_PLANTAR_PRESSURE_LANGUAGE_PATTERN = re.compile(
    r"압력|하중|체중|착지|균형|"
    r"접촉\s*(?:분포|면적|위치|영역|연속성)|영역\s*수"
)
_PLANTAR_FORBIDDEN_SHAPE_LANGUAGE_PATTERN = re.compile(r"과내전|과외전|진단")


def _plantar_shape_wording_rule_code(text: str) -> str | None:
    """Return the exact plantar wording violation, if any."""

    if _PLANTAR_HEDGE_LANGUAGE_PATTERN.search(text):
        return "PLANTAR_HEDGE_LANGUAGE_NOT_ALLOWED"
    if text == _PLANTAR_ABSTENTION_TEXT:
        return None
    if re.search(r"판단하기\s*어렵|구분하기\s*어렵", text):
        return "PLANTAR_ABSTENTION_FORMAT_INVALID"
    if not _PLANTAR_SHAPE_TERM_PATTERN.search(text):
        return "PLANTAR_ARCH_CATEGORY_REQUIRED"
    if not _PLANTAR_EVIDENCE_STATEMENT_PATTERN.search(text):
        return "PLANTAR_EVIDENCE_STATEMENT_REQUIRED"
    return None


def _validated_plantar_footprint_analysis_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if (
        not text
        or len(text) > 180
        or _PLANTAR_MARKUP_PATTERN.search(text)
        or re.search(r"\d", text)
    ):
        return None
    if (
        _PLANTAR_CERTAINTY_PATTERN.search(text)
        or _PLANTAR_PRESSURE_LANGUAGE_PATTERN.search(text)
        or _PLANTAR_FORBIDDEN_SHAPE_LANGUAGE_PATTERN.search(text)
    ):
        return None

    sentences = _full_report_sentence_list(text)
    if sentences is None or len(sentences) != 1:
        return None
    if any(len(sentence) > 180 or not re.search(r"[가-힣]", sentence) for sentence in sentences):
        return None
    result = sentences[0]
    if _plantar_shape_wording_rule_code(result) is not None:
        return None
    return result if len(result) <= 180 else None


async def generate_plantar_footprint_analysis_text(
    *,
    left_raw_image_bytes: bytes,
    right_raw_image_bytes: bytes,
    left_segmentation_facts: dict[str, Any],
    right_segmentation_facts: dict[str, Any],
) -> str:
    """Generate one observed or explicitly abstained foot-shape sentence.

    The raw uploads are composed into one labeled, pre-segmentation image.
    Local segmentation output is represented by three allow-listed scalar facts.
    """
    normalized_left = _normalize_plantar_segmentation_facts(left_segmentation_facts)
    normalized_right = _normalize_plantar_segmentation_facts(right_segmentation_facts)
    fallback = build_fallback_plantar_footprint_analysis_text(
        left_segmentation_facts=normalized_left,
        right_segmentation_facts=normalized_right,
    )
    if not _openai_enabled() or not settings.openai_report_include_images:
        return fallback

    schema = {
        "type": "object",
        "properties": {
            "plantar_footprint_analysis_text": {
                "type": "string",
                "description": _PLANTAR_SHAPE_ASSESSMENT_INSTRUCTIONS,
            }
        },
        "required": ["plantar_footprint_analysis_text"],
        "additionalProperties": False,
    }
    payload = {
        "task": "좌우 발바닥의 요족·정상·평발 형태 경향 문장 생성",
        "image_contract": _PLANTAR_COMPOSITE_IMAGE_CONTRACT,
        "deterministic_segmentation_facts_text": _plantar_facts_text(
            normalized_left,
            normalized_right,
        ),
        "plantar_shape_assessment_policy": dict(_PLANTAR_SHAPE_ASSESSMENT_POLICY),
        "output_contract": (
            "근거가 있으면 정상 형태 또는 요족·평발 경향을 관찰형으로 쓰고, "
            "근거가 부족하면 지정된 판단 보류 문장을 쓰는 한국어 한 문장을 작성한다."
        ),
    }
    system_prompt = (
        "당신은 발바닥 이미지에서 보이는 형태를 요족·정상·평발 세 범주 안에서만 "
        "설명하는 한국어 문장을 작성한다. "
        + _PLANTAR_COMPOSITE_IMAGE_CONTRACT + " "
        + _PLANTAR_SHAPE_ASSESSMENT_INSTRUCTIONS + " "
        "명확한 좌우 형태 차이가 있을 때만 한 문장 안에서 각각 설명한다. 정상은 "
        "'정상 형태', 요족과 평발은 '경향이 보입니다' 또는 '경향이 있습니다'로 쓴다. "
        "'수 있습니다'와 '가능성'은 사용하지 않는다. 근거가 부족하면 반드시 "
        f"'{_PLANTAR_ABSTENTION_TEXT}'라고만 쓴다. "
        "서버의 영역 수와 면적 비율은 압력 센서값이 아니므로 압력이나 하중으로 해석하지 않는다. "
        "압력, 하중, 체중 분포, 착지, 접촉 면적·영역 수·분포, 균형, 과내전, 과외전은 "
        "평가하거나 언급하지 않는다. "
        "'의학적 진단이 아닙니다' 같은 안내 문장도 작성하지 않는다. 결과는 숫자 없이 "
        "백팔십 자 이내의 한국어 한 문장만 사용하며 마크다운, HTML, URL은 쓰지 않는다."
    )

    try:
        # Validate MIME signatures before building or sending the request. This
        # also prevents JPEG uploads from being mislabeled as PNG data URLs.
        _image_mime_type(left_raw_image_bytes)
        _image_mime_type(right_raw_image_bytes)
        combined_photo = combine_raw_plantar_photos(left_raw_image_bytes, right_raw_image_bytes)
        data = await _create_structured_response(
            system_prompt=system_prompt,
            user_payload=payload,
            schema_name="plantar_footprint_analysis_text",
            json_schema=schema,
            images=[(PLANTAR_COMPOSITE_LABEL, combined_photo)],
            image_detail="high",
            max_output_tokens=260,
        )
        generated = _validated_plantar_footprint_analysis_text(
            data.get("plantar_footprint_analysis_text")
        )
        if generated is None:
            raise ValueError("OpenAI plantar analysis text did not satisfy the output contract.")
        return generated
    except Exception as exc:
        logger.warning(
            "GPT plantar footprint analysis text generation failed; using fallback. error=%s",
            exc,
        )
        return fallback


_FULL_REPORT_ADVICE_KEYS = (
    "HALLUX_VALGUS",
    "ATHLETES_FOOT",
    "SKIN_IRRITATION",
)
_FULL_PHOTO_VALIDATION_FIELD_PATTERN = re.compile(
    r"^(?:\$response|hallux_score_analysis_text|"
    r"fungal_suspicion_safety_description|skin_reaction_safety_description|"
    r"total_score_description|plantar_footprint_analysis_text|metric_advice(?:\."
    r"(?:HALLUX_VALGUS|ATHLETES_FOOT|SKIN_IRRITATION)(?:\."
    r"(?:observation|practical_tip))?)?)$"
)
_FULL_PHOTO_VALIDATION_RULE_CODES = frozenset(
    {
        "STRING_REQUIRED",
        "NON_EMPTY_TEXT_REQUIRED",
        "MARKUP_NOT_ALLOWED",
        "DIGITS_NOT_ALLOWED",
        "DIAGNOSTIC_OR_CERTAINTY_NOT_ALLOWED",
        "KOREAN_TEXT_REQUIRED",
        "SENTENCE_COUNT_OUT_OF_RANGE",
        "SENTENCE_TOO_LONG",
        "TEXT_TOO_LONG",
        "PLANTAR_OBSERVATION_REQUIRED",
        "PLANTAR_ARCH_CATEGORY_REQUIRED",
        "PLANTAR_ARCH_CATEGORY_NOT_ALLOWED",
        "PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
        "PLANTAR_DISCLAIMER_NOT_ALLOWED",
        "PLANTAR_HEDGE_LANGUAGE_NOT_ALLOWED",
        "PLANTAR_EVIDENCE_STATEMENT_REQUIRED",
        "PLANTAR_ABSTENTION_FORMAT_INVALID",
        "EXACT_METRIC_KEYS_REQUIRED",
        "EXACT_ADVICE_FIELDS_REQUIRED",
        "JSON_OBJECT_REQUIRED",
        "DORSAL_ONLY_SKIN_ANALYSIS",
    }
)
_FULL_PHOTO_OUTPUT_CONTRACT_INSTRUCTIONS = (
    "출력 계약은 권고가 아니라 저장 전 반드시 통과해야 하는 필수 규칙이다. 응답을 "
    "제출하기 전에 모든 필드를 스스로 다시 검사한다. JSON에는 schema가 요구한 최상위 "
    "필드만 넣고 누락하거나 추가하지 않는다. 일반 설명 네 필드는 각각 한국어 한 문장 "
    "이상 세 문장 이하로 작성한다. plantar_footprint_analysis_text는 근거가 있으면 정상 "
    "형태 또는 요족·평발 경향을 관찰형으로 쓰고, 근거가 부족하면 지정된 판단 보류 문장을 "
    "쓰는 한국어 한 문장으로 작성한다. 이 필드에는 '수 있습니다'와 '가능성', 압력 평가, "
    "비진단 안내 문장을 넣지 않는다. metric_advice에는 "
    "HALLUX_VALGUS, ATHLETES_FOOT, SKIN_IRRITATION 세 key만 넣으며, 각 key에는 "
    "observation과 practical_tip만 넣는다. 각 observation과 practical_tip은 줄바꿈 없이 "
    "정확히 한 문장이어야 한다. 한 문장 안에서 마침표, 물음표, 느낌표 같은 종결 부호는 "
    "맨 끝에 한 번만 사용하고 중간에 두 번째 문장을 시작하지 않는다. 두 내용을 담아야 하면 "
    "문장을 나누지 말고 쉼표와 연결 어미를 사용해 한 문장으로 합친다. 응답 직전에 세 지표의 "
    "observation과 practical_tip 여섯 값이 각각 한 문장인지 다시 센다. 모든 문구에서 숫자, "
    "마크다운, HTML, URL, 진단·확진·판정 표현을 사용하지 않는다."
)
_FULL_REPORT_DIAGNOSTIC_PATTERN = re.compile(
    r"확진|진단됩니다|진단되었습니다|판정됩니다|"
    r"(?:무지외반증?|무좀|백선|피부염|평발|요족|과내전|과외전|정상)"
    r"(?:\s*경향)?\s*(?:입니다|으로\s*진단|으로\s*판정|이\s*확실)",
)


def _full_report_sentence_list(value: Any) -> list[str] | None:
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or _PLANTAR_MARKUP_PATTERN.search(text) or re.search(r"\d", text):
        return None
    if _FULL_REPORT_DIAGNOSTIC_PATTERN.search(text):
        return None

    sentences: list[str] = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        for sentence in re.split(r"(?<=[.!?])(?:\s+|(?=[가-힣]))", line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence[-1] not in ".!?":
                sentence = f"{sentence}."
            sentences.append(sentence)
    if any(not re.search(r"[가-힣]", sentence) for sentence in sentences):
        return None
    return sentences


def _validated_full_report_text(
    value: Any,
    *,
    minimum_sentences: int = 1,
    maximum_sentences: int = 3,
    maximum_chars: int = 350,
) -> str | None:
    sentences = _full_report_sentence_list(value)
    if sentences is None or not minimum_sentences <= len(sentences) <= maximum_sentences:
        return None
    result = "\n".join(sentences)
    return result if len(result) <= maximum_chars else None


def _require_full_report_text(
    value: Any,
    *,
    field: str,
    minimum_sentences: int = 1,
    maximum_sentences: int = 3,
    maximum_chars: int = 350,
) -> str:
    """Validate one generated text field and report a safe failure reason."""

    if not isinstance(value, str):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="STRING_REQUIRED",
        )
    raw_text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw_text:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="NON_EMPTY_TEXT_REQUIRED",
        )
    if _PLANTAR_MARKUP_PATTERN.search(raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="MARKUP_NOT_ALLOWED",
        )
    if re.search(r"\d", raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="DIGITS_NOT_ALLOWED",
        )
    if _FULL_REPORT_DIAGNOSTIC_PATTERN.search(raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="DIAGNOSTIC_OR_CERTAINTY_NOT_ALLOWED",
        )

    sentences = _full_report_sentence_list(raw_text)
    if sentences is None:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="KOREAN_TEXT_REQUIRED",
        )
    if not minimum_sentences <= len(sentences) <= maximum_sentences:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="SENTENCE_COUNT_OUT_OF_RANGE",
        )
    result = "\n".join(sentences)
    if len(result) > maximum_chars:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="TEXT_TOO_LONG",
        )
    skin_field = field in {"fungal_suspicion_safety_description", "skin_reaction_safety_description",
                           "total_score_description"} or field.startswith(
                               ("metric_advice.ATHLETES_FOOT", "metric_advice.SKIN_IRRITATION"))
    if skin_field and re.search(r"발바닥|족저|plantar|\bsole\b", raw_text, re.IGNORECASE):
        raise FullPhotoOutputValidationError(field=field, rule_code="DORSAL_ONLY_SKIN_ANALYSIS")
    return result


_FULL_PHOTO_PLANTAR_FIELD = "plantar_footprint_analysis_text"
_FULL_PHOTO_PLANTAR_DISCLAIMER_PATTERN = re.compile(
    r"(?:의학적|의료적|전문적(?:인)?\s*의학적)?\s*진단(?:"
    r"(?:이|은)?\s*(?:아닙니다|아닌|아니며|아니므로)|"
    r"(?:을|를)?\s*(?:대신|대체)(?:하지\s*않습니다|할\s*수\s*없습니다)|"
    r"(?:으로)?\s*볼\s*수\s*없습니다|"
    r"\s*목적이\s*아닙니다|"
    r"을\s*위한\s*것이\s*아닙니다|"
    r"에\s*해당하지\s*않습니다"
    r")"
)


def _require_full_photo_plantar_footprint_analysis_text(value: Any) -> str:
    """Require one observed or explicitly abstained foot-shape sentence."""

    field = _FULL_PHOTO_PLANTAR_FIELD
    if not isinstance(value, str):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="STRING_REQUIRED",
        )
    raw_text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw_text:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="NON_EMPTY_TEXT_REQUIRED",
        )
    if _PLANTAR_MARKUP_PATTERN.search(raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="MARKUP_NOT_ALLOWED",
        )
    if re.search(r"\d", raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="DIGITS_NOT_ALLOWED",
        )
    if _FULL_PHOTO_PLANTAR_DISCLAIMER_PATTERN.search(raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="PLANTAR_DISCLAIMER_NOT_ALLOWED",
        )
    if _PLANTAR_PRESSURE_LANGUAGE_PATTERN.search(raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="PLANTAR_PRESSURE_EVALUATION_NOT_ALLOWED",
        )
    if re.search(r"과내전|과외전", raw_text):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="PLANTAR_ARCH_CATEGORY_NOT_ALLOWED",
        )
    if (
        _PLANTAR_CERTAINTY_PATTERN.search(raw_text)
        or _FULL_REPORT_DIAGNOSTIC_PATTERN.search(raw_text)
    ):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="DIAGNOSTIC_OR_CERTAINTY_NOT_ALLOWED",
        )

    sentences = _full_report_sentence_list(raw_text)
    if sentences is None:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="KOREAN_TEXT_REQUIRED",
        )
    if len(sentences) != 1:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="SENTENCE_COUNT_OUT_OF_RANGE",
        )
    if any(len(sentence) > 180 for sentence in sentences):
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code="SENTENCE_TOO_LONG",
        )

    wording_rule_code = _plantar_shape_wording_rule_code(sentences[0])
    if wording_rule_code is not None:
        raise FullPhotoOutputValidationError(
            field=field,
            rule_code=wording_rule_code,
        )
    return sentences[0]


def _full_report_plantar_facts(facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    plantar = facts.get("plantar_footprint") if isinstance(facts, dict) else None
    if not isinstance(plantar, dict):
        plantar = {}
    left = plantar.get("left")
    right = plantar.get("right")
    return (
        left if isinstance(left, dict) else {},
        right if isinstance(right, dict) else {},
    )


def build_fallback_full_photo_analysis_texts(
    *,
    facts: dict[str, Any],
) -> FullPhotoAnalysisTexts:
    inflammation_disabled = _inflammation_reporting_disabled(facts)
    left_plantar, right_plantar = _full_report_plantar_facts(facts)
    plantar_text = build_fallback_plantar_footprint_analysis_text(
        left_segmentation_facts=left_plantar,
        right_segmentation_facts=right_plantar,
    )
    hallux = facts.get("hallux_valgus", {})
    if not isinstance(hallux, dict):
        hallux = {}
    left_angle = _finite_report_number(hallux.get("left_angle_degree"))
    right_angle = _finite_report_number(hallux.get("right_angle_degree"))
    if left_angle is None or right_angle is None:
        hallux_observation = "엄지발가락 정렬을 설명할 측정 근거가 충분하지 않습니다."
    elif abs(left_angle - right_angle) < 1.0:
        hallux_observation = "좌우 엄지발가락의 기울기가 비슷하게 측정되었습니다."
    else:
        larger, smaller = ("왼발", "오른발") if left_angle > right_angle else ("오른발", "왼발")
        hallux_observation = f"{larger}에서 엄지발가락의 기울기가 {smaller}보다 크게 측정되었습니다."
    fungal_text, fungal_signal = _fallback_photo_skin_observation(
        facts, "fungal_suspicion_safety_score", "무좀",
    )
    skin_text, skin_signal = _fallback_photo_skin_observation(
        facts, "skin_reaction_safety_score", "피부 자극",
    )
    if inflammation_disabled:
        total_text = fungal_text + " " + _SKIN_IMAGE_REPORT_UNAVAILABLE
    elif fungal_signal is True and skin_signal is True:
        total_text = "무좀 의심 신호와 피부 자극 신호가 함께 관찰되어 발을 건조하게 유지하고 마찰을 줄이는 관리가 필요합니다."
    elif fungal_signal is True:
        total_text = "무좀 의심 신호가 관찰되어 발을 청결하고 건조하게 관리하며 피부 변화가 이어지는지 살펴봐 주세요."
    elif skin_signal is True:
        total_text = "피부 자극 의심 신호가 관찰되어 발에 닿는 마찰을 줄이고 피부 상태의 변화를 살펴봐 주세요."
    elif fungal_signal is False and skin_signal is False:
        total_text = "이번 발등 피부 분석에서는 뚜렷한 의심 신호가 관찰되지 않았으며 발을 청결하고 건조하게 관리해 주세요."
    else:
        total_text = "이번 발등 피부 분석에서는 상태를 설명할 관찰 근거가 충분하지 않습니다."
    return FullPhotoAnalysisTexts(
        hallux_score_analysis_text=hallux_observation,
        fungal_suspicion_safety_description=fungal_text,
        skin_reaction_safety_description=skin_text,
        total_score_description=total_text,
        plantar_footprint_analysis_text=plantar_text,
        metric_advice={
            "HALLUX_VALGUS": (
                hallux_observation,
                "발 앞부분을 압박하지 않는 신발을 선택하고 불편이 지속되면 전문가와 상담해 주세요.",
            ),
            "ATHLETES_FOOT": (
                fungal_text,
                "발을 청결하고 건조하게 관리하고 변화가 지속되면 전문가와 상담해 주세요.",
            ),
            "SKIN_IRRITATION": (
                skin_text,
                (_SKIN_IMAGE_FALLBACK_TIP if inflammation_disabled else
                 "마찰과 자극을 줄이고 보습을 유지하며 불편이 이어지면 전문가와 상담해 주세요."),
            ),
        },
        generation_source="DETERMINISTIC_FALLBACK",
        inflammation_analysis_enabled=not inflammation_disabled,
    )


def _full_photo_report_schema() -> dict[str, Any]:
    report_text_properties = {
        "hallux_score_analysis_text": {
            "type": "string",
            "description": (
                "무지외반 관련 이미지와 서버 확정 facts를 바탕으로 작성한 비진단적 한국어 "
                "관찰 문구. 숫자, 마크다운, HTML, URL 없이 한 문장 이상 세 문장 이하, "
                "삼백오십 자 이내로 작성한다."
            ),
        },
        "fungal_suspicion_safety_description": {
            "type": "string",
            "description": (
                "양발 발등의 무좀 의심 신호를 고려한 비진단적 한국어 관찰 "
                "문구. 숫자, 마크다운, HTML, URL 없이 한 문장 이상 세 문장 이하, "
                "삼백오십 자 이내로 작성한다."
            ),
        },
        "skin_reaction_safety_description": {
            "type": "string",
            "description": (
                "양발 발등의 피부 자극 신호를 고려한 비진단적 한국어 관찰 "
                "문구. 숫자, 마크다운, HTML, URL 없이 한 문장 이상 세 문장 이하, "
                "삼백오십 자 이내로 작성한다."
            ),
        },
        "total_score_description": {
            "type": "string",
            "description": (
                "서버 확정 분석 facts를 종합한 비진단적 한국어 설명. 새 숫자를 출력하지 "
                "않고, 마크다운, HTML, URL 없이 한 문장 이상 세 문장 이하, 삼백오십 자 "
                "이내로 작성한다."
            ),
        },
        "plantar_footprint_analysis_text": {
            "type": "string",
            "description": (
                "왼발과 오른발의 형태를 요족·정상·평발 세 범주 안에서만 설명하는 한국어 한 "
                "문장. 근거가 있으면 정상은 '정상 형태', 요족과 평발은 '경향이 보입니다' 또는 "
                "'경향이 있습니다'로 쓰고, 근거가 부족하면 '발바닥 형태는 요족·정상·평발 중 "
                "어느 경향인지 판단하기 어렵습니다.'라고 쓴다. '수 있습니다'와 '가능성'은 "
                "사용하지 않는다. 백팔십 자 이내이며 압력, "
                "하중, 체중 분포, 착지, 접촉 면적·영역 수·분포, 균형, 과내전, 과외전, 숫자, "
                "마크다운, HTML, URL, 진단·확진·"
                "판정 및 비진단 안내 문장을 사용하지 않는다."
                + " " + _PLANTAR_SHAPE_ASSESSMENT_INSTRUCTIONS
            ),
        },
    }
    advice_property = {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "description": (
                    "해당 지표의 비진단적 한국어 관찰을 숫자나 마크업 없이 정확히 한 문장, "
                    "백팔십 자 이내로 작성한다."
                ),
            },
            "practical_tip": {
                "type": "string",
                "description": (
                    "해당 관찰에 맞는 일상 관리 팁을 숫자나 마크업 없이 정확히 한 문장의 "
                    "한국어, 백팔십 자 이내로 작성한다."
                ),
            },
        },
        "required": ["observation", "practical_tip"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        **report_text_properties,
        "metric_advice": {
            "type": "object",
            "description": (
                "무지외반, 무좀 위험도, 피부 자극도 각각에 대한 한 문장 관찰과 한 문장 관리 팁."
            ),
            "properties": {
                key: advice_property for key in _FULL_REPORT_ADVICE_KEYS
            },
            "required": list(_FULL_REPORT_ADVICE_KEYS),
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "description": "전체 사진 분석에서 백엔드 저장에 사용할 사용자용 한국어 리포트 문구.",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _require_metric_advice(value: Any) -> dict[str, tuple[str, str]]:
    if not isinstance(value, dict) or set(value) != set(_FULL_REPORT_ADVICE_KEYS):
        raise FullPhotoOutputValidationError(
            field="metric_advice",
            rule_code="EXACT_METRIC_KEYS_REQUIRED",
        )

    validated: dict[str, tuple[str, str]] = {}
    for key in _FULL_REPORT_ADVICE_KEYS:
        item = value.get(key)
        if not isinstance(item, dict) or set(item) != {"observation", "practical_tip"}:
            raise FullPhotoOutputValidationError(
                field=f"metric_advice.{key}",
                rule_code="EXACT_ADVICE_FIELDS_REQUIRED",
            )
        observation = _require_full_report_text(
            item.get("observation"),
            field=f"metric_advice.{key}.observation",
            minimum_sentences=1,
            maximum_sentences=1,
            maximum_chars=180,
        )
        practical_tip = _require_full_report_text(
            item.get("practical_tip"),
            field=f"metric_advice.{key}.practical_tip",
            minimum_sentences=1,
            maximum_sentences=1,
            maximum_chars=180,
        )
        validated[key] = (observation, practical_tip)
    return validated


def _validate_full_photo_response_data(data: Any) -> FullPhotoAnalysisTexts:
    """Validate one complete structured response without retaining raw model text."""

    if not isinstance(data, dict):
        raise FullPhotoOutputValidationError(
            field="$response",
            rule_code="JSON_OBJECT_REQUIRED",
        )

    hallux_text = _require_full_report_text(
        data.get("hallux_score_analysis_text"),
        field="hallux_score_analysis_text",
        maximum_sentences=3,
        maximum_chars=350,
    )
    fungal_text = _require_full_report_text(
        data.get("fungal_suspicion_safety_description"),
        field="fungal_suspicion_safety_description",
        maximum_sentences=3,
        maximum_chars=350,
    )
    skin_text = _require_full_report_text(
        data.get("skin_reaction_safety_description"),
        field="skin_reaction_safety_description",
        maximum_sentences=3,
        maximum_chars=350,
    )
    total_text = _require_full_report_text(
        data.get("total_score_description"),
        field="total_score_description",
        maximum_sentences=3,
        maximum_chars=350,
    )
    plantar_text = _require_full_photo_plantar_footprint_analysis_text(
        data.get("plantar_footprint_analysis_text")
    )
    advice = _require_metric_advice(data.get("metric_advice"))

    return FullPhotoAnalysisTexts(
        hallux_score_analysis_text=hallux_text,
        fungal_suspicion_safety_description=fungal_text,
        skin_reaction_safety_description=skin_text,
        total_score_description=total_text,
        plantar_footprint_analysis_text=plantar_text,
        metric_advice=advice,
        generation_source="OPENAI",
    )


def _finite_report_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fallback_photo_skin_observation(
    facts: dict[str, Any], score_key: str, label: str,
) -> tuple[str, bool | None]:
    if score_key == "skin_reaction_safety_score" and _inflammation_reporting_disabled(facts):
        return _SKIN_IMAGE_REPORT_UNAVAILABLE, None
    tinea = facts.get("tinea_pedis", {})
    if not isinstance(tinea, dict):
        tinea = {}
    final_scores = tinea.get("final_scores", {})
    score = _finite_report_number(final_scores.get(score_key)) if isinstance(final_scores, dict) else None
    surfaces = [
        side for surface_key in ("dorsal", "plantar")
        if isinstance(tinea.get(surface_key), dict)
        for side in tinea[surface_key].values() if isinstance(side, dict)
    ]
    if surfaces and all(side.get("analysis_mode") == "disabled_forced_clear" for side in surfaces):
        score = None
    if score is None or not 0 <= score <= 100:
        return f"이번 분석에서는 {label} 의심 신호를 설명할 근거가 충분하지 않습니다.", None
    wound_sources = [
        side for side in surfaces
        if side.get("skin_assessment_source") in {"wound_segformer_b0", "custom_swin_tiny_unet"}
    ] if score_key == "skin_reaction_safety_score" else []
    if wound_sources:
        label = "상처"
    signal = any(side.get("wound_detected") is True for side in wound_sources) or safety_score_indicates_tinea_signal(score)
    if not signal:
        return f"이번 분석에서는 뚜렷한 {label} 의심 신호가 관찰되지 않습니다.", False
    return f"분석 결과에서 {label} 의심 신호가 {_score_intensity(score)} 관찰됩니다.", True


def _report_failure_code(error: Exception) -> str:
    if isinstance(error, FullPhotoOutputValidationError):
        return "OUTPUT_VALIDATION_FAILED"
    if isinstance(error, ReportGenerationError) and re.fullmatch(r"[A-Z0-9_]{1,64}", error.code):
        return error.code
    if isinstance(error, (httpx.TimeoutException, asyncio.TimeoutError)):
        return "API_TIMEOUT"
    return "REPORT_GENERATION_FAILED"


def _full_photo_validation_retry_instructions(
    error: FullPhotoOutputValidationError,
    *,
    attempt: int = 2,
    maximum_attempts: int | None = None,
) -> tuple[str, dict[str, str | int]]:
    """Build a safe corrective retry contract without echoing generated text."""

    safe_field = (
        error.field
        if _FULL_PHOTO_VALIDATION_FIELD_PATTERN.fullmatch(error.field)
        else "$response"
    )
    safe_rule_code = (
        error.rule_code
        if error.rule_code in _FULL_PHOTO_VALIDATION_RULE_CODES
        else "OUTPUT_CONTRACT_FAILED"
    )
    retry_metadata: dict[str, str | int] = {
        "attempt": attempt,
        "maximum_attempts": (
            int(settings.openai_report_max_attempts)
            if maximum_attempts is None else maximum_attempts
        ),
        "failed_field": safe_field,
        "failed_rule_code": safe_rule_code,
        "instruction": (
            "이전 응답 본문을 추측하거나 부분 수정하지 말고, 같은 facts와 이미지를 근거로 "
            "전체 JSON을 처음부터 다시 생성한 뒤 모든 출력 규칙을 재검사한다."
        ),
    }
    prompt = (
        "교정 재시도다. 이전 응답은 출력 계약 검사에서 거부되었다. 이전 응답 본문은 "
        "보안상 제공하지 않는다. "
        f"실패 필드는 {safe_field}이고 규칙 코드는 {safe_rule_code}다. "
        "해당 규칙만 고치는 데 그치지 말고 전체 JSON을 처음부터 다시 생성한다. 특히 "
        "metric_advice의 observation과 practical_tip은 각각 종결 부호가 맨 끝에 한 번만 "
        "있는 정확히 한 문장인지 제출 전에 다시 센다."
    )
    return prompt, retry_metadata


def _validated_metric_advice(value: Any) -> dict[str, tuple[str, str]] | None:
    """Backward-compatible nullable wrapper around the detailed validator."""

    try:
        return _require_metric_advice(value)
    except FullPhotoOutputValidationError:
        return None


def _json_safe_facts(facts: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(facts, dict):
        raise TypeError("facts must be a dictionary.")
    # Round-trip without a custom encoder: image bytes and arbitrary objects
    # cannot accidentally leak into the text payload, while numeric values are
    # preserved exactly as supplied by the deterministic analysis pipeline.
    encoded = json.dumps(facts, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("facts must encode to a JSON object.")
    return decoded


def _valid_related_analysis_images(
    related_analysis_images: dict[str, bytes],
    *, unannotated_tinea_only: bool = False,
) -> list[tuple[str, bytes]]:
    if not isinstance(related_analysis_images, dict):
        return []
    valid: list[tuple[str, bytes]] = []
    for raw_label, image_bytes in related_analysis_images.items():
        if not isinstance(image_bytes, bytes) or not image_bytes:
            continue
        label = " ".join(str(raw_label).split())[:120]
        if not label:
            continue
        if unannotated_tinea_only and "TINEA" in label.upper() and "CLEAN_UNANNOTATED_PHOTO" not in label.upper():
            continue
        try:
            _image_mime_type(image_bytes)
        except ValueError:
            logger.warning("Skipping invalid related analysis image label=%s", label)
            continue
        valid.append((label, image_bytes))
    return valid


async def generate_full_photo_analysis_texts(
    *,
    dorsal_raw_image_bytes: bytes,
    left_plantar_raw_image_bytes: bytes,
    right_plantar_raw_image_bytes: bytes,
    facts: dict[str, Any],
    related_analysis_images: dict[str, bytes],
) -> FullPhotoAnalysisTexts:
    """Prefer validated GPT text, then finish with explicitly sourced local text."""
    # Invalid analysis inputs remain errors; provider failures alone can use
    # completed analysis facts to produce an honest deterministic explanation.
    safe_facts = _json_safe_facts(facts)
    _image_mime_type(dorsal_raw_image_bytes)
    _image_mime_type(left_plantar_raw_image_bytes)
    _image_mime_type(right_plantar_raw_image_bytes)
    inflammation_disabled = inflammation_report_uses_unannotated_images(safe_facts)
    if inflammation_disabled:
        safe_facts = _image_based_skin_facts(safe_facts)

    def fallback(code: str) -> FullPhotoAnalysisTexts:
        return replace(
            build_fallback_full_photo_analysis_texts(facts=safe_facts),
            generation_failure_code=code,
        )

    if not _openai_enabled():
        if settings.openai_report_fallback_enabled:
            return fallback("OPENAI_DISABLED")
        raise RuntimeError("OpenAI report generation is required but is not enabled.")
    if not settings.openai_report_include_images:
        if settings.openai_report_fallback_enabled:
            return fallback("IMAGE_INPUT_DISABLED")
        raise RuntimeError("OpenAI image input is required for full photo reporting.")

    system_prompt = (
        "당신은 발등 원본 한 장, 좌우 발바닥 원본 합본 한 장, 그리고 서버가 "
        "생성한 분석 이미지를 함께 검토해 사용자용 한국어 설명을 작성한다. 제공된 facts의 "
        "각도·점수·치수는 서버가 확정한 값이므로 재계산, 추정, 반올림, 변경하지 않는다. "
        "fungal_suspicion_safety_score와 skin_reaction_safety_score는 모두 안전 점수다. "
        "백 점에 가까울수록 의심 신호가 적고 안전하며, 영 점에 가까울수록 의심 신호가 "
        "강하다. 낮은 안전 점수를 낮은 위험이나 건강한 상태로 반대로 해석하지 않는다. "
        "final_scores는 항목별로 양발 발등 중 더 낮은 안전 점수를 서버가 확정한 값이며, "
        "의심 영역 검출 시 기준 안전 점수에서 시작해 최종 의심 면적 비율이 클수록 낮아진다. "
        "각 부위 점수와 의심 영역을 함께 반영해 설명의 강도를 일치시킨다. "
        "응답에는 새 숫자 점수나 숫자 수치를 쓰지 않고 정성적 관찰만 작성한다. "
        "진단·확진·판정 표현을 사용하지 말고 관찰된 경향을 중심으로 설명한다. 이미지의 "
        "색이나 밝기만으로 질환 또는 압력을 단정하지 않는다. "
        "무좀과 피부 자극 반응은 발등 근거만 고려한다. 이 피부 설명에는 발바닥을 언급하지 않는다. "
        "skin_assessment_source가 wound_segformer_b0 또는 custom_swin_tiny_unet이면 피부 항목은 상처 분할 모델의 결과다. "
        "상처 검출 부위를 설명하고 감염이나 염증을 확진하지 않는다. wound_detected가 true이면 "
        "작은 상처 검출을 없는 것으로 바꾸지 않는다. "
        "segformer_fungal_only_segformer_wound는 양발 발등에서 무좀 전용 SegFormer-B2 Student와 독립 SegFormer 상처 모델을 분석한 상태다. "
        "무좀 모델은 무좀 한 클래스만 예측하며 피부/염증은 독립 상처 모델의 결과를 사용한다. "
        + _FUNGAL_WOUND_REPORT_INSTRUCTIONS + " "
        + (
            "tinea facts에서 발바닥 analysis_mode가 disabled_forced_clear이면 병변 모델의 "
            "결과는 없으므로 사진에서 무좀 신호를 새로 추론하지 않는다. 피부 자극 항목은 "
            "inflammation_analysis_contract에 지정된 표시 없는 사진의 외관을 설명한다. "
            if inflammation_disabled else
            "tinea facts에서 발바닥 analysis_mode가 disabled_forced_clear이면 해당 발바닥에는 "
            "병변 모델을 실행하지 않은 것으로 취급하고, 원본 또는 clean 이미지에서 무좀·피부 "
            "신호를 새로 추론하거나 발바닥 병변을 묘사하지 않는다. "
        )
        + "plantar_footprint_analysis_text는 발바닥 형태를 요족·정상·평발 세 범주 안에서만 "
        "설명한다. "
        + _PLANTAR_COMPOSITE_IMAGE_CONTRACT + " "
        + _PLANTAR_SHAPE_ASSESSMENT_INSTRUCTIONS + " "
        "명확한 좌우 형태 차이가 있을 때만 한 문장 안에서 각각 설명한다. 정상은 "
        "'정상 형태', 요족과 평발은 "
        "'경향이 보입니다' 또는 '경향이 있습니다'로 쓴다. 이 필드에는 '수 있습니다'와 "
        f"'가능성'을 사용하지 않는다. 근거가 부족하면 반드시 '{_PLANTAR_ABSTENTION_TEXT}'라고 "
        "쓴다. plantar_footprint의 영역 수·면적 비율과 관련 "
        "이미지는 압력 센서 근거가 아니다. 압력, 하중, 체중 분포, 착지, 접촉 면적·영역 "
        "수·분포, 균형, 과내전, 과외전은 평가하거나 언급하지 않는다. 비진단 안내 문장도 "
        "작성하지 않는다. 이 필드는 숫자 없이 "
        "백팔십 자 이내의 한국어 한 문장만 사용한다. "
        "metric_advice의 각 항목은 "
        "observation 한 문장과 practical_tip 한 문장으로 작성한다. 마크다운, HTML, URL을 "
        "사용하지 않는다. "
        + _FULL_PHOTO_OUTPUT_CONTRACT_INSTRUCTIONS
    )

    if inflammation_disabled:
        system_prompt += " " + _SKIN_IMAGE_REPORT_INSTRUCTIONS

    attempt_count = 0
    try:
        combined_photo = combine_raw_plantar_photos(
            left_plantar_raw_image_bytes, right_plantar_raw_image_bytes,
        )
        images = [
            ("DORSAL RAW ORIGINAL PHOTO", dorsal_raw_image_bytes),
            (PLANTAR_COMPOSITE_LABEL, combined_photo),
            *_valid_related_analysis_images(
                related_analysis_images, unannotated_tinea_only=inflammation_disabled,
            ),
        ]
        base_user_payload = {
            "task": "전체 사진 분석용 비진단적 문구와 지표별 조언 생성",
            "authoritative_facts_do_not_modify": safe_facts,
            "image_order": [label for label, _ in images],
            "skin_assessment_scope": {
                "allowed_surface": "dorsal",
                "excluded_surface": "plantar",
                "instruction": "무좀과 피부반응은 발등 사진과 발등 모델 수치만 설명한다. 발바닥 사진은 아치·눌림 설명에만 사용하며 피부 상태나 피부 점수를 판단하지 않는다.",
            },
            "plantar_image_contract": _PLANTAR_COMPOSITE_IMAGE_CONTRACT,
            "plantar_shape_assessment_policy": dict(_PLANTAR_SHAPE_ASSESSMENT_POLICY),
            "numeric_output_policy": (
                "facts의 숫자를 재계산하거나 변경하지 말고 응답 문구에 숫자 점수를 출력하지 않는다."
            ),
            "safety_score_semantics": {
                "direction": "higher_is_safer_lower_is_stronger_signal",
                "aggregation": "arithmetic_mean_of_two_dorsal_feet_half_up",
                "instruction": (
                    "낮은 안전 점수를 건강하거나 의심 신호가 적다는 뜻으로 해석하지 않는다."
                ),
            },
            "output_contract_checklist": {
                "top_level": "JSON schema의 필수 key만 빠짐없이 사용하고 추가 key를 만들지 않는다.",
                "general_descriptions": (
                    "각 필드는 숫자와 마크업 없이 한국어 한 문장 이상 세 문장 이하로 작성한다."
                ),
                "plantar_footprint_analysis_text": (
                    "근거가 있으면 정상 형태 또는 요족·평발 경향을 관찰형으로 쓰고, 근거가 "
                    "부족하면 지정된 판단 보류 문장을 쓴다. '수 있습니다'와 '가능성', 압력 "
                    "평가, 비진단 안내 문장은 쓰지 않는다."
                ),
                "metric_advice": (
                    "세 지표 각각 observation 한 문장과 practical_tip 한 문장만 작성하며, "
                    "각 문자열 안의 종결 부호는 맨 끝에 한 번만 둔다."
                ),
            },
        }
        if inflammation_disabled:
            base_user_payload["inflammation_analysis_contract"] = _skin_photo_reporting_contract(
                [label for label, _ in images if "RAW" in label or "CLEAN_UNANNOTATED_PHOTO" in label],
            )
        validation_error: FullPhotoOutputValidationError | None = None
        request_budget = _ReportRequestBudget.for_report(full_photo=True)
        maximum_attempts = request_budget.maximum_attempts
        for validation_attempt in range(1, maximum_attempts + 1):
            attempt_count = max(validation_attempt, request_budget.attempts + 1)
            attempt_prompt = system_prompt
            attempt_payload = dict(base_user_payload)
            if validation_error is not None:
                correction_prompt, retry_metadata = (
                    _full_photo_validation_retry_instructions(
                        validation_error,
                        attempt=attempt_count,
                        maximum_attempts=maximum_attempts,
                    )
                )
                attempt_prompt = f"{system_prompt}\n\n{correction_prompt}"
                attempt_payload["validation_retry"] = retry_metadata

            try:
                data = await _create_structured_response(
                    system_prompt=attempt_prompt,
                    user_payload=attempt_payload,
                    schema_name="full_photo_analysis_texts",
                    json_schema=_full_photo_report_schema(),
                    images=images,
                    image_detail="high",
                    max_output_tokens=2_400,
                    wait_for_response_without_timeout=True,
                    request_budget=request_budget,
                )
                attempt_count = max(attempt_count, request_budget.attempts)
                result = _validate_full_photo_response_data(data)
                return replace(result, inflammation_analysis_enabled=not inflammation_disabled)
            except FullPhotoOutputValidationError as exc:
                current_error = exc

            if (validation_attempt >= maximum_attempts
                    or request_budget.attempts >= maximum_attempts):
                raise current_error

            logger.warning(
                "Required GPT full photo analysis text validation failed; "
                "retrying with the corrective output contract. "
                "attempt=%s max_attempts=%s error_type=%s field=%s rule_code=%s",
                attempt_count,
                maximum_attempts,
                type(current_error).__name__,
                current_error.field,
                current_error.rule_code,
            )
            validation_error = current_error

        raise AssertionError("Full photo output retry loop exited unexpectedly.")
    except FullPhotoOutputValidationError as exc:
        logger.warning(
            "Required GPT full photo analysis text validation failed. "
            "attempts=%s error_type=%s field=%s rule_code=%s",
            attempt_count,
            type(exc).__name__,
            exc.field,
            exc.rule_code,
        )
        if settings.openai_report_fallback_enabled:
            return fallback(_report_failure_code(exc))
        raise
    except Exception as exc:
        logger.warning(
            "Required GPT full photo analysis text generation failed. error_type=%s",
            type(exc).__name__,
        )
        if settings.openai_report_fallback_enabled:
            return fallback(_report_failure_code(exc))
        raise


def _validated_environment_inputs(
    *,
    before_temperature_celsius: float,
    before_humidity_percent: float,
    after_temperature_celsius: float,
    after_humidity_percent: float,
) -> tuple[float, float, float, float, float, float]:
    raw_values = {
        "before_temperature_celsius": before_temperature_celsius,
        "before_humidity_percent": before_humidity_percent,
        "after_temperature_celsius": after_temperature_celsius,
        "after_humidity_percent": after_humidity_percent,
    }
    parsed: dict[str, float] = {}
    for name, value in raw_values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        parsed[name] = number

    for name in ("before_humidity_percent", "after_humidity_percent"):
        if not 0.0 <= parsed[name] <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")

    temperature_delta = (
        parsed["after_temperature_celsius"]
        - parsed["before_temperature_celsius"]
    )
    humidity_delta = (
        parsed["after_humidity_percent"] - parsed["before_humidity_percent"]
    )
    if not math.isfinite(temperature_delta) or not math.isfinite(humidity_delta):
        raise ValueError("Computed environment deltas must be finite.")
    return (
        parsed["before_temperature_celsius"],
        parsed["before_humidity_percent"],
        parsed["after_temperature_celsius"],
        parsed["after_humidity_percent"],
        round(temperature_delta, 4),
        round(humidity_delta, 4),
    )


def build_fallback_foot_environment_report(
    *,
    before_temperature_celsius: float,
    before_humidity_percent: float,
    after_temperature_celsius: float,
    after_humidity_percent: float,
) -> FootEnvironmentReportText:
    (
        _,
        _,
        after_temperature,
        after_humidity,
        temperature_delta,
        humidity_delta,
    ) = _validated_environment_inputs(
        before_temperature_celsius=before_temperature_celsius,
        before_humidity_percent=before_humidity_percent,
        after_temperature_celsius=after_temperature_celsius,
        after_humidity_percent=after_humidity_percent,
    )

    # This deterministic fallback is an environmental-management score, not a
    # medical risk score. The most concerning of absolute post-measurement
    # temperature, absolute humidity, and their positive trend controls the
    # result; a persistently hot or humid reading therefore cannot be hidden by
    # a small delta or by a comfortable value in the other dimension.
    temperature_component = max(
        0.0,
        min(100.0, ((42.0 - after_temperature) / 12.0) * 100.0),
    )
    humidity_component = max(
        0.0,
        min(100.0, ((100.0 - after_humidity) / 40.0) * 100.0),
    )
    trend_component = max(
        0.0,
        min(
            100.0,
            100.0
            - 8.0 * max(0.0, temperature_delta)
            - 1.5 * max(0.0, humidity_delta),
        ),
    )
    score = round(
        min(temperature_component, humidity_component, trend_component),
        1,
    )

    temperature_rose = temperature_delta > 0.5
    humidity_rose = humidity_delta > 2.0
    temperature_fell = temperature_delta < -0.5
    humidity_fell = humidity_delta < -2.0
    temperature_high = after_temperature > 30.0
    humidity_high = after_humidity > 60.0
    if temperature_high and humidity_high:
        observation = (
            "측정 후 발 주변 온도와 습도가 모두 높은 편으로 열과 습기가 머무는 경향이 관찰됩니다."
        )
        practical_tip = (
            "신발과 양말을 충분히 말리고 중간중간 통풍을 유지해 주세요."
        )
    elif temperature_rose and humidity_rose:
        observation = (
            "측정 후 발 주변의 온도와 습도가 함께 올라 열과 습기가 머무는 경향이 관찰됩니다."
        )
        practical_tip = (
            "신발과 양말을 충분히 말리고 중간중간 통풍을 유지해 주세요."
        )
    elif humidity_high or humidity_rose:
        observation = (
            "측정 후 발 주변 습도가 높은 편이거나 상승해 수분이 남는 경향이 관찰됩니다."
        )
        practical_tip = (
            "땀을 잘 흡수하는 양말을 사용하고 신발 내부를 건조하게 관리해 주세요."
        )
    elif temperature_high or temperature_rose:
        observation = (
            "측정 후 발 주변 온도가 높은 편이거나 상승해 열이 머무는 경향이 관찰됩니다."
        )
        practical_tip = (
            "신발을 잠시 벗어 열을 식히고 통풍이 잘 되는 상태를 유지해 주세요."
        )
    elif temperature_fell and humidity_fell:
        observation = (
            "측정 후 온도와 습도가 함께 낮아져 열과 습기가 줄어드는 경향이 보입니다."
        )
        practical_tip = (
            "현재의 건조와 통풍 관리 습관을 유지하며 큰 변화가 생기는지 살펴봐 주세요."
        )
    else:
        observation = (
            "측정 전후 온도와 습도 변화가 모두 크지 않아 발 주변 환경이 비교적 안정적으로 관찰됩니다."
        )
        practical_tip = (
            "현재의 건조와 통풍 관리 습관을 유지하며 변화가 커지는지 살펴봐 주세요."
        )

    return FootEnvironmentReportText(
        score=score,
        advice=(observation, practical_tip),
    )


_ENVIRONMENT_DIAGNOSTIC_PATTERN = re.compile(
    r"진단|확진|판정|(?:무좀|감염|피부염|질환)\s*(?:입니다|이\s*확실)"
)


def _validated_environment_advice(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict) or set(value) != {"observation", "practical_tip"}:
        return None
    observation_parts = _full_report_sentence_list(value.get("observation"))
    tip_parts = _full_report_sentence_list(value.get("practical_tip"))
    if (
        observation_parts is None
        or len(observation_parts) != 1
        or tip_parts is None
        or len(tip_parts) != 1
    ):
        return None
    observation = observation_parts[0]
    practical_tip = tip_parts[0]
    combined = f"{observation}\n{practical_tip}"
    if _ENVIRONMENT_DIAGNOSTIC_PATTERN.search(combined):
        return None
    if not any(
        term in observation
        for term in ("관찰", "보입니다", "경향", "변화", "유지")
    ):
        return None
    if not any(
        term in practical_tip
        for term in ("관리", "유지", "말리", "건조", "통풍", "교체", "해주세요")
    ):
        return None
    if len(observation) > 180 or len(practical_tip) > 180:
        return None
    return observation, practical_tip


def _environment_score_band(score: float) -> str:
    if score < 33.3:
        return "NEED_IMPROVEMENT"
    if score < 66.7:
        return "ATTENTION_NEEDED"
    return "VERY_GOOD"


async def generate_foot_environment_report(
    *,
    before_temperature_celsius: float,
    before_humidity_percent: float,
    after_temperature_celsius: float,
    after_humidity_percent: float,
) -> FootEnvironmentReportText:
    (
        before_temperature,
        before_humidity,
        after_temperature,
        after_humidity,
        temperature_delta,
        humidity_delta,
    ) = _validated_environment_inputs(
        before_temperature_celsius=before_temperature_celsius,
        before_humidity_percent=before_humidity_percent,
        after_temperature_celsius=after_temperature_celsius,
        after_humidity_percent=after_humidity_percent,
    )
    fallback = build_fallback_foot_environment_report(
        before_temperature_celsius=before_temperature,
        before_humidity_percent=before_humidity,
        after_temperature_celsius=after_temperature,
        after_humidity_percent=after_humidity,
    )
    if not _openai_enabled():
        return fallback

    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 100},
            "advice": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "practical_tip": {"type": "string"},
                },
                "required": ["observation", "practical_tip"],
                "additionalProperties": False,
            },
        },
        "required": ["score", "advice"],
        "additionalProperties": False,
    }
    payload = {
        "task": "측정 전후 발 주변 온습도 변화의 환경 점수와 비진단적 조언 생성",
        "raw_measurements": {
            "before_temperature_celsius": before_temperature,
            "before_humidity_percent": before_humidity,
            "after_temperature_celsius": after_temperature,
            "after_humidity_percent": after_humidity,
        },
        "server_computed_deltas": {
            "temperature_delta_celsius": temperature_delta,
            "humidity_delta_percent": humidity_delta,
        },
        "score_contract": (
            "온도와 습도의 전후 수준 및 변화 방향을 모두 고려해 환경 관리 점수를 생성한다. "
            "점수는 높을수록 열과 습기가 안정적으로 관리되는 경향을 뜻하며 의료 위험 점수가 아니다."
        ),
        "advice_contract": (
            "관찰 한 문장과 실천 팁 한 문장을 한국어로 작성하고 숫자, 진단 표현, "
            "마크다운, HTML, URL은 사용하지 않는다."
        ),
    }
    system_prompt = (
        "당신은 측정 전후 발 주변 온도와 습도 값으로 환경 관리 경향을 설명한다. "
        "온도와 습도 중 하나만 무시하지 말고 원시값과 서버 계산 변화량을 모두 고려해 "
        "영에서 백 사이의 점수를 생성한다. 높은 점수는 열과 습기가 비교적 안정적으로 "
        "관리되는 경향, 낮은 점수는 환기와 건조 관리가 더 필요한 경향을 뜻한다. "
        "이 점수는 질환 위험도나 의료 점수가 아니다. 조언은 observation 한 문장과 "
        "practical_tip 한 문장으로 작성한다. 질환을 진단·확진·판정하거나 무좀·감염을 "
        "단정하지 않는다. 두 조언 문장에는 숫자, 마크다운, HTML, URL을 쓰지 않는다."
    )

    try:
        data = await _create_structured_response(
            system_prompt=system_prompt,
            user_payload=payload,
            schema_name="foot_environment_report",
            json_schema=schema,
            images=None,
            max_output_tokens=240,
        )
        raw_score = data.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("OpenAI environment score must be numeric.")
        score = float(raw_score)
        advice = _validated_environment_advice(data.get("advice"))
        if not math.isfinite(score) or not 0.0 <= score <= 100.0 or advice is None:
            raise ValueError("OpenAI environment report did not satisfy the output contract.")
        score = round(score, 1)
        if (
            abs(score - fallback.score) > 10.0
            or _environment_score_band(score)
            != _environment_score_band(fallback.score)
        ):
            raise ValueError("OpenAI environment score diverged from the deterministic safety baseline.")
        return FootEnvironmentReportText(score=score, advice=advice)
    except Exception as exc:
        logger.warning(
            "GPT foot environment report generation failed; using fallback. error=%s",
            exc,
        )
        return fallback


_PRESSURE_SENSOR_COUNT = 12
_PRESSURE_ZONE_RANGES = {
    "forefoot": (0, 5),
    "midfoot": (5, 9),
    "heel": (9, 12),
}
_PRESSURE_SCORE_WEIGHTS = {
    "bilateralBalance": 0.6,
    "leftDistribution": 0.2,
    "rightDistribution": 0.2,
}
_PRESSURE_SCORE_FORMULA = (
    "0.6 * bilateralBalanceScore + 0.2 * leftDistributionScore "
    "+ 0.2 * rightDistributionScore"
)
_PRESSURE_DISTRIBUTION_SCORE_FORMULA = (
    "100 * max(0, 1 - sum(abs(sensorCountAdjustedZoneShare - 1/3)) / (4/3))"
)
_PRESSURE_ZONE_ADJUSTMENT_FORMULA = (
    "zoneMean = zoneSum / zoneSensorCount; adjustedZoneShare = "
    "zoneMean / sum(allZoneMeans)"
)
_PRESSURE_FORMULA_VERSION = "pressure-balance-v2"


def _validated_pressure_values(
    values: Sequence[float],
    *,
    side: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{side} pressure values must contain exactly twelve numbers.")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError(
            f"{side} pressure values must contain exactly twelve numbers."
        ) from exc
    if len(raw_values) != _PRESSURE_SENSOR_COUNT:
        raise ValueError(f"{side} pressure values must contain exactly twelve numbers.")

    validated: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{side} pressure value C{index} must be numeric.")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{side} pressure value C{index} must be numeric.") from exc
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"{side} pressure value C{index} must be finite and nonnegative."
            )
        validated.append(number)
    return tuple(validated)


def _finite_pressure_sum(values: tuple[float, ...], *, side: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{side} pressure total must be finite.") from exc
    if not math.isfinite(total):
        raise ValueError(f"{side} pressure total must be finite.")
    return total


def _pressure_zone_percentages(
    values: tuple[float, ...],
    total: float,
) -> dict[str, float]:
    if total <= 0.0:
        return {zone: 0.0 for zone in _PRESSURE_ZONE_RANGES}
    return {
        zone: round(math.fsum(values[start:end]) / total * 100.0, 1)
        for zone, (start, end) in _PRESSURE_ZONE_RANGES.items()
    }


def _pressure_sensor_count_adjusted_zone_shares(
    values: tuple[float, ...],
    total: float,
) -> dict[str, float]:
    if total <= 0.0:
        return {zone: 0.0 for zone in _PRESSURE_ZONE_RANGES}
    zone_means = {
        zone: math.fsum(values[start:end]) / (end - start)
        for zone, (start, end) in _PRESSURE_ZONE_RANGES.items()
    }
    mean_total = math.fsum(zone_means.values())
    if mean_total <= 0.0:
        return {zone: 0.0 for zone in _PRESSURE_ZONE_RANGES}
    return {zone: zone_mean / mean_total for zone, zone_mean in zone_means.items()}


def _rounded_zone_percentages_from_shares(
    zone_shares: dict[str, float],
    *,
    total: float,
) -> dict[str, float]:
    if total <= 0.0:
        return {zone: 0.0 for zone in _PRESSURE_ZONE_RANGES}
    zones = list(_PRESSURE_ZONE_RANGES)
    rounded = {
        zone: round(zone_shares[zone] * 100.0, 1)
        for zone in zones[:-1]
    }
    rounded[zones[-1]] = round(100.0 - math.fsum(rounded.values()), 1)
    return rounded


def _pressure_distribution_score(zone_shares: dict[str, float]) -> float:
    """Return a non-clinical three-zone spread index on a 0-100 scale.

    A score of 100 means the sensor-count-adjusted mean loads are split equally
    among the three configured longitudinal zones. A score of 0 means all
    measured load is in one zone. This is deliberately a transparent,
    provisional dispersion index, not a normal/abnormal plantar-pressure
    classification.
    """

    target_share = 1.0 / len(_PRESSURE_ZONE_RANGES)
    maximum_absolute_deviation = 4.0 / 3.0
    absolute_deviation = math.fsum(
        abs(share - target_share) for share in zone_shares.values()
    )
    score = 100.0 * max(
        0.0,
        1.0 - absolute_deviation / maximum_absolute_deviation,
    )
    return round(min(100.0, score), 1)


def _pressure_distribution_level(score: float | None) -> str:
    if score is None:
        return "INCONCLUSIVE"
    if score >= 80.0:
        return "BROADLY_DISTRIBUTED"
    if score >= 50.0:
        return "PARTLY_CONCENTRATED"
    return "CONCENTRATED"


def _dominant_pressure_zones(
    zone_percentages: dict[str, float],
    *,
    total: float,
) -> list[str]:
    if total <= 0.0:
        return []
    maximum = max(zone_percentages.values())
    ranked = sorted(zone_percentages.values(), reverse=True)
    # Treat a top-two gap within five percentage points as a mixed dominant
    # distribution. No sensor noise floor is applied here because the hardware
    # calibration threshold has not yet been established.
    mixed_dominance = len(ranked) > 1 and ranked[0] - ranked[1] <= 5.0
    return [
        zone.upper()
        for zone, percentage in zone_percentages.items()
        if percentage == maximum
        or (mixed_dominance and maximum - percentage <= 5.0)
    ]


def build_pressure_balance_facts(
    left_pressure_values: Sequence[float],
    right_pressure_values: Sequence[float],
) -> dict[str, Any]:
    """Build authoritative, non-diagnostic pressure-balance measurements.

    The final score is a deterministic weighted composite: 60% bilateral load
    balance plus 20% for each foot's transparent three-zone dispersion index.
    The dispersion index uses equal thirds only as a mathematical spread
    reference; it is not a clinical normal-pressure reference. GPT never
    calculates or changes these values. Missing load on either side is marked
    inconclusive with a nullable score instead of being presented as imbalance.
    """

    left = _validated_pressure_values(left_pressure_values, side="Left")
    right_raw = _validated_pressure_values(right_pressure_values, side="Right")
    right = canonicalize_right_heel_first(right_raw)
    left_total = _finite_pressure_sum(left, side="Left")
    right_total = _finite_pressure_sum(right, side="Right")
    pair_total = _finite_pressure_sum((left_total, right_total), side="Combined")
    if pair_total <= 0.0:
        left_load_percent = 0.0
        right_load_percent = 0.0
        load_imbalance = 0.0
        profile_asymmetry = 0.0
        quality_status = "no-load"
    else:
        left_load_percent = round(left_total / pair_total * 100.0, 1)
        # Deriving the right percentage from the rounded left value guarantees
        # the public API invariant L% + R% == 100.0 for every loaded frame.
        right_load_percent = round(100.0 - left_load_percent, 1)
        load_imbalance = abs(left_load_percent - right_load_percent) / 100.0
        quality_status = (
            "single-side-no-load"
            if left_total <= 0.0 or right_total <= 0.0
            else "valid"
        )

    load_difference = round(load_imbalance * 100.0, 1)
    if quality_status == "single-side-no-load":
        profile_asymmetry = 1.0
    elif quality_status == "valid":
        left_profile = tuple(value / left_total for value in left)
        right_profile = tuple(value / right_total for value in right)
        profile_asymmetry = 0.5 * math.fsum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_profile,
                right_profile,
                strict=True,
            )
        )
        profile_asymmetry = max(0.0, min(1.0, profile_asymmetry))

    bilateral_balance_score = (
        round(max(0.0, min(100.0, 100.0 - load_difference)), 1)
        if quality_status == "valid"
        else None
    )
    if quality_status != "valid":
        load_balance_level = "INCONCLUSIVE"
    elif load_imbalance * 100.0 <= 3.0:
        load_balance_level = "BALANCED"
    elif bilateral_balance_score < 33.3:
        load_balance_level = "EXTREME_ONE_SIDED"
    elif bilateral_balance_score < 66.7:
        load_balance_level = "STRONG_ONE_SIDED"
    else:
        load_balance_level = "NOTICEABLE_SHIFT"
    if quality_status == "no-load" or load_imbalance * 100.0 <= 3.0:
        dominant_foot = "BALANCED"
    elif left_load_percent > right_load_percent:
        dominant_foot = "LEFT"
    else:
        dominant_foot = "RIGHT"

    left_raw_zone_load_percent = _pressure_zone_percentages(left, left_total)
    right_raw_zone_load_percent = _pressure_zone_percentages(right, right_total)
    left_zone_shares = _pressure_sensor_count_adjusted_zone_shares(
        left,
        left_total,
    )
    right_zone_shares = _pressure_sensor_count_adjusted_zone_shares(
        right,
        right_total,
    )
    left_zone_percent = _rounded_zone_percentages_from_shares(
        left_zone_shares,
        total=left_total,
    )
    right_zone_percent = _rounded_zone_percentages_from_shares(
        right_zone_shares,
        total=right_total,
    )
    left_distribution_score = (
        _pressure_distribution_score(left_zone_shares)
        if left_total > 0.0
        else None
    )
    right_distribution_score = (
        _pressure_distribution_score(right_zone_shares)
        if right_total > 0.0
        else None
    )
    distribution_score = (
        round((left_distribution_score + right_distribution_score) / 2.0, 1)
        if left_distribution_score is not None
        and right_distribution_score is not None
        else None
    )
    balance_score = (
        round(
            _PRESSURE_SCORE_WEIGHTS["bilateralBalance"]
            * bilateral_balance_score
            + _PRESSURE_SCORE_WEIGHTS["leftDistribution"]
            * left_distribution_score
            + _PRESSURE_SCORE_WEIGHTS["rightDistribution"]
            * right_distribution_score,
            1,
        )
        if quality_status == "valid"
        and bilateral_balance_score is not None
        and left_distribution_score is not None
        and right_distribution_score is not None
        else None
    )

    return {
        "leftTotal": round(left_total, 4),
        "rightTotal": round(right_total, 4),
        "leftLoadPercent": left_load_percent,
        "rightLoadPercent": right_load_percent,
        "loadDifferencePercentagePoints": load_difference,
        "loadImbalancePercent": round(load_imbalance * 100.0, 1),
        "profileAsymmetryPercent": round(profile_asymmetry * 100.0, 1),
        "leftActiveSensorCount": sum(value > 0.0 for value in left),
        "rightActiveSensorCount": sum(value > 0.0 for value in right),
        "bilateralBalanceScore": bilateral_balance_score,
        "leftDistributionScore": left_distribution_score,
        "rightDistributionScore": right_distribution_score,
        "distributionScore": distribution_score,
        "balanceScore": balance_score,
        "scoreComponents": {
            "bilateralBalanceScore": bilateral_balance_score,
            "leftDistributionScore": left_distribution_score,
            "rightDistributionScore": right_distribution_score,
        },
        "scoreWeights": dict(_PRESSURE_SCORE_WEIGHTS),
        "scoreFormula": _PRESSURE_SCORE_FORMULA,
        "formulaVersion": _PRESSURE_FORMULA_VERSION,
        "distributionScoreFormula": _PRESSURE_DISTRIBUTION_SCORE_FORMULA,
        "zoneAdjustmentFormula": _PRESSURE_ZONE_ADJUSTMENT_FORMULA,
        "distributionScoreMetadata": {
            "provisional": True,
            "nonClinical": True,
            "meaning": "sensor-count-adjusted three-zone spread index",
        },
        "loadBalanceLevel": load_balance_level,
        "qualityStatus": quality_status,
        "leftRawZoneLoadPercent": left_raw_zone_load_percent,
        "rightRawZoneLoadPercent": right_raw_zone_load_percent,
        "leftZonePercent": left_zone_percent,
        "rightZonePercent": right_zone_percent,
        "leftSensorCountAdjustedZonePercent": left_zone_percent,
        "rightSensorCountAdjustedZonePercent": right_zone_percent,
        "leftDominantZones": _dominant_pressure_zones(
            left_zone_percent,
            total=left_total,
        ),
        "rightDominantZones": _dominant_pressure_zones(
            right_zone_percent,
            total=right_total,
        ),
        "leftDistributionLevel": _pressure_distribution_level(
            left_distribution_score
        ),
        "rightDistributionLevel": _pressure_distribution_level(
            right_distribution_score
        ),
        "dominantFoot": dominant_foot,
    }


def _strict_value_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_value_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_value_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


_PRESSURE_ZONE_LABELS = {
    "FOREFOOT": "앞꿈치",
    "MIDFOOT": "중족부",
    "HEEL": "뒤꿈치",
}


def _regional_pressure_observation(
    side_label: str,
    zones: list[str],
    distribution_level: str,
) -> str:
    labels = [_PRESSURE_ZONE_LABELS[zone] for zone in zones]
    if distribution_level == "BROADLY_DISTRIBUTED" and len(labels) == len(
        _PRESSURE_ZONE_LABELS
    ):
        return (
            f"{side_label}은 {'·'.join(labels)} 세 구역에 압력이 비교적 고르게 "
            "분산되는 경향이 관찰됩니다."
        )
    if len(labels) == 1:
        zone_text = labels[0]
        dominant_clause = f"{zone_text} 구역의 압력 비중이 상대적으로"
    else:
        zone_text = "와 ".join(labels)
        dominant_clause = f"{zone_text} 구역의 압력 비중이 함께 상대적으로"
    if distribution_level == "BROADLY_DISTRIBUTED":
        return (
            f"{side_label}은 {dominant_clause} 크지만 세 구역에 "
            "압력이 비교적 고르게 분산되는 경향이 관찰됩니다."
        )
    if distribution_level == "PARTLY_CONCENTRATED":
        return (
            f"{side_label}은 {dominant_clause} 크며 구역별 압력 "
            "비중에 일부 차이가 관찰됩니다."
        )
    return (
        f"{side_label}은 {dominant_clause} 크고 특정 구역에 압력이 "
        "집중되는 경향이 관찰됩니다."
    )


def _pressure_fallback_components(
    pressure_facts: dict[str, Any],
) -> tuple[str, str, str, str]:
    left_regional = _regional_pressure_observation(
        "왼발",
        pressure_facts["leftDominantZones"],
        pressure_facts["leftDistributionLevel"],
    )
    right_regional = _regional_pressure_observation(
        "오른발",
        pressure_facts["rightDominantZones"],
        pressure_facts["rightDistributionLevel"],
    )
    dominant_foot = pressure_facts["dominantFoot"]
    load_balance_level = pressure_facts["loadBalanceLevel"]
    strongly_one_sided = load_balance_level in {
        "STRONG_ONE_SIDED",
        "EXTREME_ONE_SIDED",
    }
    if dominant_foot == "LEFT" and strongly_one_sided:
        overall_observation = (
            "왼발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다."
        )
    elif dominant_foot == "RIGHT" and strongly_one_sided:
        overall_observation = (
            "오른발 쪽으로 압력이 크게 치우치는 경향이 관찰됩니다."
        )
    elif dominant_foot == "LEFT":
        overall_observation = (
            "왼발에 전달된 압력이 오른발보다 상대적으로 큰 경향이 관찰됩니다."
        )
    elif dominant_foot == "RIGHT":
        overall_observation = (
            "오른발에 전달된 압력이 왼발보다 상대적으로 큰 경향이 관찰됩니다."
        )
    else:
        overall_observation = (
            "좌우 발에 전달된 압력이 비교적 고르게 분포하는 경향이 관찰됩니다."
        )

    if load_balance_level == "BALANCED":
        practical_tip = (
            "현재의 좌우 체중 분배를 유지하며 양발에 고르게 힘을 싣는 습관을 이어가 주세요."
        )
    elif strongly_one_sided:
        practical_tip = (
            "측정 자세를 바로잡은 뒤 양발에 고르게 힘을 싣고 다시 살펴봐 주세요."
        )
    else:
        practical_tip = (
            "서 있을 때 한쪽으로 치우치지 않도록 양발에 천천히 고르게 힘을 실어 주세요."
        )
    return left_regional, right_regional, overall_observation, practical_tip


class PressureOutputValidationError(ReportGenerationError):
    """A rejected field and rule, without retaining the generated patient text."""

    def __init__(self, field: str, rule_code: str) -> None:
        self.field = field
        self.rule_code = rule_code
        super().__init__("OUTPUT_VALIDATION_FAILED")


def _require_pressure_advice(
    value: Any,
    pressure_facts: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Validate required text fields only; server calculations own the metrics.

    Wording, sentence count, and zone descriptions are prompt instructions, not
    reasons to discard a completed response and repeat the provider request.
    """
    fields = (
        "leftRegionalObservation",
        "rightRegionalObservation",
        "overallBalanceObservation",
        "practicalTip",
    )
    if not isinstance(value, dict) or set(value) != set(fields):
        raise PressureOutputValidationError("response", "REQUIRED_FIELDS_MISMATCH")
    cleaned: list[str] = []
    for field in fields:
        text = value[field]
        if not isinstance(text, str) or not text.strip():
            raise PressureOutputValidationError(field, "TEXT_REQUIRED")
        cleaned.append(text.strip())
    return cleaned[0], cleaned[1], cleaned[2], cleaned[3]


def _validated_pressure_advice(
    value: Any,
    pressure_facts: dict[str, Any],
) -> tuple[str, str, str, str] | None:
    try:
        return _require_pressure_advice(value, pressure_facts)
    except PressureOutputValidationError:
        return None


async def generate_pressure_balance_report(
    *,
    left_pressure_values: Sequence[float],
    right_pressure_values: Sequence[float],
    pressure_facts: dict[str, Any],
    left_heatmap_png: bytes,
    right_heatmap_png: bytes,
) -> PressureBalanceReportText:
    left = _validated_pressure_values(left_pressure_values, side="Left")
    right = _validated_pressure_values(right_pressure_values, side="Right")
    expected_facts = build_pressure_balance_facts(left, right)
    if not isinstance(pressure_facts, dict) or not _strict_value_equal(
        pressure_facts,
        expected_facts,
    ):
        raise ValueError(
            "pressure_facts must exactly match build_pressure_balance_facts output."
        )
    if expected_facts["qualityStatus"] != "valid":
        raise ValueError(
            "Pressure balance report generation requires positive load on both sides."
        )
    (
        fallback_left_regional,
        fallback_right_regional,
        fallback_overall,
        fallback_tip,
    ) = _pressure_fallback_components(expected_facts)
    fallback = PressureBalanceReportText(
        score=expected_facts["balanceScore"],
        left_regional_observation=fallback_left_regional,
        right_regional_observation=fallback_right_regional,
        overall_balance_observation=fallback_overall,
        practical_tip=fallback_tip,
        generation_source="DETERMINISTIC_FALLBACK",
    )
    unavailable_code = (
        "OPENAI_DISABLED" if not _openai_enabled() else
        "IMAGE_INPUT_DISABLED" if not settings.openai_report_include_images else None
    )
    if unavailable_code is not None:
        if not settings.openai_report_fallback_enabled:
            raise ReportGenerationError(unavailable_code)
        return replace(fallback, generation_failure_code=unavailable_code)

    try:
        # Heatmaps are derived display assets. Valid sensor facts above remain
        # sufficient for a local explanation if a derived image is unusable.
        try:
            if (_image_mime_type(left_heatmap_png) != "image/png"
                    or _image_mime_type(right_heatmap_png) != "image/png"):
                raise ValueError("Pressure heatmaps must be PNG images.")
        except ValueError:
            raise ReportGenerationError("INVALID_DERIVED_IMAGE") from None
        payload = {
            "task": "좌우 히트맵의 공간적 압력 양상과 센서 계산값을 함께 검토한 관찰 및 실천 팁 생성",
            "numeric_source_of_truth": {
                "serverComputedPressureFacts": expected_facts,
            },
            "sentence_grounding_examples": {
                "leftRegionalObservation": fallback_left_regional,
                "rightRegionalObservation": fallback_right_regional,
                "overallBalanceObservation": fallback_overall,
                "practicalTip": fallback_tip,
            },
            "sentence_grounding_contract": (
                "sentence_grounding_examples는 서버 수치와 일치하는 기본 문장이다. "
                "기본 문장의 수치적 의미를 유지하면서 자연스러운 한국어로 바꾸거나 "
                "실제 히트맵에서 확인한 색의 모임·퍼짐을 덧붙일 수 있다. 특정 어휘나 "
                "문장 시작 표현을 그대로 복사할 필요는 없다. 시각적 특징이 불명확하면 "
                "이를 만들어내지 않고 한계를 표현한다. 좌우 총하중과 구역별 비중은 "
                "서버 계산값을 따르고 실천 팁은 그 결과에 맞는 체중 분배 행동을 설명한다."
            ),
            "score_authority_contract": (
                "balanceScore, bilateralBalanceScore, leftDistributionScore, "
                "rightDistributionScore와 scoreComponents/scoreWeights/scoreFormula는 서버가 "
                "확정한 값이다. 모델은 점수를 계산, 추정, 보정, 변경하거나 출력하지 않는다. "
                "각 발 distributionScore는 세 구역을 동일 비중 기준과 비교한 설명용 분산도 "
                "지수이며 임상 정상치나 좋은 발을 뜻하지 않는다."
            ),
            "one_sided_load_contract": (
                "loadBalanceLevel이 STRONG_ONE_SIDED 또는 EXTREME_ONE_SIDED이면 "
                "dominantFoot 방향을 명시하고 압력이 크게 치우치는 관찰 경향을 표현한다. "
                "그 외 단계에는 과도한 쏠림을 새로 만들어 내지 않는다."
            ),
            "regional_contract": (
                "leftZonePercent와 rightZonePercent는 구역별 압력 비중의 수치 근거이며, "
                "각 구역 안에서 강한 색 영역이 모이거나 퍼지는 공간적 양상은 해당 발의 "
                "히트맵을 직접 보고 설명한다. "
                "leftDominantZones와 rightDominantZones의 모든 구역을 각각 왼발과 오른발 "
                "관찰에 정확히 포함하고, 우세하지 않은 구역을 우세하다고 표현하지 않는다. "
                "FOREFOOT은 앞꿈치, MIDFOOT은 중족부, HEEL은 뒤꿈치라는 표현만 사용한다. "
                "각 발 DistributionLevel이 BROADLY_DISTRIBUTED일 때만 '비교적 고르게 분산'을 "
                "사용한다. 세 구역이 모두 dominant이면 '앞꿈치·중족부·뒤꿈치 세 구역에 "
                "압력이 비교적 고르게 분산'처럼 우세 구역 없이 직접 표현한다. "
                "PARTLY_CONCENTRATED는 '구역별 비중에 일부 차이', CONCENTRATED는 "
                "'특정 구역에 집중되는 경향'으로 표현한다. "
                "이 구역별 분산도와 구역 안의 국소적인 색 집중은 서로 다른 관찰이다. "
                "세 구역에 비교적 고르게 분산되어도 각 구역 내부에 좁은 고강도 부분이 "
                "보일 수 있으므로 이를 모순으로 취급하지 않는다."
            ),
            "heatmap_reading_contract": {
                "source_role": "spatial_visual_evidence",
                "orientation": {
                    "top": "FOREFOOT",
                    "middle": "MIDFOOT",
                    "bottom": "HEEL",
                    "right_display_transform": "horizontal_flip_only",
                },
                "color_scale": {
                    "normalization": "independent_foot_peak_and_mask_minmax",
                    "low_to_high": [
                        "blue", "cyan", "green", "yellow", "orange", "red", "dark_red",
                    ],
                    "linear_pressure_scale": False,
                },
                "review_each_foot": (
                    "왼발과 오른발 이미지를 각각 확인해 우세 구역 안의 고강도 색 영역이 "
                    "좁게 모이는지, 여러 곳으로 나뉘는지, 넓게 이어지는지와 주변으로의 색 "
                    "변화를 살펴본다. 각 발 관찰 문장에는 실제로 확인되는 시각적 특징을 "
                    "하나 이상 구체적으로 포함한다. 수치만으로 색 영역의 모양을 추측하지 "
                    "않고 세부 형태가 불명확하면 그 한계를 짧게 표현한다."
                ),
                "compare_feet": (
                    "각 발은 자체 최대 센서값과 발 내부 최솟값·최댓값으로 따로 정규화하므로 "
                    "좌우 색의 진하기를 서로 같은 압력 크기로 비교하지 않는다. 각 발 안의 "
                    "고강도 영역 위치와 모임·퍼짐을 관찰한다. 좌우 총하중 방향과 강도는 "
                    "dominantFoot와 loadBalanceLevel을 따른다. 서로 다른 발이나 측정의 "
                    "색을 같은 절대 압력으로 비교하지 않는다."
                ),
                "reconcile_with_sensor_facts": (
                    "구역별 비중은 센서 개수를 보정한 값이며 히트맵은 센서값을 보간한 "
                    "그림이라 가장 진한 위치와 구역 평균의 우세 방향이 다를 수 있다. "
                    "이때 우세 구역과 점수는 facts대로 쓰고 해당 구역에서 확인되는 색의 "
                    "퍼짐이나 집중만 덧붙인다. 수치와 맞지 않는 시각적 특징은 만들어내지 "
                    "않는다."
                ),
            },
            "image_contract": (
                "두 이미지를 공간적 관찰의 주요 근거로 사용한다. 발가락은 위, 뒤꿈치는 "
                "아래이며 RIGHT의 표시용 수평 반전은 세로 구역을 바꾸지 않는다. "
                "파랑에서 청록·초록·노랑·주황·빨강·짙은 빨강으로 갈수록 표시 강도가 높다. "
                "센서값을 발마다 정규화하고 Multiquadric RBF와 센서 중심 Gaussian을 "
                "결합하며 발 내부 정규화, 경계 감쇠, 강조 및 비선형 색 보정을 적용한다. "
                "색의 진하기나 면적으로 정확한 총하중·구역별 하중 비율을 역산하지 않는다. "
                "투명 배경은 분석에서 제외한다. 발 윤곽은 개인의 실제 발을 추출한 모양이 "
                "아닌 표시용 큰·작은 템플릿이며 일부 약한 색 영역에는 센서 사이 보간과 "
                "보간용 가상 영점의 영향도 있다. 윤곽이나 색칠된 면적을 "
                "실제 접촉 면적, 발 형태, 비접촉 또는 질환 근거로 사용하지 않는다."
            ),
            "output_contract": (
                "점수나 숫자를 출력하지 않고 왼발 구역 관찰, 오른발 구역 관찰, 전체 좌우 "
                "균형 관찰, 실천 팁을 각각 백팔십 자 이내의 한국어 한 문장으로 작성한다. "
                "각 발 관찰은 해당 발 이름으로 시작해 수치로 확인된 우세 구역과 분산도에 "
                "히트맵의 시각적 특징을 결합한다. 예를 들어 실제 이미지에서 확인될 때만 "
                "'왼발은 앞꿈치 구역의 압력 비중이 상대적으로 크며 히트맵에서도 강한 색 "
                "영역이 좁게 모이는 집중 경향이 관찰됩니다.'처럼 쓴다. 실천 팁은 일상에서 "
                "양발에 힘을 고르게 싣는 행동으로 설명한다. 아치, 내측, 외측, "
                "과내전, 과외전, 평발, 요족 또는 질환을 추론하거나 진단하지 않는다. "
                "정상/비정상, 좋은 발/나쁜 발처럼 임상적이거나 가치 판단적인 표현도 쓰지 않는다."
            ),
        }
        schema = {
            "type": "object",
            "properties": {
                "leftRegionalObservation": {
                    "type": "string",
                    "description": (
                        "왼발로 시작하는 한 문장: 서버가 정한 우세 구역과 구역별 분산도, "
                        "왼발 히트맵에서 직접 확인되는 고강도 영역의 모임·퍼짐·연속성 "
                        "중 구체적인 특징을 함께 설명한다."
                    ),
                },
                "rightRegionalObservation": {
                    "type": "string",
                    "description": (
                        "오른발로 시작하는 한 문장: 서버가 정한 우세 구역과 구역별 분산도, "
                        "오른발 히트맵에서 직접 확인되는 고강도 영역의 모임·퍼짐·연속성 "
                        "중 구체적인 특징을 함께 설명한다."
                    ),
                },
                "overallBalanceObservation": {
                    "type": "string",
                    "description": (
                        "서버가 정한 좌우 총하중 방향과 강도에 맞는 관찰 한 문장으로, "
                        "같은 측정의 좌우 히트맵에서 확인되는 상대적인 시각적 차이를 "
                        "함께 고려한다. 구역명이나 새 수치는 출력하지 않는다."
                    ),
                },
                "practicalTip": {
                    "type": "string",
                    "description": "관찰된 분포를 고려해 양발에 고르게 힘을 싣도록 돕는 실천 팁 한 문장.",
                },
            },
            "required": [
                "leftRegionalObservation",
                "rightRegionalObservation",
                "overallBalanceObservation",
                "practicalTip",
            ],
            "additionalProperties": False,
        }
        system_prompt = (
                "당신은 좌우 발 압력 센서와 히트맵을 함께 분석하는 한국어 리포트 작성자다. "
                "sentence_grounding_examples의 수치적 의미와 sentence_grounding_contract를 "
                "따르고 각 발 문장에 확인 가능한 시각적 관찰을 자연스러운 표현으로 결합한다. "
                "먼저 두 히트맵을 각각 자세히 확인하고, 각 발의 강한 색이 국소적으로 "
                "모이는지, 여러 곳으로 나뉘는지, 넓게 이어지는지를 관찰한다. "
                "이 시각적 특징을 각 발의 구역 관찰에 구체적으로 반영하고 서버 계산 "
                "facts는 점수·비율·우세 구역·구역별 분산도의 확정 근거로 사용한다. "
                "히트맵에서 확인되지 않는 색이나 공간적 양상을 수치만 보고 만들어내지 "
                "않는다. 각 발의 zone facts에서 우세 구역만 정확히 언급하고 왼발과 "
                "오른발을 서로 바꾸지 않는다. 오른발 히트맵의 표시용 수평 반전을 안쪽과 바깥쪽 "
                "근거로 해석하지 않는다. dominantFoot와 loadBalanceLevel도 서로 모순 없이 "
                "반영한다. 점수와 비율은 서버 확정값이며 계산하거나 수정하지 않는다. "
                "각 발 DistributionLevel에 맞춰 비교적 고르게 분산 또는 특정 구역에 상대적으로 "
                "집중되는 경향을 설명하며 해당 구역 안의 시각적 특징을 함께 담는다. "
                "구역 평균의 분산도와 구역 내부의 국소 색 집중을 구분한다. "
                "이를 정상/비정상 또는 좋은/나쁜 발로 해석하지 않는다. "
                "아치 형태, 안쪽과 바깥쪽 분포, 과내전, 과외전, 평발, 요족은 추론하지 않는다. "
                "네 필드는 각각 한국어 한 문장이어야 하며 숫자, 점수, 질환명, 진단, 확진, 판정, "
                "마크다운, HTML, URL을 출력하지 않는다."
        )
        request_budget = _ReportRequestBudget.for_report()
        maximum_attempts = request_budget.maximum_attempts
        validation_error: PressureOutputValidationError | None = None
        for validation_attempt in range(1, maximum_attempts + 1):
            attempt_payload = dict(payload)
            if validation_error is not None:
                attempt_payload["validation_retry"] = {
                    "attempt": max(validation_attempt, request_budget.attempts + 1),
                    "maximum_attempts": maximum_attempts,
                    "failed_field": validation_error.field,
                    "failed_rule_code": validation_error.rule_code,
                    "expected_grounding_sentence": payload["sentence_grounding_examples"].get(
                        validation_error.field,
                    ),
                    "instruction": (
                        "failed_field가 비어 있거나 문자열이 아니거나 필수 필드 구성이 잘못되었다. "
                        "필수 네 필드를 모두 비어 있지 않은 문자열로 작성해 전체 JSON을 다시 보낸다. "
                        "같은 수치 facts와 히트맵을 사용하고 expected_grounding_sentence의 "
                        "수치적 의미를 참고하되 자연스러운 문장으로 설명한다."
                    ),
                }
            data = await _create_structured_response(
                system_prompt=system_prompt,
                user_payload=attempt_payload,
                schema_name="pressure_balance_report",
                json_schema=schema,
                images=[
                    ("LEFT DERIVED PRESSURE HEATMAP", left_heatmap_png),
                    ("RIGHT DERIVED PRESSURE HEATMAP", right_heatmap_png),
                ],
                image_detail="high",
                max_output_tokens=720,
                request_budget=request_budget,
            )
            try:
                validated = _require_pressure_advice(data, expected_facts)
                break
            except PressureOutputValidationError as exc:
                validation_error = exc
            if (validation_attempt >= maximum_attempts
                    or request_budget.attempts >= maximum_attempts):
                raise validation_error
            logger.warning(
                "GPT pressure text validation failed; retrying. attempt=%s max_attempts=%s "
                "code=%s field=%s rule_code=%s",
                max(validation_attempt, request_budget.attempts),
                maximum_attempts,
                "OUTPUT_VALIDATION_FAILED",
                validation_error.field,
                validation_error.rule_code,
            )
        else:
            raise AssertionError("Pressure output retry loop exited unexpectedly.")
        left_regional, right_regional, overall_observation, practical_tip = validated
        return PressureBalanceReportText(
            score=expected_facts["balanceScore"],
            left_regional_observation=left_regional,
            right_regional_observation=right_regional,
            overall_balance_observation=overall_observation,
            practical_tip=practical_tip,
        )
    except Exception as exc:
        logger.warning(
            "GPT pressure balance report generation failed. code=%s field=%s rule_code=%s",
            _report_failure_code(exc),
            exc.field if isinstance(exc, PressureOutputValidationError) else None,
            exc.rule_code if isinstance(exc, PressureOutputValidationError) else None,
        )
        if not settings.openai_report_fallback_enabled:
            raise
        return replace(fallback, generation_failure_code=_report_failure_code(exc))


async def generate_hallux_score_analysis_text(left_angle: float, right_angle: float, fallback_text: str) -> str:
    fallback = _compact_text(fallback_text, fallback_text, max_chars=160)
    if not _openai_enabled():
        return fallback

    schema = {
        "type": "object",
        "properties": {"score_analysis_text": {"type": "string"}},
        "required": ["score_analysis_text"],
        "additionalProperties": False,
    }
    payload = {
        "task": "무지외반 HVA 각도 리포트 한 줄 문구 생성",
        "left_toe_angle_degree": round(float(left_angle), 1),
        "right_toe_angle_degree": round(float(right_angle), 1),
        "angle_rule": {
            "normal": "15도 미만",
            "borderline": "15도 이상 20도 미만",
            "progressed": "20도 이상 40도 미만",
            "severe": "40도 이상",
        },
        "style": [
            "한국어 한 문장만 작성한다.",
            "더 높은 쪽과 각도를 포함한다.",
            "양쪽 모두 15도 이상이면 양발 모두를 언급한다.",
            "진단 확정이 아니라 관리/주의 필요성으로 표현한다.",
        ],
        "fallback_text": fallback,
    }

    try:
        data = await _create_structured_response(
            system_prompt=(
                "너는 무지외반 AI 분석 결과를 사용자에게 설명하는 한국어 리포트 작성자다. "
                "HVA 각도만 근거로 간단한 scoreAnalysisText 한 문장을 작성한다."
            ),
            user_payload=payload,
            schema_name="hallux_report_text",
            json_schema=schema,
            images=None,
            max_output_tokens=160,
        )
    except Exception as exc:
        logger.warning("GPT hallux report text generation failed; using fallback. error=%s", exc)
        return fallback

    return _compact_text(data.get("score_analysis_text"), fallback, max_chars=160)
