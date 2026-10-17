from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import ValidationError

from app.core.config import settings
from app.prompts.shoe_fit_comment_prompts import (
    SCHEMA_CORRECTION_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.schemas.shoe_server import ServerPersonalFitEvidence, ServerShoeCharacteristics
from app.services.shoe.shoe_feature_rules import (
    REASON_TYPES,
    _has_negated_term,
    _has_unnegated_term,
    _point_subject_contexts,
    normalize_review_text,
)
from app.services.shoe.shoe_point_summary import (
    PointEvidenceFact,
    consensus_size_unit_step,
    format_point_evidence_facts,
    point_evidence_strength,
    prepare_point_evidence_facts,
    render_exact_lab_claim,
    render_product_point_summary,
)
from app.services.shoe.shoe_review_evidence_policy import (
    NO_REVIEW_SUMMARY,
    ReviewCandidate,
    analyze_reason_review,
    render_reason_summary,
    select_review_evidence,
    validate_reason_output,
)

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_DISPLAY_MODEL_CODE_SUFFIX_PATTERN = re.compile(
    r"\s*/\s*[A-Z0-9][A-Z0-9._-]{4,}$", re.IGNORECASE
)
_NO_REVIEW_NOTE = "없음 (판단만 근거로 사용하고, 구체적인 리뷰 내용을 지어내지 말 것)"
_SHOE_FIT_SUMMARY_JSON_SCHEMA = ShoeFitSummary.model_json_schema(by_alias=True)
# ``point_summary_claims`` has an internal default so deterministic fallbacks can
# be represented when no product evidence exists.  For an Ollama response,
# however, omitting the field makes every non-trivial pointSummary impossible to
# verify and silently turns a successful inference into generic fallback copy.
# Tighten only the generation schema, while preserving the internal model's
# useful default.
_summary_schema_required = _SHOE_FIT_SUMMARY_JSON_SCHEMA.setdefault("required", [])
if "pointSummaryClaims" not in _summary_schema_required:
    _summary_schema_required.append("pointSummaryClaims")

_REASON_LABEL = {"FOREFOOT": "발볼/앞코", "HEEL": "뒤꿈치", "INSOLE": "깔창"}
_SUMMARY_FIELD_BY_REASON = {
    "FOREFOOT": "forefoot_summary",
    "HEEL": "heel_summary",
    "INSOLE": "insole_summary",
}
_REVIEW_IDS_FIELD_BY_REASON = {
    "FOREFOOT": "forefoot_review_ids",
    "HEEL": "heel_review_ids",
    "INSOLE": "insole_review_ids",
}
_RISK_LABEL = {"LOW": "좋음", "MEDIUM": "보통", "HIGH": "주의"}

# These topics cannot be valid evidence for FOREFOOT/HEEL/INSOLE fit. They are
# rejected even when they appear in an untrusted review body; this closes the
# exact failure mode where fashion text or an emotional body metaphor was
# rewritten as a shoe-fit claim.
_REASON_OUT_OF_SCOPE_SUMMARY_TERMS = (
    "바지",
    "밑단",
    "코디",
    "패션",
    "옷",
    "예뻐",
    "예쁘",
    "이뻐",
    "디자인",
    "색상",
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
)
_POINT_OUT_OF_SCOPE_SUMMARY_TERMS = (
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
)
_REVIEW_GROUNDED_CONCEPTS = (
    ("통기", "통풍", "습도", "땀", "열감", "답답"),
    ("쿠션", "푹신", "딱딱", "충격"),
    ("발목",),
    ("사이즈", "크기", "사이즈 업", "사이즈업", "사이즈 다운", "사이즈다운", "1업", "반업"),
    ("무게", "무겁", "무거"),
    ("가볍", "가벼"),
    ("마찰", "쓸", "쓸리", "쓸림", "까짐", "까졌"),
    ("압박", "눌림", "좁"),
    ("넓", "여유", "넉넉"),
    ("적응",),
    ("해결",),
    ("개선",),
    ("완화",),
)
_POINT_GROUNDED_CONCEPTS = (
    ("색감", "색상"),
    ("디자인", "실루엣"),
    ("예쁘", "예뻐", "이뻐"),
    ("청바지", "데님"),
    ("슬랙스",),
    ("스커트", "치마"),
    ("통큰 바지", "와이드 팬츠"),
    ("바지",),
    ("데일리", "일상용"),
    ("코디",),
    ("날렵",),
    ("로우", "낮은 굽", "굽은 낮"),
)
_AREA_CONCEPTS = {
    "FOREFOOT": ("발볼", "앞볼", "앞코", "앞꿈치", "발가락", "엄지"),
    "HEEL": ("뒤꿈치", "뒤축", "하중", "충격", "반발", "쿠션", "안정감", "착화감"),
    "INSOLE": ("깔창", "인솔", "통기", "통풍", "열감", "습기", "신발 내부"),
}
_CAUSAL_CLAIM_SUBJECTS = (
    "발볼",
    "앞코",
    "뒤꿈치",
    "발목",
    "깔창",
    "쿠션",
    "통기성",
    "통풍",
    "바닥",
    "무게",
)

_llm: ChatOllama | None = None
_cpu_llm: ChatOllama | None = None
_llm_lock = threading.Lock()
_ollama_semaphore = asyncio.Semaphore(settings.ollama_max_concurrency)

logger = logging.getLogger(__name__)


class ShoeFitCommentError(RuntimeError):
    pass


def _display_shoe_name(shoe_name: str) -> str:
    return _DISPLAY_MODEL_CODE_SUFFIX_PATTERN.sub("", shoe_name).strip() or shoe_name


@dataclass(frozen=True)
class ReasonFactsForPrompt:
    reason_type: str  # "FOREFOOT" | "HEEL" | "INSOLE"
    title: str
    risk_level: str
    review_ids: list[int]
    review_texts: list[str]

    def __post_init__(self) -> None:
        if self.reason_type not in REASON_TYPES:
            raise ValueError(f"Unknown reason_type: {self.reason_type}")
        if self.risk_level not in _RISK_LABEL:
            raise ValueError(f"Unknown risk_level: {self.risk_level}")
        if len(self.review_ids) != len(self.review_texts):
            raise ValueError("review_ids and review_texts must have the same length")
        if len(self.review_ids) > 3 or len(self.review_ids) != len(set(self.review_ids)):
            raise ValueError("review_ids must be unique and contain at most three ids")
        if any(review_id < 1 for review_id in self.review_ids):
            raise ValueError("review_ids must contain positive ids")


def prepare_grounded_reasons(
    reasons: list[ReasonFactsForPrompt],
) -> list[ReasonFactsForPrompt]:
    """Canonicalize legacy evidence and expose only reason-specific clauses.

    The first recommendation stage persists review IDs, not BGE's selected
    sentence. A later Server response therefore contains each full review body.
    This boundary reconstructs a conservative excerpt and removes duplicate
    bodies before any untrusted text reaches Ollama.
    """

    reason_types = [reason.reason_type for reason in reasons]
    if len(reason_types) != len(set(reason_types)) or set(reason_types) != set(REASON_TYPES):
        raise ShoeFitCommentError(
            "Summary facts must contain FOREFOOT, HEEL, and INSOLE exactly once."
        )

    prepared: list[ReasonFactsForPrompt] = []
    for reason_type in REASON_TYPES:
        reason = next(item for item in reasons if item.reason_type == reason_type)

        # Preserve the first relevance-ranked position, while using the
        # smallest reviewId as the stable canonical ID for a duplicate body.
        body_groups: dict[str, tuple[int, int, str]] = {}
        for index, (review_id, review_text) in enumerate(
            zip(reason.review_ids, reason.review_texts, strict=True)
        ):
            body_fingerprint = normalize_review_text(review_text)
            if not body_fingerprint:
                continue
            current = body_groups.get(body_fingerprint)
            if current is None:
                body_groups[body_fingerprint] = (index, review_id, review_text)
            elif review_id < current[1]:
                body_groups[body_fingerprint] = (current[0], review_id, review_text)

        candidates = [
            ReviewCandidate(
                review_id=review_id,
                sentence=review_text,
                similarity=max(0.0, 1.0 - index * 0.01),
                source_text=review_text,
            )
            for index, review_id, review_text in sorted(
                body_groups.values(), key=lambda item: item[0]
            )
        ]
        selected = select_review_evidence(
            reason_type,
            reason.risk_level,
            candidates,
            limit=3,
        )

        prepared.append(
            ReasonFactsForPrompt(
                reason_type=reason.reason_type,
                title=reason.title,
                risk_level=reason.risk_level,
                review_ids=[candidate.review_id for candidate in selected],
                review_texts=[candidate.sentence for candidate in selected],
            )
        )
    return prepared


def prepare_point_summary_evidence(
    reasons: list[ReasonFactsForPrompt],
    characteristics: ServerShoeCharacteristics | None = None,
) -> list[PointEvidenceFact]:
    """Aggregate typed facts without losing reviewId provenance or consensus."""

    review_pairs = [
        (review_id, review_text)
        for reason_type in REASON_TYPES
        for reason in reasons
        if reason.reason_type == reason_type
        for review_id, review_text in zip(
            reason.review_ids, reason.review_texts, strict=True
        )
    ]
    try:
        return prepare_point_evidence_facts(review_pairs, characteristics)
    except ValueError as exc:
        raise ShoeFitCommentError(str(exc)) from exc


def _new_llm(num_gpu: int) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        format=_SHOE_FIT_SUMMARY_JSON_SCHEMA,
        num_gpu=num_gpu,
        keep_alive=settings.ollama_keep_alive,
        client_kwargs={"timeout": settings.ollama_request_timeout_seconds},
    )


def _get_llm(*, force_cpu: bool = False) -> ChatOllama:
    global _llm, _cpu_llm
    if force_cpu:
        if _cpu_llm is None:
            with _llm_lock:
                if _cpu_llm is None:
                    _cpu_llm = _new_llm(0)
        return _cpu_llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = _new_llm(settings.ollama_num_gpu)
    return _llm


async def _invoke_ollama(messages: list[BaseMessage]) -> BaseMessage:
    try:
        return await _get_llm().ainvoke(messages)
    except (httpx.HTTPError, ResponseError, RuntimeError) as exc:
        if not settings.ollama_cpu_fallback_enabled or settings.ollama_num_gpu == 0:
            raise ShoeFitCommentError(f"Ollama 요청에 실패했습니다: {exc}") from exc
        logger.warning(
            "Ollama GPU 우선 요청 실패; 동일 prompt를 CPU로 한 번 fallback합니다.",
            exc_info=True,
        )
        try:
            return await _get_llm(force_cpu=True).ainvoke(messages)
        except (httpx.HTTPError, ResponseError, RuntimeError) as cpu_exc:
            raise ShoeFitCommentError(
                f"Ollama GPU/CPU 요청이 모두 실패했습니다: {cpu_exc}"
            ) from cpu_exc


async def ollama_runtime_status() -> dict[str, object]:
    """Read-only Ollama GPU preflight. It never loads a model or triggers inference."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/ps"
    try:
        async with httpx.AsyncClient(
            timeout=min(settings.ollama_request_timeout_seconds, 10.0),
            trust_env=False,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "reachable": False,
            "model": settings.ollama_model,
            "gpuInUse": None,
            "detail": str(exc),
        }
    models = payload.get("models", []) if isinstance(payload, dict) else []
    loaded = next(
        (
            model
            for model in models
            if isinstance(model, dict)
            and str(model.get("name", "")).split(":")[0]
            == settings.ollama_model.split(":")[0]
        ),
        None,
    )
    size_vram = loaded.get("size_vram") if loaded else None
    return {
        "reachable": True,
        "model": settings.ollama_model,
        "gpuInUse": bool(size_vram) if size_vram is not None else None,
        "sizeVram": size_vram,
    }


async def prewarm_ollama_model() -> bool:
    """Load the configured model without running summary inference.

    Ollama treats an empty ``/api/generate`` prompt as a load-only request.
    The operation is opt-in because keeping Qwen and BGE-M3 resident together
    can exhaust an 8 GB GPU during a recommendation batch.
    """

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    try:
        async with httpx.AsyncClient(
            timeout=settings.ollama_request_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.post(
                url,
                json={
                    "model": settings.ollama_model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                },
            )
            response.raise_for_status()
    except (httpx.HTTPError, ValueError):
        logger.warning(
            "Ollama model prewarm failed. model=%s",
            settings.ollama_model,
            exc_info=True,
        )
        return False
    logger.info(
        "Ollama model prewarm completed. model=%s keep_alive=%s",
        settings.ollama_model,
        settings.ollama_keep_alive,
    )
    return True


def _format_reason_block(reason: ReasonFactsForPrompt) -> str:
    if reason.review_texts:
        review_lines = "\n".join(
            f'  {i + 1}. reviewId={review_id}: "{text}"'
            for i, (review_id, text) in enumerate(
                zip(reason.review_ids, reason.review_texts, strict=True)
            )
        )
    else:
        review_lines = f"  {_NO_REVIEW_NOTE}"

    label = _REASON_LABEL[reason.reason_type]
    return (
        f"- 항목: {label} ({reason.reason_type})\n"
        f"- 제목: {reason.title}\n"
        f"- 위험도: {reason.risk_level}\n"
        f"- 참고 리뷰 문장:\n{review_lines}"
    )


def _format_point_evidence(point_evidence: list[PointEvidenceFact]) -> str:
    return format_point_evidence_facts(point_evidence)


def _format_personal_measurement_block(
    evidence: ServerPersonalFitEvidence | None,
) -> str:
    lines: list[str] = []
    for reason_type in REASON_TYPES:
        sentence = _personal_measurement_sentence(reason_type, evidence)
        lines.append(
            f"- {reason_type}: {sentence or '제공된 실측 근거 없음'}"
        )
    return "\n".join(lines)


def _build_user_prompt(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> str:
    reason_by_type = {reason.reason_type: reason for reason in reasons}
    return USER_PROMPT_TEMPLATE.format(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        personal_measurement_block=_format_personal_measurement_block(
            personal_fit_evidence
        ),
        forefoot_block=_format_reason_block(reason_by_type["FOREFOOT"]),
        heel_block=_format_reason_block(reason_by_type["HEEL"]),
        insole_block=_format_reason_block(reason_by_type["INSOLE"]),
        point_evidence_block=_format_point_evidence(point_evidence),
    )


def _parse_llm_json(raw_text: object) -> dict:
    """정상적인 JSON이면 그대로 파싱하고, 실패하면 텍스트 안에서 JSON 블록만 추출해 재파싱한다."""
    if not isinstance(raw_text, str):
        raise ShoeFitCommentError(
            "Ollama 응답 content가 문자열이 아니어서 JSON을 검증할 수 없습니다."
        )
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_PATTERN.search(text)
    if not match:
        raise ShoeFitCommentError(f"Ollama 응답에서 JSON을 찾을 수 없습니다: {text[:200]}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ShoeFitCommentError(f"Ollama 응답 JSON 파싱에 실패했습니다: {text[:200]}") from exc


def _validate_review_selection(
    summary: ShoeFitSummary, reasons: list[ReasonFactsForPrompt]
) -> None:
    allowed = {reason.reason_type: set(reason.review_ids) for reason in reasons}
    selected = {
        "FOREFOOT": summary.forefoot_review_ids,
        "HEEL": summary.heel_review_ids,
        "INSOLE": summary.insole_review_ids,
    }
    for reason_type, review_ids in selected.items():
        if not set(review_ids).issubset(allowed[reason_type]):
            raise ShoeFitCommentError(
                "Ollama selected a reviewId outside the deterministic shoe/reason set "
                f"(reasonType={reason_type})."
            )


def _ensure_review_selection_minimum(
    summary: ShoeFitSummary, reasons: list[ReasonFactsForPrompt]
) -> ShoeFitSummary:
    """Lock output IDs to the deterministic domain/risk evidence selection."""

    updates: dict[str, list[int]] = {}
    for reason in reasons:
        field_name = _REVIEW_IDS_FIELD_BY_REASON[reason.reason_type]
        updates[field_name] = list(reason.review_ids)
    return summary.model_copy(update=updates)


def _ensure_point_evidence_reviews_selected(
    summary: ShoeFitSummary,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> ShoeFitSummary:
    """Keep product point evidence separate from per-reason review evidence."""

    del point_evidence
    return _ensure_review_selection_minimum(summary, reasons)


def _contains_out_of_scope_content(text: str, blocked_terms: tuple[str, ...]) -> bool:
    normalized = normalize_review_text(text)
    return any(term in normalized for term in blocked_terms)


def _uses_unsupported_concept(
    text: str,
    source_text: str,
    *,
    additional_concepts: tuple[tuple[str, ...], ...] = (),
) -> bool:
    normalized_text = normalize_review_text(text)
    normalized_source = normalize_review_text(source_text)
    if "여유가 부족" in normalized_text and _contains_any(
        normalized_source, ("작", "좁", "끼", "눌", "답답", "타이트")
    ):
        # Direction-preserving paraphrase used by the deterministic renderer.
        normalized_source = f"{normalized_source} 여유가 부족"
    if re.search(r"발볼.{0,8}넓", normalized_text) and (
        "발볼러" in normalized_source
        or re.search(r"발볼.{0,10}있(?:는|으면|다면|어서)", normalized_source)
    ):
        normalized_source = f"{normalized_source} 발볼 넓"
    if re.search(r"발등.{0,8}높", normalized_text) and re.search(
        r"발등.{0,10}있(?:는|으면|다면|어서)", normalized_source
    ):
        normalized_source = f"{normalized_source} 발등 높"

    for terms in (*_REVIEW_GROUNDED_CONCEPTS, *additional_concepts):
        if any(term in normalized_text for term in terms) and not any(
            term in normalized_source for term in terms
        ):
            return True
    for terms in _AREA_CONCEPTS.values():
        for term in terms:
            if term in normalized_text and term not in normalized_source:
                return True
    for subject in _CAUSAL_CLAIM_SUBJECTS:
        output_causal_forms = (
            f"{subject} 때문",
            f"{subject}때문",
            f"{subject}으로 인해",
            f"{subject}로 인해",
        )
        if any(form in normalized_text for form in output_causal_forms) and not any(
            form in normalized_source for form in output_causal_forms
        ):
            return True
    return False


_POINT_REPORT_STYLE_PATTERNS = (
    "후기를 참고",
    "후기에 따르면",
    "후기에서는",
    "후기상",
    "리뷰에 따르면",
    "리뷰에서는",
    "리뷰상",
    "구매평",
    "착화평",
    "평가에 따르면",
    "확인하는 것이 좋",
    "체크하는 것이 좋",
    "체크해 보",
    "참고할 만",
    "라는 의견이 있고",
    "라는 의견이 있습니다",
    "라는 후기가 있고",
    "라는 후기가 있습니다",
    "의견이 있습니다",
    "후기가 있습니다",
    "한편",
    "반면 일부 사용자",
    "일부 사용자는",
    "발볼 리뷰에서는",
    "뒤꿈치 리뷰에서는",
    "깔창 리뷰에서는",
)
_POINT_HEDGE_MARKERS = (
    "일부 착화",
    "한 착화",
    "한 사례",
    "개인차",
    "착화에 따라",
    "느껴질 수",
    "생길 수",
)
_POINT_MIXED_SCOPE_MARKERS = (
    "개인차",
    "엇갈",
    "서로 다르",
    "착화에 따라",
    "착화 경험에 따라",
)
_POINT_UNIVERSAL_CLAIM_TERMS = (
    "누구나",
    "대부분",
    "많은 구매자",
    "많은 사용자",
    "많은 사람",
    "구매자 대부분",
    "사용자 대부분",
    "항상",
    "모든 사람",
    "모든 사용자",
    "무조건",
)
_POINT_UNIVERSAL_CLAIM_PATTERN = re.compile(
    r"(?:모든|거의\s*모든|다수의|많은|대부분의)\s*"
    r"(?:구매자|사용자|사람)|(?:구매자|사용자|사람)\s*(?:모두|대부분|전부)"
)
_POINT_UNTRACKED_ATTRIBUTE_TERMS = (
    "내구성",
    "내구력이",
    "방수",
    "방풍",
    "접지력",
    "미끄럼 방지",
    "보온성",
    "가격 대비",
    "가성비",
    "천연가죽",
    "가죽 소재",
    "소재",
    "레더",
    "스웨이드",
    "캔버스",
    "니트",
    "메시",
    "코팅",
    "발수",
    "냄새",
    "변색",
    "찢어",
)
_POINT_EXAGGERATION_TERMS = (
    "최고",
    "최상",
    "압도적",
    "완벽",
    "극강",
    "구름 위",
    "탁월",
)

_SIZE_THREE_DIGIT_PATTERN = re.compile(r"(?<!\d)(?:2\d{2}|3\d{2})(?!\d)")
_SIZE_EXACT_STEP_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*업")
_SIZE_UNIT_STEP_PATTERN = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*"
    r"(mm|cm|미리|밀리미터|센티|센티미터)\s*(?:정도\s*)?"
    r"(?:사이즈\s*)?업"
)
_SIZE_AVOIDANCE_CLAIM_PATTERN = re.compile(
    r"(?:비추천|비권장|추천드리지\s*않|권하지\s*않|권하고\s*싶지\s*않|"
    r"피하(?:는|는\s*게|는\s*것이)|하지\s*않는\s*게|삼가|"
    r"선택하면\s*안|적절하지\s*않|(?:권장|추천)\s*대상(?:이)?\s*(?:아니|아닙))"
)


def _comfort_polarity(text: str) -> tuple[bool, bool]:
    normalized = normalize_review_text(text)
    reduced_burden = bool(
        re.search(r"부담(?:이|은)?\s*(?:적|크지 않|덜)", normalized)
    )
    discomfort = _contains_any(
        normalized, ("불편", "아프", "피로", "아쉬")
    ) or ("부담" in normalized and not reduced_burden)
    comfort = (
        _has_unnegated_term(normalized, ("편안", "편하게", "편한 편"))
        and "불편" not in normalized
        and not discomfort
    ) or reduced_burden
    return comfort, discomfort


def _claim_uses_uncited_signal(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    normalized = normalize_review_text(text)

    def cited(subject: str, *values: str) -> bool:
        comparable_subjects = (
            {"WIDTH_SPACE", "LAB_WIDTH_SPACE"}
            if subject == "WIDTH_SPACE"
            else (
                {"TOEBOX_SPACE", "LAB_TOEBOX_SPACE"}
                if subject == "TOEBOX_SPACE"
                else {subject}
            )
        )
        return any(
            fact.signal.subject in comparable_subjects
            and (not values or fact.signal.value in values)
            and fact.mode != "CONFLICT"
            for fact in cited_facts
        )

    checks: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
        (("한 사이즈 업", "1업"), "SIZE_OPTION", ("FULL_UP",)),
        (("반 사이즈 업", "반업"), "SIZE_OPTION", ("HALF_UP",)),
        (("정사이즈",), "SIZE_OPTION", ("TRUE_SIZE",)),
        (("전체적으로 딱 맞", "딱 맞는 핏"), "OVERALL_FIT", ("EXACT",)),
        (("슬림", "좁은 편", "밀착되는 핏"), "WIDTH_SPACE", ("TIGHT",)),
        (("여유로운 편", "넉넉한 편"), "WIDTH_SPACE", ("ROOMY",)),
        (
            ("안정적으로 잡", "발목을 잡", "발을 잘 잡", "잘 잡아주는 편"),
            "HEEL_HOLD",
            ("STABLE",),
        ),
        (("헐렁", "들뜨"), "HEEL_HOLD", ("LOOSE",)),
        (("뒤꿈치 주변은 편안", "뒤꿈치는 편안"), "HEEL_COMFORT", ("COMFORTABLE",)),
        (("뒤꿈치 주변은 마찰", "뒤꿈치는 쓸", "뒤꿈치가 쓸"), "HEEL_COMFORT", ("RUBBING",)),
        (("뒤꿈치 주변은 다소 불편", "뒤꿈치는 불편"), "HEEL_COMFORT", ("DISCOMFORT",)),
        (("바닥이 얇", "바닥은 얇", "깔창이 얇", "깔창은 얇"), "SOLE_THICKNESS", ("THIN",)),
        (("바닥이 두껍", "바닥은 두껍", "깔창이 두껍", "깔창은 두껍"), "SOLE_THICKNESS", ("THICK",)),
        (("쿠션감에 만족", "쿠션감이 좋", "쿠션이 좋"), "CUSHION_QUALITY", ("POSITIVE",)),
        (("쿠션감이 아쉽", "쿠션감이 별로", "쿠션이 부족"), "CUSHION_QUALITY", ("NEGATIVE",)),
        (
            (
                "쿠션감은 높은",
                "쿠션감이 높은",
                "쿠션은 부드럽",
                "쿠션이 부드럽",
                "부드럽게 느껴",
                "푹신",
                "폭신",
            ),
            "CUSHION_SOFTNESS",
            ("SOFT",),
        ),
        (
            ("쿠션감은 낮", "쿠션감이 낮", "쿠션은 단단", "쿠션이 단단", "쿠션은 딱딱", "쿠션이 딱딱", "바닥감"),
            "CUSHION_SOFTNESS",
            ("FIRM",),
        ),
        (("무게감", "무겁"), "WEIGHT", ("HEAVY",)),
        (("가볍",), "WEIGHT", ("LIGHT",)),
        (("색감", "디자인"), "APPEARANCE", ("POSITIVE", "MENTIONED")),
        (("청바지", "데님"), "OUTFIT", ("JEANS",)),
        (("슬랙스",), "OUTFIT", ("SLACKS",)),
        (("스커트", "치마"), "OUTFIT", ("SKIRT",)),
        (("통큰 바지", "와이드 팬츠"), "OUTFIT", ("WIDE_PANTS",)),
        (("데일리", "일상용"), "OUTFIT", ("DAILY",)),
        (("날렵",), "SILHOUETTE", ("SLIM",)),
        (("로우 프로파일", "낮은 실루엣"), "PROFILE", ("LOW",)),
    )
    for terms, subject, values in checks:
        if any(term in normalized for term in terms) and not cited(subject, *values):
            return True

    if re.search(r"(?:발|발목|뒤꿈치).{0,10}(?:잘\s*)?잡아?주", normalized) and not cited(
        "HEEL_HOLD", "STABLE"
    ):
        return True
    if re.search(r"쿠션.{0,24}(?:단단|딱딱|바닥감)", normalized) and not cited(
        "CUSHION_SOFTNESS", "FIRM"
    ):
        return True
    if (
        _contains_any(normalized, ("전반적인 착화감", "전반적으로 편안", "착화 시 편안"))
        and _comfort_polarity(normalized)[0]
        and not cited("WEAR_COMFORT", "COMFORTABLE")
    ):
        return True
    if (
        "착화감" in normalized
        and _has_unnegated_term(
            normalized, ("만족", "좋", "편안", "편하게", "편한 편")
        )
        and not _comfort_polarity(normalized)[1]
        and not cited("WEAR_COMFORT", "COMFORTABLE")
    ):
        return True

    if (
        ("사이즈 업" in normalized or "사이즈업" in normalized)
        and "반 사이즈 업" not in normalized
        and "한 사이즈 업" not in normalized
        and not cited("SIZE_OPTION", "GENERIC_UP")
    ):
        return True
    if "앞코 공간" in normalized or "토박스 공간" in normalized:
        if _contains_any(normalized, ("좁", "타이트", "밀착", "낮은 편")) and not cited(
            "TOEBOX_SPACE", "TIGHT"
        ):
            return True
        if _contains_any(normalized, ("여유", "넉넉", "넓", "높은 편")) and not cited(
            "TOEBOX_SPACE", "ROOMY"
        ):
            return True
    if (
        any(term in normalized for term in ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신"))
        and not cited("SIZE_OPTION", "DOWN")
    ):
        return True
    duration_terms = ("오래", "장시간", "종일", "하루종일")
    if _contains_any(normalized, duration_terms):
        describes_comfort, describes_discomfort = _comfort_polarity(normalized)
        if describes_comfort and not cited(
            "LONG_WEAR_COMFORT", "COMFORTABLE"
        ):
            return True
        if describes_discomfort and not cited(
            "LONG_WEAR_COMFORT", "DISCOMFORT"
        ):
            return True
    if "runrepeat" in normalized and not any(
        fact.source == "RUNREPEAT" for fact in cited_facts
    ):
        return True
    return False


def _single_facts_are_safely_scoped(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    single_facts = [
        fact
        for fact in cited_facts
        if fact.source == "MUSINSA" and fact.mode == "SINGLE"
    ]
    if not single_facts:
        return True
    normalized = normalize_review_text(text)
    if any(term in normalized for term in _POINT_UNIVERSAL_CLAIM_TERMS):
        return False
    prefix_markers = ("일부 착화", "한 착화", "한 사례", "착화에 따라")
    prefix = normalized[:40]
    return any(marker in prefix for marker in prefix_markers) or any(
        marker in normalized for marker in ("느껴질 수", "생길 수", "맞을 수")
    )


def _mixed_facts_are_safely_scoped(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    if not any(
        fact.source == "MUSINSA" and fact.mode == "MIXED"
        for fact in cited_facts
    ):
        return True
    normalized = normalize_review_text(text)
    return any(marker in normalized for marker in _POINT_MIXED_SCOPE_MARKERS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _lab_fact_is_accurately_expressed(
    text: str, fact: PointEvidenceFact
) -> bool:
    """Allow a LAB citation only for its exact characteristic and level direction."""

    normalized = normalize_review_text(text)
    if _contains_any(
        normalized,
        (
            "여름",
            "시원",
            "관절",
            "부상",
            "보호",
            "빨리 달",
            "속도가",
            "기록 향상",
        ),
    ):
        return False
    subject = fact.signal.subject
    value = fact.signal.value
    specifications: dict[
        str,
        tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]],
    ] = {
        "CUSHION_SOFTNESS": (
            ("쿠션",),
            {
                "SOFT": ("높은 편", "높게", "부드럽", "부드러", "푹신"),
                "BALANCED": ("보통", "중간", "균형"),
                "FIRM": ("낮은 편", "낮게", "단단", "딱딱", "바닥감"),
            },
            ("높은 편", "높게", "부드럽", "부드러", "푹신", "보통", "중간", "균형", "낮은 편", "낮게", "단단", "딱딱", "바닥감"),
        ),
        "LAB_WIDTH_SPACE": (
            ("발볼 공간", "발볼 너비"),
            {
                "TIGHT": ("낮은 편", "좁", "타이트", "밀착"),
                "BALANCED": ("보통", "중간", "균형"),
                "ROOMY": ("높은 편", "넓", "여유", "넉넉"),
            },
            ("낮은 편", "좁", "타이트", "밀착", "보통", "중간", "균형", "높은 편", "넓", "여유", "넉넉"),
        ),
        "LAB_TOEBOX_SPACE": (
            ("앞코 공간", "토박스 공간"),
            {
                "TIGHT": ("낮은 편", "좁", "타이트", "밀착"),
                "BALANCED": ("보통", "중간", "균형"),
                "ROOMY": ("높은 편", "넓", "여유", "넉넉"),
            },
            ("낮은 편", "좁", "타이트", "밀착", "보통", "중간", "균형", "높은 편", "넓", "여유", "넉넉"),
        ),
        "HEEL_STRUCTURE": (
            ("뒤꿈치 구조", "힐 카운터", "구조 강성"),
            {
                "FLEXIBLE": ("낮은 편", "유연"),
                "BALANCED": ("보통", "중간", "균형"),
                "FIRM": ("높은 편", "단단", "강한 편"),
            },
            ("낮은 편", "유연", "보통", "중간", "균형", "높은 편", "단단", "강한 편"),
        ),
        "BREATHABILITY": (
            ("통기성", "통풍"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
        "SHOCK_ABSORPTION": (
            ("충격 완화", "충격 흡수"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
        "ENERGY_RETURN": (
            ("반발력", "에너지 리턴"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
    }
    specification = specifications.get(subject)
    if specification is None:
        return False
    subject_terms, value_terms, all_direction_terms = specification
    allowed_terms = value_terms.get(value)
    if allowed_terms is None or not _contains_any(normalized, subject_terms):
        return False
    if not _has_unnegated_term(normalized, allowed_terms) or _has_negated_term(
        normalized, allowed_terms
    ):
        return False
    opposite_terms = tuple(
        term for term in all_direction_terms if term not in allowed_terms
    )
    return not _contains_any(normalized, opposite_terms)


def _review_fact_is_expressed(text: str, fact: PointEvidenceFact) -> bool:
    normalized = normalize_review_text(text)
    expressions: dict[tuple[str, str], tuple[str, ...]] = {
        ("SIZE_OPTION", "FULL_UP"): ("한 사이즈 업", "1업"),
        ("SIZE_OPTION", "HALF_UP"): ("반 사이즈 업", "반업", "0.5업"),
        ("SIZE_OPTION", "TRUE_SIZE"): ("정사이즈",),
        ("SIZE_OPTION", "GENERIC_UP"): ("사이즈 업", "사이즈업", "사이즈를 올"),
        ("SIZE_OPTION", "DOWN"): ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신"),
        ("FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP"): ("발볼", "발등"),
        ("WIDTH_SPACE", "TIGHT"): ("좁", "타이트", "밀착", "압박", "슬림"),
        ("WIDTH_SPACE", "ROOMY"): ("넓", "여유", "넉넉"),
        ("TOEBOX_SPACE", "TIGHT"): ("좁", "타이트", "밀착", "끼", "여유가 부족"),
        ("TOEBOX_SPACE", "ROOMY"): ("넓", "여유", "넉넉"),
        ("OVERALL_FIT", "EXACT"): ("딱 맞", "밀착"),
        ("HEEL_HOLD", "STABLE"): ("안정적으로 잡", "발목을 잡", "받쳐", "지지"),
        ("HEEL_HOLD", "LOOSE"): ("헐렁", "들뜨", "벗겨", "미끄러"),
        ("HEEL_COMFORT", "RUBBING"): ("쓸", "까", "마찰"),
        ("HEEL_COMFORT", "DISCOMFORT"): ("불편", "아프", "통증"),
        ("HEEL_COMFORT", "COMFORTABLE"): ("편하", "편안", "폭신", "부드럽"),
        ("CUSHION_SOFTNESS", "SOFT"): ("푹신", "폭신", "부드럽"),
        ("CUSHION_SOFTNESS", "FIRM"): ("딱딱", "단단", "쿠션감이 낮", "바닥감"),
        ("SOLE_THICKNESS", "THIN"): ("바닥이 얇", "바닥은 얇", "깔창이 얇", "깔창은 얇"),
        ("SOLE_THICKNESS", "THICK"): ("바닥이 두껍", "바닥은 두껍", "깔창이 두껍", "깔창은 두껍"),
        ("CUSHION_QUALITY", "POSITIVE"): ("쿠션감에 만족", "쿠션감이 좋", "쿠션이 좋"),
        ("CUSHION_QUALITY", "NEGATIVE"): ("쿠션감이 아쉽", "쿠션감이 별로", "쿠션이 부족"),
        ("LONG_WEAR_COMFORT", "DISCOMFORT"): ("오래 서", "장시간", "종일", "발바닥 부담"),
        ("LONG_WEAR_COMFORT", "COMFORTABLE"): (
            "오래 신어도 편",
            "장시간 편",
            "장시간 착용",
            "오래 착용",
            "종일 편",
        ),
        ("WEAR_COMFORT", "COMFORTABLE"): ("착화감", "편하게 느껴", "편안하게 느껴"),
        ("WEAR_COMFORT", "DISCOMFORT"): ("착화감", "불편하게 느껴"),
        ("WEIGHT", "HEAVY"): ("무게감", "무겁"),
        ("WEIGHT", "LIGHT"): ("가볍",),
        ("APPEARANCE", "POSITIVE"): ("색감", "디자인", "예쁘", "만족"),
        ("APPEARANCE", "MENTIONED"): ("색감", "디자인", "색상"),
        ("SILHOUETTE", "SLIM"): ("날렵", "슬림"),
        ("SILHOUETTE", "ROUNDED"): ("둥글",),
        ("PROFILE", "LOW"): ("로우", "낮은 실루엣", "굽이 낮"),
        ("OUTFIT", "JEANS"): ("청바지", "데님"),
        ("OUTFIT", "SLACKS"): ("슬랙스",),
        ("OUTFIT", "SKIRT"): ("스커트", "치마"),
        ("OUTFIT", "WIDE_PANTS"): ("통큰 바지", "와이드 팬츠"),
        ("OUTFIT", "DAILY"): ("데일리", "일상용"),
        ("OUTFIT", "GENERAL"): ("코디", "잘 어울"),
        ("PRODUCT_DETAIL", "MENTIONED"): ("소재", "굽", "착화감"),
    }
    terms = expressions.get((fact.signal.subject, fact.signal.value))
    return bool(terms and _contains_any(normalized, terms))


def _review_fact_stance_is_consistent(
    text: str, fact: PointEvidenceFact
) -> bool:
    """Block polarity flips while preserving each typed concept's scope."""

    normalized = normalize_review_text(text)
    subject = fact.signal.subject
    value = fact.signal.value

    expression_roots: dict[tuple[str, str], tuple[str, ...]] = {
        ("WIDTH_SPACE", "TIGHT"): ("좁", "타이트", "밀착", "압박", "슬림"),
        ("WIDTH_SPACE", "ROOMY"): ("넓", "여유", "넉넉"),
        ("TOEBOX_SPACE", "TIGHT"): ("좁", "타이트", "끼"),
        ("TOEBOX_SPACE", "ROOMY"): ("넓", "여유", "넉넉"),
        ("OVERALL_FIT", "EXACT"): ("딱 맞", "밀착"),
        ("HEEL_HOLD", "STABLE"): ("안정", "잡", "받쳐", "지지"),
        ("HEEL_HOLD", "LOOSE"): ("헐렁", "들뜨", "벗겨"),
        ("HEEL_COMFORT", "RUBBING"): ("쓸", "마찰", "까"),
        ("HEEL_COMFORT", "DISCOMFORT"): ("불편", "아프", "통증"),
        ("HEEL_COMFORT", "COMFORTABLE"): ("편하", "편해", "편안", "부드럽"),
        ("CUSHION_SOFTNESS", "SOFT"): ("푹신", "폭신", "부드럽"),
        ("CUSHION_SOFTNESS", "FIRM"): ("딱딱", "단단", "바닥감"),
        ("CUSHION_QUALITY", "POSITIVE"): ("만족", "좋", "편안", "편하"),
        ("CUSHION_QUALITY", "NEGATIVE"): ("아쉽", "별로", "부족", "불편"),
        ("SOLE_THICKNESS", "THIN"): ("얇",),
        ("SOLE_THICKNESS", "THICK"): ("두껍",),
        ("WEIGHT", "HEAVY"): ("무겁", "무거", "무게감"),
        ("WEIGHT", "LIGHT"): ("가볍", "가벼"),
        ("APPEARANCE", "POSITIVE"): ("예쁘", "예뻐", "이뻐", "좋", "깔끔", "만족"),
        ("SILHOUETTE", "SLIM"): ("날렵", "슬림"),
        ("SILHOUETTE", "ROUNDED"): ("둥글",),
        ("PROFILE", "LOW"): ("로우", "낮은 실루엣", "굽이 낮"),
    }
    roots = expression_roots.get((subject, value), ())
    if roots and _has_negated_term(normalized, roots):
        return False

    if subject == "SIZE_OPTION":
        recommendation_terms = (
            "고려할 만",
            "추천",
            "잘 맞",
            "편하게 맞",
            "선택할 만",
        )
        failure_terms = (
            "여전히 작",
            "그래도 작",
            "작게 느껴",
            "여유가 부족",
            "크게 느껴",
            "너무 크",
            "헐렁",
            "맞지 않",
            "불편",
            "실패",
        )
        recommends = _contains_any(normalized, recommendation_terms)
        describes_failure = _contains_any(normalized, failure_terms)
        if fact.signal.stance == "NEGATIVE":
            return describes_failure and not recommends
        if fact.signal.stance == "POSITIVE":
            if _SIZE_AVOIDANCE_CLAIM_PATTERN.search(normalized) or "피해야" in normalized:
                return False
            return not describes_failure and not _has_negated_term(
                normalized, ("추천", "고려", "선택", "맞", "편")
            )

    if subject in {"LONG_WEAR_COMFORT", "WEAR_COMFORT"}:
        describes_comfort, describes_discomfort = _comfort_polarity(normalized)
        if value == "COMFORTABLE":
            return describes_comfort and not describes_discomfort
        if value == "DISCOMFORT":
            return describes_discomfort and not describes_comfort

    negations_by_signal: dict[tuple[str, str], tuple[str, ...]] = {
        ("WIDTH_SPACE", "TIGHT"): ("좁지 않", "타이트하지 않", "압박이 없"),
        ("WIDTH_SPACE", "ROOMY"): ("넓지 않", "여유가 없", "여유 없", "넉넉하지 않"),
        ("TOEBOX_SPACE", "TIGHT"): ("좁지 않", "타이트하지 않", "끼지 않"),
        ("TOEBOX_SPACE", "ROOMY"): ("넓지 않", "여유가 없", "넉넉하지 않"),
        ("HEEL_HOLD", "STABLE"): ("안정적이지 않", "잡아주지 않"),
        ("HEEL_HOLD", "LOOSE"): ("헐렁하지 않", "들뜨지 않", "벗겨지지 않"),
        ("HEEL_COMFORT", "RUBBING"): ("쓸리지 않", "마찰이 없", "까짐이 없"),
        ("HEEL_COMFORT", "DISCOMFORT"): ("불편하지 않", "아프지 않", "통증이 없"),
        ("HEEL_COMFORT", "COMFORTABLE"): ("편하지 않", "불편"),
        ("CUSHION_SOFTNESS", "SOFT"): ("푹신하지 않", "부드럽지 않"),
        ("CUSHION_SOFTNESS", "FIRM"): ("딱딱하지 않", "단단하지 않", "바닥감이 없"),
        ("SOLE_THICKNESS", "THIN"): ("얇지 않",),
        ("SOLE_THICKNESS", "THICK"): ("두껍지 않",),
        ("WEIGHT", "HEAVY"): ("무겁지 않", "무게감이 없"),
        ("WEIGHT", "LIGHT"): ("가볍지 않",),
        ("APPEARANCE", "POSITIVE"): (
            "예쁘지 않",
            "이쁘지 않",
            "디자인이 별로",
            "디자인은 별로",
            "마음에 들지 않",
            "만족스럽지 않",
            "만족도가 낮",
            "만족도는 낮",
            "좋지 않",
            "깔끔하지 않",
        ),
        ("SILHOUETTE", "SLIM"): ("날렵하지 않", "슬림하지 않"),
        ("SILHOUETTE", "ROUNDED"): ("둥글지 않",),
        ("PROFILE", "LOW"): ("로우하지 않", "낮지 않"),
    }
    if _contains_any(normalized, negations_by_signal.get((subject, value), ())):
        return False
    if subject == "OUTFIT" and fact.signal.stance == "POSITIVE":
        if _contains_any(
            normalized,
            (
                "어울리지 않",
                "안 어울",
                "코디하기 어렵",
                "코디하기 어려",
                "활용하기 어렵",
                "활용하기 어려",
                "매치하기 어렵",
                "매치하기 어려",
                "어울리기 어렵",
                "어울리기 어려",
            ),
        ) or _has_negated_term(normalized, ("어울", "좋", "쉽")):
            return False
    if subject == "HEEL_COMFORT" and value == "COMFORTABLE" and _contains_any(
        normalized, ("통증", "아프", "까짐", "쓸림", "마찰")
    ):
        return False
    return True


def _fact_is_expressed(text: str, fact: PointEvidenceFact) -> bool:
    if fact.source == "RUNREPEAT" or fact.mode == "LAB":
        return _lab_fact_is_accurately_expressed(text, fact)
    return _review_fact_is_expressed(
        text, fact
    ) and _review_fact_stance_is_consistent(text, fact)


_LOCAL_CONTEXT_GROUP_BY_SUBJECT = {
    "SIZE_OPTION": "SIZE",
    "OVERALL_FIT": "SIZE",
    "FOOT_CONDITION": "WIDTH",
    "WIDTH_SPACE": "WIDTH",
    "TOEBOX_SPACE": "TOEBOX",
    "HEEL_HOLD": "HEEL",
    "HEEL_COMFORT": "HEEL",
    "CUSHION_SOFTNESS": "CUSHION",
    "CUSHION_QUALITY": "CUSHION",
    "SOLE_THICKNESS": "SOLE",
    "LONG_WEAR_COMFORT": "LONG_WEAR",
    "WEAR_COMFORT": "WEAR",
    "WEIGHT": "WEIGHT",
    "APPEARANCE": "APPEARANCE",
    "SILHOUETTE": "APPEARANCE",
    "PROFILE": "APPEARANCE",
    "OUTFIT": "STYLING",
}


def _review_facts_are_subject_locally_expressed(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Bind each review predicate to its own product attribute span.

    A global bag of words can otherwise validate a sentence that swaps two
    grounded predicates (for example, narrow heel + stable forefoot).  The
    source classifier and the output validator therefore share the same local
    subject boundaries.
    """

    normalized = normalize_review_text(text)

    def heel_anchor_contexts() -> tuple[str, ...]:
        markers = ("뒤꿈치", "뒷꿈치", "뒤축", "발목")
        occurrences = sorted(
            {
                match.start()
                for marker in markers
                for match in re.finditer(re.escape(marker), normalized)
            }
        )
        return tuple(
            normalized[start : occurrences[index + 1] if index + 1 < len(occurrences) else len(normalized)]
            for index, start in enumerate(occurrences)
        )

    heel_contexts = heel_anchor_contexts()
    for fact in cited_facts:
        if fact.source != "MUSINSA":
            continue
        group = _LOCAL_CONTEXT_GROUP_BY_SUBJECT.get(fact.signal.subject)
        if group is None:
            return False
        contexts = _point_subject_contexts(normalized, group)
        if not contexts or not any(
            _fact_is_expressed(context, fact) for context in contexts
        ):
            return False
        if fact.signal.subject == "HEEL_COMFORT":
            # pointSummary exposes heel comfort as a 뒤꿈치/뒤축 property.
            # Do not let a separate 발목 predicate borrow the cited friction or
            # comfort token when two heel facts share one sentence.
            heel_comfort_contexts = tuple(
                context
                for context in heel_contexts
                if context.startswith(("뒤꿈치", "뒷꿈치", "뒤축"))
            )
            if not heel_comfort_contexts or not any(
                _fact_is_expressed(context, fact)
                for context in heel_comfort_contexts
            ):
                return False
    return True


def _lab_claim_is_exactly_scoped(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Keep a RunRepeat metric to its exact characteristic and level only."""

    lab_facts = [fact for fact in cited_facts if fact.source == "RUNREPEAT"]
    if not lab_facts:
        return True
    if len(cited_facts) != 1 or len(lab_facts) != 1:
        return False
    canonical = render_exact_lab_claim(lab_facts[0])
    return canonical is not None and normalize_review_text(text) == normalize_review_text(
        canonical
    )


def _claim_segments_are_typed(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Reject an uncited extra clause appended after an otherwise valid fact."""

    segments = [
        segment.strip()
        for segment in re.split(
            r"[,;:]\s*|(?:이며|이고|하고|이면서|하면서|하며|지만|하나|거나|는데|"
            r"그러나|반면|때문에|따라서|그래서)\s+",
            normalize_review_text(text),
        )
        if segment.strip()
    ]
    return all(
        any(_fact_is_expressed(segment, fact) for fact in cited_facts)
        for segment in segments
    )


def _styling_scope_is_supported(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Do not broaden one outfit mention into generic styling versatility."""

    normalized = normalize_review_text(text)
    broad_terms = (
        "다양한 코디",
        "다양한 스타일",
        "여러 코디",
        "여러 스타일",
        "여기저기",
        "어디에나",
        "어디든",
        "코디하기 쉽",
        "다양하게 활용",
        "폭넓게 활용",
        "활용도가 높은",
        "모든 스타일",
        "어떤 옷",
        "어떤 코디",
        "어느 옷",
    )
    if not _contains_any(normalized, broad_terms):
        return True
    outfit_values = {
        fact.signal.value
        for fact in cited_facts
        if fact.signal.subject == "OUTFIT" and fact.signal.stance == "POSITIVE"
    }
    if "GENERAL" in outfit_values:
        return True
    specific_values = outfit_values & {"JEANS", "SLACKS", "SKIRT", "WIDE_PANTS"}
    return len(specific_values) >= 2


def _appearance_scope_is_supported(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Require repeated color-specific evidence, not generic design consensus."""

    normalized = normalize_review_text(text)
    color_groups = (
        ("검정", "검은색", "블랙"),
        ("흰색", "하얀", "화이트"),
        ("갈색", "브라운"),
        ("빨강", "레드"),
        ("파랑", "블루"),
        ("초록", "그린"),
        ("베이지",),
        ("아이보리",),
        ("네이비",),
        ("회색", "그레이"),
        ("핑크",),
        ("실버",),
        ("노랑", "노란색", "옐로", "yellow"),
        ("보라", "보라색", "퍼플", "purple"),
        ("주황", "주황색", "오렌지", "orange"),
        ("버건디",),
        ("카키",),
        ("민트",),
        ("골드", "금색"),
    )
    for markers in color_groups:
        if not _contains_any(normalized, markers):
            continue
        supporting_texts = {
            normalize_review_text(evidence_text)
            for fact in cited_facts
            if fact.signal.subject == "APPEARANCE"
            for evidence_text in fact.evidence_texts
            if _contains_any(normalize_review_text(evidence_text), markers)
        }
        if len(supporting_texts) < 2:
            return False
    generic_prefixes = {
        "디자인과",
        "디자인",
        "전체적인",
        "깔끔한",
        "예쁜",
        "좋은",
        "제품의",
        "신발의",
    }
    for descriptor in re.findall(
        r"([a-z가-힣0-9-]{1,16})\s*(?:색감|색상|컬러)", normalized
    ):
        if descriptor in generic_prefixes:
            continue
        supporting_texts = {
            normalize_review_text(evidence_text)
            for fact in cited_facts
            if fact.signal.subject == "APPEARANCE"
            for evidence_text in fact.evidence_texts
            if descriptor in normalize_review_text(evidence_text)
        }
        if len(supporting_texts) < 2:
            return False
    return True


def _size_claim_is_consistent(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    normalized = normalize_review_text(text)

    unit_aliases = {
        "미리": "mm",
        "밀리미터": "mm",
        "센티": "cm",
        "센티미터": "cm",
    }
    claimed_unit_steps = {
        (value, unit_aliases.get(unit, unit))
        for value, unit in _SIZE_UNIT_STEP_PATTERN.findall(normalized)
    }
    if claimed_unit_steps:
        supported_unit_steps = {
            (value, unit_aliases.get(unit, unit))
            for fact in cited_facts
            for evidence_text in fact.evidence_texts
            for value, unit in _SIZE_UNIT_STEP_PATTERN.findall(
                normalize_review_text(evidence_text)
            )
        }
        if not claimed_unit_steps.issubset(supported_unit_steps):
            return False
    required_consensus_steps = {
        exact_step
        for fact in cited_facts
        if (exact_step := consensus_size_unit_step(fact)) is not None
    }
    if required_consensus_steps and not required_consensus_steps.issubset(
        claimed_unit_steps
    ):
        return False

    def has_size_value(*values: str) -> bool:
        return any(
            fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value in values
            and fact.mode != "CONFLICT"
            for fact in cited_facts
        )

    for raw_step in _SIZE_EXACT_STEP_PATTERN.findall(normalized):
        step = float(raw_step)
        if step == 1.0:
            if not has_size_value("FULL_UP"):
                return False
        elif step == 0.5:
            if not has_size_value("HALF_UP"):
                return False
        else:
            return False

    without_exact_units = re.sub(
        r"한\s*사이즈\s*업|반\s*사이즈\s*업|반업|사이즈업",
        "",
        normalized,
    )
    if "사이즈 업" in without_exact_units and not has_size_value("GENERIC_UP"):
        return False
    if any(
        term in normalized
        for term in ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신")
    ) and not has_size_value("DOWN"):
        return False

    claimed_sizes = _SIZE_THREE_DIGIT_PATTERN.findall(normalized)
    if len(claimed_sizes) >= 2:
        supported_order = False
        for fact in cited_facts:
            for evidence_text in fact.evidence_texts:
                source_sizes = _SIZE_THREE_DIGIT_PATTERN.findall(
                    normalize_review_text(evidence_text)
                )
                cursor = 0
                for source_size in source_sizes:
                    if cursor < len(claimed_sizes) and source_size == claimed_sizes[cursor]:
                        cursor += 1
                if cursor == len(claimed_sizes):
                    supported_order = True
                    break
            if supported_order:
                break
        if not supported_order:
            return False
    return True


def _foot_condition_scope_is_supported(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Keep wearer conditions separate and require repeated condition evidence."""

    normalized = normalize_review_text(text)
    condition_facts = [
        fact for fact in cited_facts if fact.signal.subject == "FOOT_CONDITION"
    ]
    if not condition_facts:
        return True
    evidence_texts = {
        normalize_review_text(evidence_text)
        for fact in condition_facts
        for evidence_text in fact.evidence_texts
    }
    wide_texts = {
        evidence_text
        for evidence_text in evidence_texts
        if "발볼러" in evidence_text
        or re.search(
            r"발볼.{0,10}(?:넓(?:으면|다면|은\s*(?:사람|분))|크(?:면|다면|는\s*(?:사람|분))|있(?:으면|다면|어서|는\s*(?:경우|편|사람|분)))",
            evidence_text,
        )
    }
    instep_texts = {
        evidence_text
        for evidence_text in evidence_texts
        if re.search(
            r"발등.{0,10}(?:높(?:으면|다면|은\s*(?:사람|분))|있(?:으면|다면|어서|는\s*(?:경우|편|사람|분)))",
            evidence_text,
        )
    }
    if (
        re.search(r"발볼.{0,8}(?:없|보통|넓지\s*않)", normalized)
        or re.search(r"발등.{0,8}(?:없|보통|낮|높지\s*않)", normalized)
    ):
        return False
    mentions_wide_condition = bool(
        re.search(r"발볼.{0,12}(?:넓|있는\s*편|있다면)", normalized)
    )
    mentions_instep_condition = bool(
        re.search(r"발등.{0,12}(?:높|있는\s*편|있다면)", normalized)
    )
    asserts_both = bool(
        re.search(r"발볼(?:과|·|,)?\s*발등.{0,10}(?:모두|둘\s*다|있는)", normalized)
        or re.search(r"발볼과\s*발등", normalized)
    )
    if asserts_both:
        supporting_both = {
            evidence_text
            for evidence_text in evidence_texts
            if evidence_text in wide_texts and evidence_text in instep_texts
        }
        return len(supporting_both) >= 2
    if mentions_wide_condition and len(wide_texts) < 2:
        return False
    if mentions_instep_condition and len(instep_texts) < 2:
        return False
    return True


def _cross_evidence_causality_is_supported(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    """Require repeated explicit source causality before joining typed facts."""

    normalized = normalize_review_text(text)
    if not re.search(
        r"(?:바닥|밑창|깔창).{0,10}얇.{0,8}(?:아서|어서|이라|때문|탓).{0,24}"
        r"(?:오래|장시간|종일|부담|피로|아프)",
        normalized,
    ):
        return True
    sole_facts = [
        fact
        for fact in cited_facts
        if fact.signal.subject == "SOLE_THICKNESS" and fact.signal.value == "THIN"
    ]
    wear_facts = [
        fact
        for fact in cited_facts
        if fact.signal.subject == "LONG_WEAR_COMFORT"
        and fact.signal.value == "DISCOMFORT"
    ]
    if not sole_facts or not wear_facts:
        return False
    explicit_source_ids: set[int] = set()
    for sole_fact in sole_facts:
        sole_by_id = dict(zip(sole_fact.review_ids, sole_fact.evidence_texts))
        for wear_fact in wear_facts:
            wear_by_id = dict(zip(wear_fact.review_ids, wear_fact.evidence_texts))
            for review_id in set(sole_by_id) & set(wear_by_id):
                sole_text = normalize_review_text(sole_by_id[review_id])
                if sole_text != normalize_review_text(wear_by_id[review_id]):
                    continue
                if re.search(
                    r"바닥.{0,10}(?:얇아서|얇기\s*때문|얇은\s*탓|얇아\s+)"
                    r".{0,24}(?:오래|장시간|종일|부담|피로|아프)",
                    sole_text,
                ):
                    explicit_source_ids.add(review_id)
    return len(explicit_source_ids) >= 2


def _point_claims_are_traceable(
    summary: ShoeFitSummary,
    *,
    shoe_name: str,
    facts: list[PointEvidenceFact],
) -> bool:
    if not facts:
        # Preserve only the legacy non-assertive smoke-test placeholder. Any
        # product claim without typed evidence must be replaced by the trusted
        # deterministic fallback.
        neutral = normalize_review_text(summary.point_summary)
        return not summary.point_summary_claims and neutral in {
            "종합",
            "종합 결과입니다",
            "종합 적합도 결과입니다",
        }
    if not summary.point_summary_claims:
        return False
    if any(
        re.search(r"[.!?]", claim.text.rstrip(". "))
        for claim in summary.point_summary_claims
    ):
        return False
    expected_summary = " ".join(
        f"{claim.text.rstrip('.')}." for claim in summary.point_summary_claims
    )
    if normalize_review_text(expected_summary) != normalize_review_text(
        summary.point_summary
    ):
        return False

    facts_by_id = {fact.evidence_id: fact for fact in facts}
    for claim in summary.point_summary_claims:
        if not claim.evidence_ids or not set(claim.evidence_ids).issubset(facts_by_id):
            return False
        normalized_claim = normalize_review_text(claim.text)
        if any(
            term in normalized_claim
            for term in (
                *_POINT_UNIVERSAL_CLAIM_TERMS,
                *_POINT_UNTRACKED_ATTRIBUTE_TERMS,
                *_POINT_EXAGGERATION_TERMS,
            )
        ):
            return False
        if _POINT_UNIVERSAL_CLAIM_PATTERN.search(normalized_claim):
            return False
        cited_facts = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        if any(
            fact.mode == "CONFLICT"
            or point_evidence_strength(fact, facts) == "LOW"
            for fact in cited_facts
        ):
            return False
        cited_text = " ".join(
            [shoe_name, *(text for fact in cited_facts for text in fact.evidence_texts)]
        )
        if _uses_unsupported_concept(
            claim.text,
            cited_text,
            additional_concepts=_POINT_GROUNDED_CONCEPTS,
        ):
            return False
        if _claim_uses_uncited_signal(claim.text, cited_facts):
            return False
        if not _lab_claim_is_exactly_scoped(claim.text, cited_facts):
            return False
        if not all(_fact_is_expressed(claim.text, fact) for fact in cited_facts):
            return False
        if not _review_facts_are_subject_locally_expressed(
            claim.text, cited_facts
        ):
            return False
        if not _claim_segments_are_typed(claim.text, cited_facts):
            return False
        if not _styling_scope_is_supported(claim.text, cited_facts):
            return False
        if not _appearance_scope_is_supported(claim.text, cited_facts):
            return False
        if not _single_facts_are_safely_scoped(claim.text, cited_facts):
            return False
        if not _mixed_facts_are_safely_scoped(claim.text, cited_facts):
            return False
        if not _size_claim_is_consistent(claim.text, cited_facts):
            return False
        if not _foot_condition_scope_is_supported(claim.text, cited_facts):
            return False
        if not _cross_evidence_causality_is_supported(claim.text, cited_facts):
            return False
        numeric_claims = set(re.findall(r"\d+(?:\.\d+)?", claim.text))
        grounded_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_text))
        if numeric_claims - grounded_numbers:
            return False
    return True


def _contains_mechanical_area_result_summary(text: str) -> bool:
    """Reject a pointSummary that merely repeats the three fit-result labels.

    The individual reason cards already expose those results.  The product-level
    summary should explain concrete fit/product evidence instead of opening with
    a compressed ``FOREFOOT / HEEL / INSOLE`` status list.
    """

    normalized = normalize_review_text(text)
    if "적합도" in normalized:
        return True
    if any(enum_name.lower() in normalized for enum_name in REASON_TYPES):
        return True

    result_terms = ("적합도", "주의", "좋음", "좋습니다", "보통", "위험도")
    area_result_sentence_count = 0
    area_subjects: set[str] = set()
    for sentence in re.split(r"[.!?]+", normalized):
        mentioned_areas = sum(
            any(term in sentence for term in terms)
            for terms in _AREA_CONCEPTS.values()
        )
        if mentioned_areas >= 2 and any(term in sentence for term in result_terms):
            return True
        if mentioned_areas and any(term in sentence for term in result_terms):
            area_result_sentence_count += 1
        for reason_type, terms in _AREA_CONCEPTS.items():
            if any(
                re.search(rf"{re.escape(term)}\s*(?:은|는|:)", sentence)
                for term in terms
            ):
                area_subjects.add(reason_type)
    if area_result_sentence_count >= 2 or len(area_subjects) >= 3:
        return True
    return any(pattern in normalized for pattern in _POINT_REPORT_STYLE_PATTERNS)


_REASON_TOPIC_CONCEPTS = {
    "FOREFOOT": ("발볼", "앞볼", "앞코", "발가락", "엄지"),
    "HEEL": ("뒤꿈치", "하중", "충격", "반발", "쿠션", "안정감", "착화감"),
    "INSOLE": ("깔창", "인솔", "통기", "통풍", "열감", "습기", "내부"),
}
_REASON_CAUTION_TERMS = (
    "주의",
    "위험",
    "부족",
    "좁",
    "압박",
    "타이트",
    "불편",
    "아프",
    "피로",
    "헐렁",
    "들뜨",
    "쓸",
    "얇",
    "단단",
    "딱딱",
    "사이즈업",
    "사이즈 업",
    "맞지 않",
    "충분하지 못",
    "낮은 편",
)


def _generated_reason_summary_is_grounded(
    generated: str,
    reason: ReasonFactsForPrompt,
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> bool:
    """Validate one card without requiring mechanical title duplication.

    The title is already rendered by the detail UI.  Requiring the generated
    prose to repeat it caused otherwise grounded Ollama text to be discarded,
    and a single rejected card previously replaced all three cards.  We instead
    require the correct body area, source-grounded concepts, and a caution cue
    for HIGH risk.  LOW risk still rejects an unnegated caution claim so the
    model cannot reverse the quantitative decision.
    """

    try:
        validate_reason_output(
            reason.reason_type,
            reason.risk_level,
            [
                ReviewCandidate(review_id, text, 1.0)
                for review_id, text in zip(
                    reason.review_ids, reason.review_texts, strict=True
                )
            ],
            generated,
        )
    except AssertionError:
        return False
    if not reason.review_ids:
        return generated == NO_REVIEW_SUMMARY

    generated_evidence = analyze_reason_review(reason.reason_type, generated)
    expected_polarity = {"LOW": "POSITIVE", "HIGH": "NEGATIVE"}.get(
        reason.risk_level
    )
    if (
        expected_polarity is not None
        and (
            generated_evidence is None
            or generated_evidence.polarity != expected_polarity
        )
    ):
        return False

    normalized = normalize_review_text(generated)
    normalized_title = normalize_review_text(reason.title)
    if not normalized or not (
        _contains_any(normalized, _REASON_TOPIC_CONCEPTS[reason.reason_type])
        or (normalized_title and normalized.startswith(normalized_title))
    ):
        return False
    if _contains_out_of_scope_content(generated, _REASON_OUT_OF_SCOPE_SUMMARY_TERMS):
        return False
    measurement = _personal_measurement_sentence(
        reason.reason_type, personal_fit_evidence
    )
    source_text = " ".join(
        [reason.title, *reason.review_texts, measurement or ""]
    )
    if _uses_unsupported_concept(generated, source_text):
        return False
    has_unnegated_caution = _has_unnegated_term(normalized, _REASON_CAUTION_TERMS)
    if reason.risk_level == "HIGH" and not has_unnegated_caution:
        return False
    if (
        reason.risk_level == "LOW"
        and has_unnegated_caution
        and measurement is None
    ):
        return False
    return True


def _generated_reason_summary_grounding(
    summary: ShoeFitSummary,
    *,
    reasons: list[ReasonFactsForPrompt],
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> dict[str, bool]:
    return {
        reason.reason_type: _generated_reason_summary_is_grounded(
            getattr(summary, _SUMMARY_FIELD_BY_REASON[reason.reason_type]),
            reason,
            personal_fit_evidence,
        )
        for reason in reasons
    }


def _generated_reason_summaries_are_grounded(
    summary: ShoeFitSummary,
    *,
    reasons: list[ReasonFactsForPrompt],
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> bool:
    return all(
        _generated_reason_summary_grounding(
            summary,
            reasons=reasons,
            personal_fit_evidence=personal_fit_evidence,
        ).values()
    )


def _generated_point_summary_is_grounded(
    summary: ShoeFitSummary,
    *,
    fit_score: float,
    shoe_name: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> bool:
    if _contains_mechanical_area_result_summary(summary.point_summary):
        return False
    if _contains_out_of_scope_content(
        summary.point_summary, _POINT_OUT_OF_SCOPE_SUMMARY_TERMS
    ):
        return False
    normalized_summary = normalize_review_text(summary.point_summary)
    if any(term in normalized_summary for term in ("리뷰", "후기", "의견")):
        return False
    if not _point_claims_are_traceable(
        summary, shoe_name=shoe_name, facts=point_evidence
    ):
        return False
    point_source = " ".join(
        [
            shoe_name,
            str(fit_score),
            *(reason.title for reason in reasons),
            *(
                text
                for fact in point_evidence
                for text in fact.evidence_texts
            ),
        ]
    )
    if _uses_unsupported_concept(
        summary.point_summary,
        point_source,
        additional_concepts=_POINT_GROUNDED_CONCEPTS,
    ):
        return False
    numeric_claims = set(re.findall(r"\d+(?:\.\d+)?", summary.point_summary))
    grounded_numbers = set(re.findall(r"\d+(?:\.\d+)?", point_source))
    grounded_numbers.update(
        {
            str(fit_score),
            f"{fit_score:g}",
            f"{fit_score:.1f}",
            f"{fit_score:.2f}",
        }
    )
    if numeric_claims - grounded_numbers:
        return False
    return True


def _fallback_point_summary(
    *,
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> tuple[str, list[ShoePointSummaryClaim]]:
    del fit_score, overall_risk_level, reasons
    claims = render_product_point_summary(
        shoe_name,
        point_evidence,
        include_attributed_single_evidence=True,
    )
    natural_claim_text = {
        "발볼 공간은 낮은 편입니다": "발볼은 비교적 슬림하게 나온 편입니다",
        "발볼 공간은 보통 수준입니다": "발볼은 평균적인 너비로 나온 편입니다",
        "발볼 공간은 높은 편입니다": "발볼은 비교적 여유 있게 나온 편입니다",
        "앞코 공간은 낮은 편입니다": "앞코는 비교적 여유가 적은 편입니다",
        "앞코 공간은 보통 수준입니다": "앞코는 무난한 너비로 나온 편입니다",
        "앞코 공간은 높은 편입니다": "앞코는 비교적 여유 있게 나온 편입니다",
        "쿠션감은 낮은 편입니다": "쿠션은 비교적 단단하게 느껴지는 편입니다",
        "쿠션감은 보통 수준입니다": "쿠션은 단단함과 부드러움이 균형 잡힌 편입니다",
        "쿠션감은 높은 편입니다": "쿠션은 비교적 부드럽게 느껴지는 편입니다",
        "뒤꿈치 구조 강성은 낮은 편입니다": "뒤꿈치 구조는 비교적 유연한 편입니다",
        "뒤꿈치 구조 강성은 보통 수준입니다": "뒤꿈치 구조는 유연함과 지지력이 균형 잡힌 편입니다",
        "뒤꿈치 구조 강성은 높은 편입니다": "뒤꿈치 구조는 비교적 단단하게 잡아주는 편입니다",
        "충격 완화 수준은 낮은 편입니다": "충격 흡수력은 비교적 낮은 편입니다",
        "충격 완화 수준은 보통 수준입니다": "충격 흡수력은 무난한 편입니다",
        "충격 완화 수준은 높은 편입니다": "충격 흡수력은 비교적 높은 편입니다",
        "착화 시 가볍게 느껴지는 편입니다": "가볍게 느껴진다는 착화 경험이 반복됩니다",
        "착화 시 다소 무게감이 느껴지는 편입니다": "다소 무게감이 느껴진다는 착화 경험이 반복됩니다",
    }
    naturalized_claims = [
        ShoePointSummaryClaim(
            text=natural_claim_text.get(claim.text, claim.text),
            evidence_ids=list(claim.evidence_ids),
        )
        for claim in claims
    ]
    return (
        " ".join(f"{claim.text.rstrip('.')}." for claim in naturalized_claims),
        naturalized_claims,
    )


def _max_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _personal_width_ratio(evidence: ServerPersonalFitEvidence) -> float | None:
    ratios = [
        float(width) / float(length)
        for width, length in (
            (
                evidence.left_foot_width_mm,
                evidence.measured_left_foot_size_mm,
            ),
            (
                evidence.right_foot_width_mm,
                evidence.measured_right_foot_size_mm,
            ),
        )
        if width is not None and length is not None
    ]
    # One wider foot is enough to require extra room in a pair of shoes.
    return max(ratios) if ratios else None


def _personal_pressure_ratio(
    evidence: ServerPersonalFitEvidence, attribute: str
) -> float | None:
    return _max_present(
        [
            getattr(analysis, attribute)
            for analysis in evidence.static_pressure_analyses
        ]
    )


def _join_korean_items(items: list[str]) -> str:
    if not items:
        raise ValueError("items cannot be empty")
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f"와 {items[-1]}"


def _measurement_findings_sentence(
    findings: list[str], checked_signals: list[str]
) -> str | None:
    if findings:
        if len(findings) == 1:
            return f"측정 결과 {findings[0]}입니다."
        return (
            f"측정 결과 {', '.join(findings[:-1])}이고, "
            f"{findings[-1]}입니다."
        )
    if checked_signals:
        return (
            f"측정 결과 {_join_korean_items(checked_signals)}에서 "
            "추가 주의 신호는 크지 않습니다."
        )
    return None


def _personal_measurement_sentence(
    reason_type: str,
    evidence: ServerPersonalFitEvidence | None,
) -> str | None:
    """Describe only measurements explicitly present in the Server context.

    A risk level alone is enough for compatibility guidance, but never enough
    to claim pressure, load, humidity, or a measured foot-width ratio.  Those
    statements are therefore gated by the matching nullable evidence fields.
    """

    if evidence is None:
        return None

    if reason_type == "FOREFOOT":
        findings: list[str] = []
        checked: list[str] = []
        width_ratio = _personal_width_ratio(evidence)
        if width_ratio is not None:
            checked.append("발볼 비율")
            if width_ratio >= settings.shoe_forefoot_width_ratio_neutral:
                findings.append("발볼 여유가 더 필요한 상태")
        toe_angles = [
            value
            for value in (
                evidence.left_toe_angle_degree,
                evidence.right_toe_angle_degree,
            )
            if value is not None
        ]
        if toe_angles:
            checked.append("발 정렬 지표")
            if max(toe_angles) >= settings.shoe_hallux_angle_neutral_degree:
                findings.append("앞코 압박을 줄일 여유가 필요한 상태")
        return _measurement_findings_sentence(findings, checked)

    if reason_type == "HEEL":
        findings = []
        checked = []
        rearfoot_pressure = _personal_pressure_ratio(
            evidence, "rearfoot_pressure_ratio"
        )
        if rearfoot_pressure is not None:
            checked.append("뒤꿈치 하중")
            if (
                rearfoot_pressure
                >= settings.shoe_rearfoot_pressure_neutral_percent
            ):
                findings.append("뒤꿈치 쪽 하중이 높은 편")

        if (
            evidence.left_pressure_percent is not None
            and evidence.right_pressure_percent is not None
        ):
            pressure_gaps = [
                abs(evidence.left_pressure_percent - evidence.right_pressure_percent)
            ]
        else:
            pressure_gaps = []
        pressure_gaps.extend(
            abs(row.left_pressure_ratio - row.right_pressure_ratio)
            for row in evidence.static_pressure_analyses
            if row.left_pressure_ratio is not None
            and row.right_pressure_ratio is not None
        )
        if pressure_gaps:
            checked.append("좌우 압력 균형")
            if max(pressure_gaps) >= settings.shoe_pressure_imbalance_neutral_percent:
                findings.append("좌우 압력 차이가 큰 편")

        balance_scores = [
            value
            for value in (
                evidence.balance_score,
                *(row.balance_score for row in evidence.static_pressure_analyses),
            )
            if value is not None
        ]
        if balance_scores:
            checked.append("자세 균형")
            if min(balance_scores) < settings.shoe_balance_neutral_score:
                findings.append("자세 균형 점수가 낮은 편")
        return _measurement_findings_sentence(findings, checked)

    if reason_type == "INSOLE":
        findings = []
        checked = []
        if evidence.avg_humidity_percent is not None:
            checked.append("발 내부 습도")
            if (
                evidence.avg_humidity_percent
                >= settings.shoe_humidity_neutral_percent
            ):
                findings.append("발 내부 습도가 높은 편")
        if evidence.fungal_suspicion_safety_score is not None:
            checked.append("피부 환경 지표")
            if (
                evidence.fungal_suspicion_safety_score
                < settings.shoe_fungal_safety_neutral_score
            ):
                findings.append("통기성 관리가 필요한 상태")
        if evidence.skin_reaction_safety_score is not None:
            checked.append("피부 반응 지표")
            if (
                evidence.skin_reaction_safety_score
                < settings.shoe_skin_safety_neutral_score
            ):
                findings.append("피부 반응 관리가 필요한 상태")
        return _measurement_findings_sentence(findings, checked)

    raise ShoeFitCommentError(f"Unknown reason type: {reason_type}")


def _natural_reason_fallback(
    reason: ReasonFactsForPrompt,
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> str:
    measurement = _personal_measurement_sentence(
        reason.reason_type, personal_fit_evidence
    )
    candidates = [
        ReviewCandidate(review_id, review_text, 1.0)
        for review_id, review_text in zip(
            reason.review_ids, reason.review_texts, strict=True
        )
    ]
    return render_reason_summary(
        reason.reason_type,
        reason.risk_level,
        candidates,
        measurement_sentence=measurement,
    )


def _lock_reason_summaries_to_policy(
    summary: ShoeFitSummary,
    reasons: list[ReasonFactsForPrompt],
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> ShoeFitSummary:
    """Make the deterministic evidence policy the final authority for cards.

    Ollama can refine the product-wide pointSummary, but a free-form model
    response must never reintroduce a cross-area review, reverse LOW/HIGH
    polarity, or describe a reviewless area. The selected review IDs and all
    three reason summaries are therefore locked together immediately before
    the response leaves this service.
    """

    updates: dict[str, object] = {}
    for reason in reasons:
        updates[_SUMMARY_FIELD_BY_REASON[reason.reason_type]] = (
            _natural_reason_fallback(reason, personal_fit_evidence)
        )
        updates[_REVIEW_IDS_FIELD_BY_REASON[reason.reason_type]] = list(
            reason.review_ids
        )
    return summary.model_copy(update=updates)


def _generated_point_summary_keeps_important_evidence(
    summary: ShoeFitSummary,
    point_evidence: list[PointEvidenceFact],
) -> bool:
    """Prevent a valid but thin LLM sentence from replacing richer safe copy.

    Schema and grounding validation can prove that a sentence is true while it
    still ignores the most useful fit evidence (for example, returning only a
    design sentence when three reviews agree on sizing).  The deterministic
    renderer establishes a conservative coverage floor.  Ollama may rewrite or
    condense it, but must retain each fit/comfort claim and must not collapse a
    multi-point description to one claim.
    """

    fallback_claims = [
        claim
        for claim in render_product_point_summary(
            "",
            point_evidence,
            include_attributed_single_evidence=True,
        )
        if claim.evidence_ids
    ]
    if not fallback_claims:
        return True

    facts_by_id = {fact.evidence_id: fact for fact in point_evidence}
    priority_categories = {
        "SIZE_FIT",
        "WIDTH_FIT",
        "HEEL_FEEL",
        "CUSHION_FEEL",
        "LONG_WEAR",
        "WEIGHT_FEEL",
        "OTHER",
    }
    required_groups = [
        {
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in facts_by_id
            and facts_by_id[evidence_id].signal.category in priority_categories
        }
        for claim in fallback_claims
    ]
    required_groups = [group for group in required_groups if group]
    generated_ids = {
        evidence_id
        for claim in summary.point_summary_claims
        for evidence_id in claim.evidence_ids
    }
    if any(not group.issubset(generated_ids) for group in required_groups):
        return False

    fit_categories = {"SIZE_FIT", "WIDTH_FIT"}
    required_fit_groups = [
        {
                evidence_id
                for evidence_id in claim.evidence_ids
                if evidence_id in facts_by_id
                and facts_by_id[evidence_id].signal.category in fit_categories
        }
        for claim in fallback_claims
        if any(
            evidence_id in facts_by_id
            and facts_by_id[evidence_id].signal.category in fit_categories
            for evidence_id in claim.evidence_ids
        )
    ]
    if required_fit_groups:
        if not summary.point_summary_claims:
            return False
        generated_claim_id_sets = [
            set(claim.evidence_ids) for claim in summary.point_summary_claims
        ]
        fit_group_indices: list[int] = []
        for required_fit_group in required_fit_groups:
            group_index = next(
                (
                    index
                    for index, evidence_ids in enumerate(generated_claim_id_sets)
                    if required_fit_group.issubset(evidence_ids)
                ),
                -1,
            )
            if group_index < 0:
                return False
            fit_group_indices.append(group_index)
        if fit_group_indices[0] != 0 or fit_group_indices != sorted(fit_group_indices):
            return False
        last_fit_index = max(fit_group_indices)
        for evidence_ids in generated_claim_id_sets[:last_fit_index]:
            if not any(
                evidence_id in facts_by_id
                and facts_by_id[evidence_id].signal.category in fit_categories
                for evidence_id in evidence_ids
            ):
                return False
        first_text = normalize_review_text(summary.point_summary_claims[0].text)
        fit_positions = [
            first_text.find(term)
            for term in (
                "발볼 공간",
                "앞코 공간",
                "발볼",
                "앞코",
                "발등",
                "정사이즈",
                "사이즈",
                "전체 핏",
                "착화 핏",
                "전체적으로 딱 맞",
                "딱 맞는 핏",
            )
            if term in first_text
        ]
        non_fit_positions = [
            first_text.find(term)
            for term in (
                "디자인",
                "외관",
                "색감",
                "색상",
                "예쁘",
                "예뻐",
                "이쁘",
                "이뻐",
                "깔끔",
                "만족",
                "실루엣",
                "쉐입",
                "날렵",
                "로우",
                "뒤꿈치",
                "발목",
                "쿠션",
                "바닥",
                "무게",
                "가볍",
                "무겁",
                "착화감",
                "장시간",
                "오래",
                "청바지",
                "데님",
                "슬랙스",
                "스커트",
                "치마",
                "코디",
                "스타일",
                "잘 어울",
                "어울",
                "활용",
                "데일리",
                "통기",
                "반발력",
                "에너지 리턴",
                "충격 완화",
                "뒤꿈치 구조",
            )
            if term in first_text
        ]
        if not fit_positions or (
            non_fit_positions and min(non_fit_positions) < min(fit_positions)
        ):
            return False

    minimum_claim_count = min(2, len(fallback_claims))
    return len(summary.point_summary_claims) >= minimum_claim_count


def _deterministic_grounded_summary(
    *,
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
    selected_summary: ShoeFitSummary,
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> ShoeFitSummary:
    reason_summaries = {
        reason.reason_type: _natural_reason_fallback(
            reason, personal_fit_evidence
        )
        for reason in reasons
    }

    point_summary, point_summary_claims = _fallback_point_summary(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        reasons=reasons,
        point_evidence=point_evidence,
    )
    return ShoeFitSummary(
        point_summary=point_summary,
        point_summary_claims=point_summary_claims,
        forefoot_summary=reason_summaries["FOREFOOT"],
        heel_summary=reason_summaries["HEEL"],
        insole_summary=reason_summaries["INSOLE"],
        forefoot_review_ids=selected_summary.forefoot_review_ids,
        heel_review_ids=selected_summary.heel_review_ids,
        insole_review_ids=selected_summary.insole_review_ids,
    )


def _typed_evidence_fallback(
    *,
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> ShoeFitSummary:
    seed = ShoeFitSummary(
        point_summary="typed evidence fallback",
        forefoot_summary=next(
            reason.title for reason in reasons if reason.reason_type == "FOREFOOT"
        ),
        heel_summary=next(
            reason.title for reason in reasons if reason.reason_type == "HEEL"
        ),
        insole_summary=next(
            reason.title for reason in reasons if reason.reason_type == "INSOLE"
        ),
        forefoot_review_ids=next(
            reason.review_ids for reason in reasons if reason.reason_type == "FOREFOOT"
        ),
        heel_review_ids=next(
            reason.review_ids for reason in reasons if reason.reason_type == "HEEL"
        ),
        insole_review_ids=next(
            reason.review_ids for reason in reasons if reason.reason_type == "INSOLE"
        ),
    )
    fallback = _deterministic_grounded_summary(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        reasons=reasons,
        point_evidence=point_evidence,
        selected_summary=seed,
        personal_fit_evidence=personal_fit_evidence,
    )
    return _ensure_point_evidence_reviews_selected(fallback, reasons, point_evidence)


def _output_contract_error_summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        issues: list[str] = []
        for issue in error.errors(include_url=False, include_input=False)[:8]:
            location = ".".join(str(part) for part in issue.get("loc", ())) or "<root>"
            issues.append(f"{location}:{issue.get('type', 'validation_error')}")
        return ", ".join(issues)
    message = str(error).replace("\n", " ").strip()
    return f"{type(error).__name__}: {message[:300]}"


def _log_summary_generation_completed(*, shoe_name: str, source: str) -> None:
    logger.info(
        "Shoe summary generation completed. shoe_name=%s source=%s",
        shoe_name,
        source,
    )


def build_immediate_shoe_summaries(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    characteristics: ServerShoeCharacteristics | None = None,
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> ShoeFitSummary:
    """Return a fully grounded provisional summary without an LLM call.

    This is persisted before the slower Ollama refinement.  It keeps first-view
    latency bounded by Server round trips while reusing the exact same evidence,
    review-ID, and product-copy safety rules as the final fallback.
    """

    if overall_risk_level not in _RISK_LABEL:
        raise ShoeFitCommentError(f"Unknown overall risk level: {overall_risk_level}")
    if not 0 <= fit_score <= 100:
        raise ShoeFitCommentError("fit_score must be between 0 and 100")
    display_shoe_name = _display_shoe_name(shoe_name)
    point_evidence = prepare_point_summary_evidence(reasons, characteristics)
    grounded_reasons = prepare_grounded_reasons(reasons)
    return _typed_evidence_fallback(
        shoe_name=display_shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        reasons=grounded_reasons,
        point_evidence=point_evidence,
        personal_fit_evidence=personal_fit_evidence,
    )


async def generate_shoe_summaries(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    characteristics: ServerShoeCharacteristics | None = None,
    personal_fit_evidence: ServerPersonalFitEvidence | None = None,
) -> ShoeFitSummary:
    """신발 하나의 fitScore/riskLevel/부위별 판단 결과를 바탕으로, Ollama로
    pointSummary + 부위별 reviewSummary(발볼/뒤꿈치/깔창)를 한 번의 호출로 생성한다."""
    if overall_risk_level not in _RISK_LABEL:
        raise ShoeFitCommentError(f"Unknown overall risk level: {overall_risk_level}")
    if not 0 <= fit_score <= 100:
        raise ShoeFitCommentError("fit_score must be between 0 and 100")
    display_shoe_name = _display_shoe_name(shoe_name)
    point_evidence = prepare_point_summary_evidence(reasons, characteristics)
    grounded_reasons = prepare_grounded_reasons(reasons)
    user_prompt = _build_user_prompt(
        display_shoe_name,
        fit_score,
        overall_risk_level,
        grounded_reasons,
        point_evidence,
        personal_fit_evidence,
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    generation_source = "OLLAMA"

    try:
        await asyncio.wait_for(
            _ollama_semaphore.acquire(),
            timeout=settings.ollama_queue_timeout_seconds,
        )
    except TimeoutError as exc:
        raise ShoeFitCommentError(
            "Ollama summary runtime is busy; the exact-session request can be retried."
        ) from exc
    try:
        response = await _invoke_ollama(messages)
        try:
            parsed = _parse_llm_json(response.content)
            summary = ShoeFitSummary.model_validate(parsed)
        except (ShoeFitCommentError, ValidationError) as first_error:
            logger.warning(
                "Ollama shoe-summary schema validation failed; retrying one correction. "
                "errors=%s",
                _output_contract_error_summary(first_error),
            )
            correction_messages: list[BaseMessage] = [
                *messages,
                HumanMessage(content=SCHEMA_CORRECTION_PROMPT),
            ]
            try:
                corrected_response = await _invoke_ollama(correction_messages)
                corrected_parsed = _parse_llm_json(corrected_response.content)
                summary = ShoeFitSummary.model_validate(corrected_parsed)
            except (ShoeFitCommentError, ValidationError) as correction_error:
                logger.warning(
                    "Ollama shoe-summary correction failed; "
                    "source=TYPED_EVIDENCE_FALLBACK first_errors=%s correction_errors=%s",
                    _output_contract_error_summary(first_error),
                    _output_contract_error_summary(correction_error),
                )
                fallback = _typed_evidence_fallback(
                    shoe_name=display_shoe_name,
                    fit_score=fit_score,
                    overall_risk_level=overall_risk_level,
                    reasons=grounded_reasons,
                    point_evidence=point_evidence,
                    personal_fit_evidence=personal_fit_evidence,
                )
                _log_summary_generation_completed(
                    shoe_name=display_shoe_name,
                    source="TYPED_EVIDENCE_FALLBACK",
                )
                return fallback
            generation_source = "OLLAMA_CORRECTED"
    finally:
        _ollama_semaphore.release()
    try:
        _validate_review_selection(summary, grounded_reasons)
    except ShoeFitCommentError as selection_error:
        logger.warning(
            "Ollama returned a reviewId outside the allowed per-reason whitelist; "
            "ignoring model-selected IDs and applying the deterministic selection. "
            "errors=%s",
            _output_contract_error_summary(selection_error),
        )
        generation_source = f"{generation_source}_WITH_REVIEW_ID_OVERRIDE"
    summary = _ensure_review_selection_minimum(summary, grounded_reasons)
    reason_summary_grounding = _generated_reason_summary_grounding(
        summary,
        reasons=grounded_reasons,
        personal_fit_evidence=personal_fit_evidence,
    )
    reason_summaries_grounded = all(reason_summary_grounding.values())
    point_summary_grounded = _generated_point_summary_is_grounded(
        summary,
        shoe_name=display_shoe_name,
        fit_score=fit_score,
        reasons=grounded_reasons,
        point_evidence=point_evidence,
    )
    if point_summary_grounded and not _generated_point_summary_keeps_important_evidence(
        summary, point_evidence
    ):
        logger.warning(
            "Ollama pointSummary omitted higher-priority typed evidence; "
            "using the deterministic product-copy fallback."
        )
        point_summary_grounded = False
    if not reason_summaries_grounded or not point_summary_grounded:
        logger.warning(
            "Ollama shoe summary failed deterministic grounding checks "
            "(reasonSummaries=%s, pointSummary=%s); replacing only unsafe fields.",
            reason_summary_grounding,
            point_summary_grounded,
        )
        fallback = _deterministic_grounded_summary(
            shoe_name=display_shoe_name,
            fit_score=fit_score,
            overall_risk_level=overall_risk_level,
            reasons=grounded_reasons,
            point_evidence=point_evidence,
            selected_summary=summary,
            personal_fit_evidence=personal_fit_evidence,
        )
        updates: dict[str, object] = {}
        if not point_summary_grounded:
            updates["point_summary"] = fallback.point_summary
            updates["point_summary_claims"] = fallback.point_summary_claims
        for reason_type, grounded in reason_summary_grounding.items():
            if grounded:
                continue
            field_name = _SUMMARY_FIELD_BY_REASON[reason_type]
            updates[field_name] = getattr(fallback, field_name)
        grounded_result = summary.model_copy(update=updates)
        result = _ensure_point_evidence_reviews_selected(
            grounded_result, grounded_reasons, point_evidence
        )
        generation_source = f"{generation_source}_WITH_GROUNDED_FIELD_FALLBACK"
    else:
        result = _ensure_point_evidence_reviews_selected(
            summary, grounded_reasons, point_evidence
        )
    result = _lock_reason_summaries_to_policy(
        result,
        grounded_reasons,
        personal_fit_evidence,
    )
    _log_summary_generation_completed(
        shoe_name=display_shoe_name,
        source=generation_source,
    )
    return result
