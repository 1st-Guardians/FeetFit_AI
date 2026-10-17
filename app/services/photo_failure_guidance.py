"""Allowlisted, user-facing guidance for photo-analysis failures.

Only stable diagnostic codes and known upload-field names are accepted here.
Raw exception messages must never be used as client messages or backend
``failureDetail`` values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhotoFailureGuidance:
    diagnostic_code: str
    user_message: str
    recommended_action: str

    @property
    def failure_detail(self) -> str:
        """Human-readable text safe to forward from the backend to the app."""

        return f"{self.user_message} {self.recommended_action}"


_MARKER_OCCLUSION_EXPLANATIONS = {
    "PARTIAL_MARKER_NOT_FOUND": "발은 검출됐지만 예상 구간에서 부분 마커의 검은 테두리 후보를 찾지 못했습니다.",
    "PARTIAL_MARKER_CORNERS_INSUFFICIENT": "부분 마커 후보는 있지만 인접한 두 외곽 모서리와 흰 여백을 확인하지 못했습니다.",
    "PARTIAL_MARKER_FOOT_OVERLAP_UNCONFIRMED": "부분 마커의 예상 영역을 신뢰할 수 있는 같은 발 마스크가 덮는 조건을 확인하지 못했습니다.",
    "PARTIAL_MARKER_PATTERN_UNCONFIRMED": "발과 겹치는 부분 마커 후보는 있지만 노출된 흑백 무늬를 검증하지 못했습니다.",
    "PARTIAL_MARKER_INPUT_INVALID": "부분 마커 검사에 필요한 좌표·발 마스크·설정이 유효하지 않습니다.",
    "PARTIAL_MARKER_PROCESSING_FAILED": "부분 마커 검사 실행에 실패해 발 가림을 판정하지 못했습니다.",
    "PARTIAL_MARKER_CORNERS_COVERED_BY_FOOT": "마커의 남아 있는 두 모서리와 흑백 무늬를 확인했고, 나머지 모서리 위치를 발이 덮어 가림으로 판정했습니다.",
    "SOURCE_IMAGE_INVALID": "사진 형식이 유효하지 않아 발 가림을 검사하지 못했습니다.",
    "ARUCO_RUNTIME_UNAVAILABLE": "마커 검출기를 불러오지 못해 발 가림을 검사하지 못했습니다.",
    "MARKER_DETECTION_FAILED": "마커 검출 실행에 실패해 발 가림을 검사하지 못했습니다.",
    "REQUIRED_MARKERS_NOT_DETECTED": "두 마커가 모두 보이지 않아 발과 비교할 마커 위치를 특정하지 못했습니다.",
    "REQUIRED_MARKERS_VISIBLE": "재검사에서는 두 마커가 보입니다. 마커 배치와 보정 조건을 확인해야 합니다.",
    "DUPLICATE_MARKERS_DETECTED": "동일한 마커 ID가 중복 검출되어 기준 위치를 특정하지 못했습니다.",
    "MARKER_IDENTITY_UNVERIFIED": "검출한 마커 ID를 검증하지 못해 기준 위치를 특정하지 못했습니다.",
    "CALIBRATION_UNAVAILABLE": "카메라 보정 자료를 불러오지 못해 발 가림을 검사하지 못했습니다.",
    "CALIBRATION_QUALITY_INVALID": "카메라 보정 오차가 기준을 벗어나 발 가림을 판정하지 못했습니다.",
    "LENS_CORRECTION_FAILED": "렌즈 보정에 실패해 발 가림을 검사하지 못했습니다.",
    "MARKER_DETECTIONS_INCONSISTENT": "보정 전후 마커 검출이 일치하지 않아 발 가림을 판정하지 못했습니다.",
    "MARKER_PROJECTION_UNCERTAIN": "마커 예상 위치가 불안정해 발과의 겹침을 판정하지 못했습니다.",
    "MARKER_PROJECTION_OUTSIDE_VALID_IMAGE": "마커 예상 위치가 유효 촬영 영역을 벗어났습니다. 카메라 구도와 보정 조건을 확인해야 합니다.",
    "FOOT_MODEL_UNAVAILABLE": "발 검출 모델을 불러오지 못해 발 가림을 검사하지 못했습니다.",
    "FOOT_MODEL_INFERENCE_FAILED": "발 검출 모델 실행에 실패해 발 가림을 검사하지 못했습니다.",
    "FOOT_MODEL_NO_RESULTS": "발 검출 모델이 결과를 반환하지 않아 발 가림을 판정하지 못했습니다.",
    "FOOT_MODEL_RESULT_INVALID": "발 검출 결과 형식이 유효하지 않아 발 가림을 판정하지 못했습니다.",
    "NO_FOOT_DETECTED": "발 마스크를 찾지 못해 발 가림을 판정하지 못했습니다.",
    "NO_CONFIDENT_FOOT_DETECTED": "신뢰할 수 있는 발 마스크를 찾지 못해 발 가림을 판정하지 못했습니다.",
    "FOOT_COMPONENT_TOO_SMALL": "검출한 발 영역이 너무 작아 발 가림을 판정하지 못했습니다.",
    "FOOT_MASK_REMAP_FAILED": "발 마스크의 좌표 보정에 실패해 발 가림을 판정하지 못했습니다.",
    "FOOT_MASK_COVERS_VISIBLE_MARKER": "발 마스크가 보이는 마커까지 덮어 검출 오류 가능성이 있습니다. 가림을 확정하지 못했습니다.",
    "FOOT_OVERLAP_AMBIGUOUS": "발과 마커 예상 영역이 일부 겹치지만 가림 판정 기준을 충족하지 못했습니다.",
    "NO_FOOT_OVERLAP": "검출된 발은 마커 예상 위치와 겹치지 않습니다. 마커가 보이지 않는 다른 원인은 별도 확인이 필요합니다.",
    "FOOT_OVERLAP_CONFIRMED": "검출된 발이 보이지 않는 마커의 예상 영역을 덮어 발 가림으로 판정했습니다.",
    "DIAGNOSTIC_PROCESSING_FAILED": "가림 진단 처리에 실패해 원인을 판정하지 못했습니다.",
}


def marker_occlusion_explanation(reason_code: str | None) -> str | None:
    """Only fixed diagnostic explanations may reach logs or backend details."""
    return _MARKER_OCCLUSION_EXPLANATIONS.get(reason_code) if isinstance(reason_code, str) else None


_DORSAL_GUIDANCE = {
    "DORSAL_DECODE_FAILED": (
        "발등 사진 파일을 읽지 못했습니다.",
        "카메라 연결과 이미지 전송 상태를 확인한 뒤 새 측정을 시작해 발등 사진을 다시 촬영해 주세요.",
    ),
    "DORSAL_DIMENSIONS_MISMATCH": (
        "발등 사진 해상도가 측정 기준과 맞지 않습니다.",
        "카메라 해상도를 측정 기준에 맞춘 뒤 새 측정을 시작해 다시 촬영해 주세요.",
    ),
    "DORSAL_MARKERS_INSUFFICIENT": (
        "ArUco 마커 ID 1 또는 ID 4를 찾지 못했습니다.",
        "마커 가림·흐림·반사를 확인하고 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_MARKER_OCCLUDED_BY_FOOT": (
        "발이 기준 ArUco 마커 영역을 가려 마커를 인식하지 못했습니다.",
        "발이 마커를 가리지 않도록 발 위치를 조정한 뒤 다시 촬영하기를 눌러 주세요.",
    ),
    "DORSAL_ARUCO_GEOMETRY_INVALID": (
        "ArUco 마커 ID 1·4의 배치 또는 평면 보정이 측정 기준을 벗어났습니다.",
        "카메라를 발판과 평행하게 맞추고 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_FEET_NOT_DETECTED": (
        "발등 사진에서 양발 윤곽을 모두 찾지 못했습니다.",
        "양발 전체가 보이도록 자세를 맞추고 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_FOOT_POSITION_INVALID": (
        "마커 검증은 정상이나 설정된 뒤꿈치 오프셋을 포함한 발길이가 18~32cm 범위를 벗어났습니다.",
        "뒤꿈치를 고정 위치에 맞추고 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_FOOT_OUTLINE_INCOMPLETE": (
        "ArUco 검증은 정상이나 발 윤곽이 불완전해 치수를 계산하지 못했습니다.",
        "발 전체가 선명하게 보이도록 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_BALL_WIDTH_ZONE_NOT_VISIBLE": (
        "마커 검증은 정상이나 발 윤곽이 발길이의 32~36% 발볼 구간까지 나오지 않았습니다.",
        "발을 표시 위치까지 앞으로 맞추고 새 측정으로 다시 촬영해 주세요.",
    ),
    "DORSAL_BALL_WIDTH_UNSTABLE": (
        "ArUco 검증은 정상이나 발볼 단면이 부족하거나 너비가 5~14cm 범위를 벗어났습니다.",
        "발볼 전체가 선명하게 보이도록 새 측정으로 다시 촬영해 주세요.",
    ),
}

_UPLOAD_GUIDANCE = {
    "PHOTO_UPLOAD_EMPTY": (
        "{label} 데이터가 비어 있습니다.",
        "카메라 촬영 결과와 이미지 전송 상태를 확인한 뒤 새 측정을 시작해 {label}을 다시 촬영해 주세요.",
    ),
    "PHOTO_UPLOAD_TOO_LARGE": (
        "{label} 파일 용량이 허용 범위를 초과했습니다.",
        "카메라 해상도 또는 이미지 압축 설정을 확인한 뒤 새 측정을 시작해 {label}을 다시 촬영해 주세요.",
    ),
}

_PLANTAR_COMMON_GUIDANCE = {
    "PLANTAR_DECODE_FAILED": (
        "{side} 발바닥 사진 파일을 읽지 못했습니다.",
        "카메라 연결과 이미지 전송 상태를 확인한 뒤 새 측정을 시작해 {side} 발바닥 사진을 다시 촬영해 주세요.",
    ),
    "PLANTAR_DIMENSIONS_UNSAFE": (
        "{side} 발바닥 사진 크기가 분석 가능한 범위를 벗어났습니다.",
        "카메라 해상도와 이미지 파일 상태를 확인한 뒤 새 측정을 시작해 {side} 발바닥 사진을 다시 촬영해 주세요.",
    ),
    "PLANTAR_CAPTURE_BLANK": (
        "{side} 발바닥 사진에서 발 영역을 인식하지 못했습니다.",
        "발 전체가 화면 안에 선명하게 나오도록 자세와 카메라 위치를 맞춘 뒤 새 측정을 시작해 다시 촬영해 주세요.",
    ),
    "PLANTAR_CAPTURE_NEAR_WHITE": (
        "{side} 발바닥 사진이 너무 밝거나 비어 있어 발 영역을 인식하지 못했습니다.",
        "조명과 카메라 노출을 조정하고 발 전체가 화면 안에 나오도록 새 측정을 시작해 다시 촬영해 주세요.",
    ),
}

_SIDE_SPECIFIC_GUIDANCE = {
    "LEFT_PLANTAR_FOOT_NOT_DETECTED": (
        "왼발 발바닥 사진에서 발을 인식하지 못했습니다.",
        "왼발 전체가 프레임 안에 나오도록 제대로 선 뒤 새 측정을 시작해 다시 촬영해 주세요.",
    ),
    "RIGHT_PLANTAR_FOOT_NOT_DETECTED": (
        "오른발 발바닥 사진에서 발을 인식하지 못했습니다.",
        "오른발 전체가 프레임 안에 나오도록 제대로 선 뒤 새 측정을 시작해 다시 촬영해 주세요.",
    ),
}

_PLANTAR_FIELD_SIDE = {
    "leftPlantarFootImage": "왼발",
    "rightPlantarFootImage": "오른발",
}

_IMAGE_FIELD_LABEL = {
    "dorsalFootImage": "발등 사진",
    "leftPlantarFootImage": "왼발 발바닥 사진",
    "rightPlantarFootImage": "오른발 발바닥 사진",
}

_REPORT_GENERATION_GUIDANCE = {
    "GPT_REPORT_RESPONSE_INVALID": (
        "사진 분석 결과 문구 생성 단계에서 응답 형식 검증에 실패했습니다.",
        "잠시 후 분석 결과 생성을 다시 시도해 주세요.",
    ),
    "GPT_REPORT_GENERATION_FAILED": (
        "사진 분석 결과 문구 생성 단계에서 오류가 발생했습니다.",
        "잠시 후 분석 결과 생성을 다시 시도해 주세요.",
    ),
}


def dorsal_failure_diagnostic(error: BaseException) -> str | None:
    """Map allowlisted dorsal capture/measurement errors to stable codes."""

    message = str(error).casefold()
    if any(
        marker in message
        for marker in (
            "uploaded file is not a readable image",
            "not a supported decodable image",
            "could not decode",
        )
    ):
        return "DORSAL_DECODE_FAILED"
    if "requires an unmodified" in message or (
        "image" in message and "received" in message and "calibration" in message
    ):
        return "DORSAL_DIMENSIONS_MISMATCH"
    if any(
        marker in message
        for marker in (
            "did not produce both feet",
            "segmentation returned an empty mask",
            "did not detect both feet",
            "yolo did not detect a foot in every requested bay",
            "yolo did not detect every requested foot",
            "no foot detected",
            "no feet detected",
        )
    ):
        return "DORSAL_FEET_NOT_DETECTED"
    if any(
        marker in message
        for marker in (
            "implausible total foot length",
            "implausible visible foot length",
            "foot does not reach the heel-side marker reference edge",
            "does not reach the id 4 reference edge",
            "implausible foot length in",
        )
    ):
        return "DORSAL_FOOT_POSITION_INVALID"
    if any(
        marker in message
        for marker in (
            "detected visible foot region is too small",
            "external foot mask is empty in its bay",
            "foot mask is too small for ball width",
            "toe boundary is too small or noisy",
            "visible foot width could not be estimated",
            "implausible foot width",
            "no foot mask found in",
            "foot mask in image_left is too small",
            "foot mask in image_right is too small",
            "toe boundary in image_left is too noisy",
            "toe boundary in image_right is too noisy",
            "reliable straight bottom alignment edge",
            "foot bottom alignment could not be verified",
            "foot bottom alignment did not converge",
            "foot bottom alignment requires an unsafe rotation",
            "foot bottom alignment did not produce an output frame",
        )
    ):
        return "DORSAL_FOOT_OUTLINE_INCOMPLETE"
    if "visible foot mask does not reach the configured ball-width zone" in message:
        return "DORSAL_BALL_WIDTH_ZONE_NOT_VISIBLE"
    if any(
        marker in message
        for marker in (
            "too few stable forefoot cross-sections for ball width",
            "implausible ball width",
            "could not find a stable horizontal ball-width band",
        )
    ):
        return "DORSAL_BALL_WIDTH_UNSTABLE"
    if any(
        marker in message
        for marker in (
            "marker missing",
            "missing marker",
            "marker was not detected",
            "required aruco marker",
            "need at least 3",
            "insufficient marker",
            "not enough marker",
            "missing=[",
        )
    ):
        return "DORSAL_MARKERS_INSUFFICIENT"
    if any(
        marker in message
        for marker in (
            "marker-offset preprocessing failed",
            "homography",
            "duplicate marker",
            "plane rmse",
            "marker layout",
            "canonical id 1 -> id 4 order",
            "marker is too small",
            "marker centre/side ratio",
            "marker scale changes too strongly",
            "marker id 1 is not convex",
            "marker id 4 is not convex",
            "two-marker fit rmse",
            "leave-one-corner-out error",
        )
    ):
        return "DORSAL_ARUCO_GEOMETRY_INVALID"
    return None


def photo_failure_guidance(
    diagnostic_code: str | None,
    image_field: str | None = None,
) -> PhotoFailureGuidance | None:
    """Resolve a stable code to allowlisted Korean copy.

    Unknown codes and invalid field/code combinations intentionally return
    ``None``. This makes it impossible for arbitrary exception strings to
    cross the API or backend failure-notification boundary through this helper.
    """

    if not isinstance(diagnostic_code, str):
        return None
    report_generation = _REPORT_GENERATION_GUIDANCE.get(diagnostic_code)
    if report_generation is not None:
        user_message, recommended_action = report_generation
        return PhotoFailureGuidance(
            diagnostic_code=diagnostic_code,
            user_message=user_message,
            recommended_action=recommended_action,
        )
    upload = _UPLOAD_GUIDANCE.get(diagnostic_code)
    upload_label = _IMAGE_FIELD_LABEL.get(str(image_field))
    if upload is not None and upload_label is not None:
        user_template, action_template = upload
        return PhotoFailureGuidance(
            diagnostic_code=diagnostic_code,
            user_message=user_template.format(label=upload_label),
            recommended_action=action_template.format(label=upload_label),
        )
    dorsal = _DORSAL_GUIDANCE.get(diagnostic_code)
    if dorsal is not None:
        user_message, recommended_action = dorsal
        return PhotoFailureGuidance(
            diagnostic_code=diagnostic_code,
            user_message=user_message,
            recommended_action=recommended_action,
        )

    side_specific = _SIDE_SPECIFIC_GUIDANCE.get(diagnostic_code)
    if side_specific is not None:
        user_message, recommended_action = side_specific
        return PhotoFailureGuidance(
            diagnostic_code=diagnostic_code,
            user_message=user_message,
            recommended_action=recommended_action,
        )

    common = _PLANTAR_COMMON_GUIDANCE.get(diagnostic_code)
    side = _PLANTAR_FIELD_SIDE.get(str(image_field))
    if common is None or side is None:
        return None
    user_template, action_template = common
    return PhotoFailureGuidance(
        diagnostic_code=diagnostic_code,
        user_message=user_template.format(side=side),
        recommended_action=action_template.format(side=side),
    )


def photo_failure_detail(
    diagnostic_code: str | None,
    image_field: str | None = None,
) -> str | None:
    """Return only pre-approved text suitable for backend ``failureDetail``."""

    guidance = photo_failure_guidance(diagnostic_code, image_field)
    return guidance.failure_detail if guidance is not None else None
