from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import cv2
import httpx
import numpy as np
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.routes import photo_analysis as route
from app.main import app
from app.services.hallux_valgus_analysis import HalluxValgusAnalysisResult
from app.services.integrated_foot_analysis import (
    FootGeometryResult,
    IntegratedFootAnalysisResult,
    PreparedFoot,
)
from app.services import photo_capture_preflight as preflight_service
from app.services.report_text_generation import FullPhotoAnalysisTexts
from app.services.tinea_analysis import TineaAnalysisResult
from app.services.tinea_analysis import AnalysisError
from app.services.photo_capture_status import PhotoCaptureStatusOutcome


ENDPOINT = "/api/reports/photo-analysis"
_INPUT_COMPLETED_STAGES = (
    "input-read-dorsal",
    "input-read-left-plantar",
    "input-read-right-plantar",
    "capture-preflight",
    "dorsal-measurement-preflight",
    "capture-status-transition",
)


async def _confirm_photo_capture_passed(**kwargs):
    attempt = kwargs["photo_capture_attempt"]
    return PhotoCaptureStatusOutcome(
        confirmed=True, status="WAITING_FOR_ENVIRONMENT",
        recapture_count=attempt, max_recaptures=3,
        photo_capture_attempt=attempt, backend_status_code=200,
    )


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise AssertionError("Could not encode a photo preflight fixture.")
    return encoded.tobytes()


def _valid_dorsal_capture() -> bytes:
    """Build a decodable 1280x720 frame with the ID 1/4 ruler."""

    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (x, y) in ((1, (620, 135)), (4, (620, 525))):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 40)
        image[y : y + 40, x : x + 40] = cv2.cvtColor(
            marker,
            cv2.COLOR_GRAY2BGR,
        )
    cv2.ellipse(image, (430, 360), (100, 245), 0, 0, 360, (75, 75, 75), -1)
    cv2.ellipse(image, (850, 360), (100, 245), 0, 0, 360, (75, 75, 75), -1)
    return _encode_jpeg(image)


def _valid_plantar_capture(*, mirror: bool = False) -> bytes:
    """Build a visibly non-blank plantar-shaped capture."""

    image = np.full((640, 480, 3), 245, dtype=np.uint8)
    center_x = 255 if mirror else 225
    cv2.ellipse(
        image,
        (center_x, 350),
        (92, 220),
        -8 if mirror else 8,
        0,
        360,
        (65, 85, 110),
        -1,
    )
    for index in range(5):
        toe_x = center_x + (index - 2) * 30
        cv2.circle(image, (toe_x, 125 + abs(index - 2) * 6), 25, (65, 85, 110), -1)
    return _encode_jpeg(image)


VALID_DORSAL_CAPTURE = _valid_dorsal_capture()
VALID_LEFT_PLANTAR_CAPTURE = _valid_plantar_capture()
VALID_RIGHT_PLANTAR_CAPTURE = _valid_plantar_capture(mirror=True)


def execute_photo_pipeline_for_test(
    *,
    output_dir: Path,
    request_id: str = "photo-request",
    measurement_session_id: int = 34,
    photo_capture_attempt: int = 0,
    dorsal_bytes: bytes = b"raw-dorsal",
    left_plantar_bytes: bytes = b"raw-left-plantar",
    right_plantar_bytes: bytes = b"raw-right-plantar",
    authorization: str = "Bearer photo-token",
) -> httpx.Response:
    """Run the background worker body while preserving HTTP-style assertions."""

    work_item = route.PhotoAnalysisWorkItem(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        photo_capture_attempt=photo_capture_attempt,
        output_dir=output_dir,
        dorsal_bytes=dorsal_bytes,
        left_plantar_bytes=left_plantar_bytes,
        right_plantar_bytes=right_plantar_bytes,
        dorsal_filename="dorsal.jpg",
        authorization=authorization,
        completed_stages=_INPUT_COMPLETED_STAGES,
    )
    try:
        response = asyncio.run(route._execute_photo_pipeline(work_item))
    except HTTPException as exc:
        return httpx.Response(
            status_code=exc.status_code,
            json={"detail": jsonable_encoder(exc.detail)},
            headers=exc.headers,
        )
    return httpx.Response(
        status_code=response.status_code,
        content=response.body,
        headers=dict(response.headers),
    )


def _queued_work_item(
    measurement_session_id: int,
    *,
    request_id: str,
    photo_capture_attempt: int = 0,
) -> route.PhotoAnalysisWorkItem:
    return route.PhotoAnalysisWorkItem(
        request_id=request_id,
        measurement_session_id=measurement_session_id,
        photo_capture_attempt=photo_capture_attempt,
        output_dir=Path("photo-analysis-async-test-output") / request_id,
        dorsal_bytes=b"raw-dorsal",
        left_plantar_bytes=b"raw-left-plantar",
        right_plantar_bytes=b"raw-right-plantar",
        dorsal_filename="dorsal.jpg",
        authorization="Bearer photo-token",
        completed_stages=_INPUT_COMPLETED_STAGES,
    )


def _tinea(label: str, fungal: int, skin: int) -> TineaAnalysisResult:
    return TineaAnalysisResult(
        suspicion_map_png=f"{label}-map".encode(),
        photo_overlay_png=f"{label}-overlay".encode(),
        original_filename=f"{label}.png",
        fungal_safety_score=fungal,
        skin_reaction_safety_score=skin,
        metrics={
            "score_reliability": "reliable",
            "fungal_regions": [],
            "inflammation_regions": [],
        },
        clean_original_png=f"{label}-clean".encode(),
    )


def _neutral_plantar_tinea(label: str) -> TineaAnalysisResult:
    clean = f"{label}-clean".encode()
    return TineaAnalysisResult(
        suspicion_map_png=clean,
        photo_overlay_png=clean,
        original_filename=f"{label}.png",
        fungal_safety_score=100,
        skin_reaction_safety_score=100,
        metrics={
            "analysis_mode": "segformer_fungal_only_segformer_wound",
            "fungal_analysis_enabled": True,
            "fungal_model": "segformer_b2_teacher_student",
            "sam_inflammation_class_used": False,
            "inflammation_analysis_enabled": True,
            "inflammation_analysis_mode": "wound_segformer_single_pass",
            "wound_model": "wound_segformer_b0",
            "wound_detected": False,
            "model_inference_executed": True,
            "score_reliability": "normal",
            "fungal_regions": [],
            "inflammation_regions": [],
        },
        clean_original_png=clean,
        authoritative_fungal_mask=np.zeros((8, 8), dtype=bool),
    )


def _integrated(*, measurement_valid: bool = True) -> IntegratedFootAnalysisResult:
    mask = np.full((8, 8), 255, np.uint8)
    feet = {
        "left": PreparedFoot(
            anatomical_side="left",
            board_side="image_right",
            image_png=b"dorsal-left-foot",
            mask=mask,
            length_mm=253.0,
            ball_width_mm=85.0,
            segmentation_confidence=0.98,
            length_details={},
            ball_width_details={},
        ),
        "right": PreparedFoot(
            anatomical_side="right",
            board_side="image_left",
            image_png=b"dorsal-right-foot",
            mask=mask,
            length_mm=248.0,
            ball_width_mm=70.0,
            segmentation_confidence=0.97,
            length_details={},
            ball_width_details={},
        ),
    }
    geometry = FootGeometryResult(
        original_filename="dorsal.jpg",
        input_width=1280,
        input_height=720,
        measurement_valid=measurement_valid,
        measurement_status=("valid" if measurement_valid else "invalid"),
        measurement_invalid_reasons=(() if measurement_valid else ("bad_markers",)),
        orientation_transform="identity",
        detected_marker_ids=((1, 4) if measurement_valid else (1,)),
        missing_marker_ids=(() if measurement_valid else (4,)),
        lens_correction={"applied": True},
        global_calibration={},
        feet=feet,
    )
    return IntegratedFootAnalysisResult(
        geometry=geometry,
        tinea={
            "left": _tinea("dorsal-left", 93, 91),
            "right": _tinea("dorsal-right", 89, 94),
        },
        hallux_valgus={
            "left": HalluxValgusAnalysisResult(
                analysis_png=b"hallux-left-png",
                original_filename="left.png",
                angle_degree=12.0,
                analysis_text="left",
                score=0.9,
                metrics={},
            ),
            "right": HalluxValgusAnalysisResult(
                analysis_png=b"hallux-right-png",
                original_filename="right.png",
                angle_degree=18.0,
                analysis_text="right",
                score=0.8,
                metrics={},
            ),
        },
    )


def _plantar(output_dir: Path, side: str, count: int, ratio: float):
    return SimpleNamespace(
        visualization_path=output_dir / f"{side}_visualization.png",
        tinea_input_path=output_dir / f"{side}_tinea.png",
        sole_mask_path=output_dir / f"{side}_mask.png",
        region_count=count,
        contact_area_ratio=ratio,
    )


TEXTS = FullPhotoAnalysisTexts(
    hallux_score_analysis_text="무지외반 정렬 경향 설명",
    fungal_suspicion_safety_description="무좀 의심 경향 설명",
    skin_reaction_safety_description="피부 자극 경향 설명",
    total_score_description="발등과 발바닥 종합 설명",
    plantar_footprint_analysis_text="왼발과 오른발은 정상 형태입니다.",
    metric_advice={
        "HALLUX_VALGUS": ("무지 정렬 관찰 문장", "편한 앞볼 신발 권장 문장"),
        "ATHLETES_FOOT": ("무좀 의심 관찰 문장", "건조 관리 권장 문장"),
        "SKIN_IRRITATION": ("피부 자극 관찰 문장", "마찰 관리 권장 문장"),
    },
    generation_source="OPENAI",
)


class _Backend:
    def __init__(self, *, status_by_url: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.status_by_url = status_by_url or {}
        self.analysis_done = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url: str, **kwargs):
        if not self.analysis_done:
            raise AssertionError("backend POST started before local/GPT analysis completed")
        self.calls.append((url, kwargs))
        status_code = self.status_by_url.get(url, 201)
        if url == route.settings.hallux_valgus_report_endpoint:
            body = {"result": {"riskScore": 76.5}}
        elif url == route.settings.summary_report_endpoint:
            body = {"metricType": kwargs["json"]["metricType"]}
        else:
            body = {"saved": True}
        return httpx.Response(
            status_code,
            json=body,
            headers={"content-type": "application/json"},
        )


class PhotoAnalysisAsyncIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        # The developer's local diagnostics opt-in must not write test uploads
        # into the real failed-capture archive. Dedicated tests opt in explicitly.
        diagnostic_capture = patch.object(route.settings, "photo_capture_diagnostics_enabled", False)
        diagnostic_capture.start()
        self.addCleanup(diagnostic_capture.stop)
        self.client = TestClient(app)
        route._active_photo_analysis_sessions.clear()
        self.prepared_geometry = _integrated().geometry
        # Capture decoding/ArUco inspection has a separate service boundary.
        # Intake tests pin the HTTP and scheduling contract without loading
        # optional hardware runtimes or full photo models.
        self.preflight_patcher = patch.object(
            route,
            "validate_photo_capture_preflight",
            return_value=SimpleNamespace(
                dorsal_width=1280,
                dorsal_height=720,
                detected_marker_count=2,
                dorsal_detected_foot_count=2,
                left_plantar_foot_confidence=0.91,
                right_plantar_foot_confidence=0.92,
                left_plantar_width=480,
                left_plantar_height=640,
                right_plantar_width=480,
                right_plantar_height=640,
            ),
        )
        self.preflight_validator = self.preflight_patcher.start()
        self.measurement_preflight_patcher = patch.object(
            route,
            "validate_dorsal_measurement_preflight",
            return_value=self.prepared_geometry,
        )
        self.measurement_preflight_validator = (
            self.measurement_preflight_patcher.start()
        )
        self.capture_passed_patcher = patch.object(
            route, "notify_photo_capture_passed",
            new=AsyncMock(side_effect=_confirm_photo_capture_passed),
        )
        self.capture_passed_callback = self.capture_passed_patcher.start()

    def tearDown(self) -> None:
        self.capture_passed_patcher.stop()
        self.measurement_preflight_patcher.stop()
        self.preflight_patcher.stop()
        route._active_photo_analysis_sessions.clear()
        self.client.close()

    def _post(
        self,
        *,
        measurement_session_id: int = 34,
        photo_capture_attempt: int = 0,
        dorsal_bytes: bytes | None = None,
        left_plantar_bytes: bytes | None = None,
        right_plantar_bytes: bytes | None = None,
    ) -> httpx.Response:
        dorsal_content = (
            VALID_DORSAL_CAPTURE if dorsal_bytes is None else dorsal_bytes
        )
        left_plantar_content = (
            VALID_LEFT_PLANTAR_CAPTURE
            if left_plantar_bytes is None
            else left_plantar_bytes
        )
        right_plantar_content = (
            VALID_RIGHT_PLANTAR_CAPTURE
            if right_plantar_bytes is None
            else right_plantar_bytes
        )
        return self.client.post(
            ENDPOINT,
            data={"measurementSessionId": str(measurement_session_id), "photoCaptureAttempt": str(photo_capture_attempt)},
            files={
                "dorsalFootImage": (
                    "dorsal.jpg",
                    dorsal_content,
                    "image/jpeg",
                ),
                "leftPlantarFootImage": (
                    "left.jpg",
                    left_plantar_content,
                    "image/jpeg",
                ),
                "rightPlantarFootImage": (
                    "right.jpg",
                    right_plantar_content,
                    "image/jpeg",
                ),
            },
            headers={"Authorization": "Bearer photo-token"},
        )

    def test_valid_upload_returns_202_and_registers_owned_work_item(self) -> None:
        output_dir = Path("photo-analysis-async-test-output") / "queued-request"
        with (
            patch.object(
                route,
                "new_request_dir",
                return_value=("queued-request", output_dir),
            ),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            self.assertLogs(route.progress_logger.name, level="INFO") as captured,
        ):
            response = self._post()

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json(),
            {
                "accepted": True,
                "status": "QUEUED",
                "captureValidationStatus": "PASSED",
                "mayStartPressure": True,
                "measurementSessionId": 34,
                "photoCaptureAttempt": 0,
                "measurementStatus": "WAITING_FOR_ENVIRONMENT",
                "statusCallbackConfirmed": True,
                "pipeline": route.PIPELINE_VERSION,
                "requestId": "queued-request",
            },
        )
        self.assertEqual(
            response.headers["x-feetfit-pipeline"], route.PIPELINE_VERSION
        )
        self.assertEqual(
            response.headers["x-feetfit-request-id"], "queued-request"
        )
        add_task.assert_called_once()
        self.preflight_validator.assert_called_once_with(
            VALID_DORSAL_CAPTURE,
            VALID_LEFT_PLANTAR_CAPTURE,
            VALID_RIGHT_PLANTAR_CAPTURE,
        )
        self.measurement_preflight_validator.assert_called_once_with(
            VALID_DORSAL_CAPTURE,
            "dorsal.jpg",
        )
        scheduled_runner, work_item = add_task.call_args.args
        self.assertIs(scheduled_runner, route._run_photo_analysis_job)
        self.assertIsInstance(work_item, route.PhotoAnalysisWorkItem)
        self.assertEqual(work_item.measurement_session_id, 34)
        self.assertIs(work_item.prepared_geometry, self.prepared_geometry)
        self.assertEqual(work_item.request_id, "queued-request")
        self.assertEqual(work_item.output_dir, output_dir)
        self.assertEqual(work_item.dorsal_bytes, VALID_DORSAL_CAPTURE)
        self.assertEqual(
            work_item.left_plantar_bytes,
            VALID_LEFT_PLANTAR_CAPTURE,
        )
        self.assertEqual(
            work_item.right_plantar_bytes,
            VALID_RIGHT_PLANTAR_CAPTURE,
        )
        self.assertEqual(work_item.authorization, "Bearer photo-token")
        self.assertEqual(work_item.completed_stages, _INPUT_COMPLETED_STAGES)

        marker = "pipeline_progress "
        payloads = [
            json.loads(message.split(marker, 1)[1])
            for record in captured.records
            if marker in (message := record.getMessage())
        ]
        self.assertEqual(
            {
                item["stage"]: item["details"]["byteCount"]
                for item in payloads
                if item["event"] == "input_read_completed"
            },
            {
                "input-read-dorsal": len(VALID_DORSAL_CAPTURE),
                "input-read-left-plantar": len(VALID_LEFT_PLANTAR_CAPTURE),
                "input-read-right-plantar": len(VALID_RIGHT_PLANTAR_CAPTURE),
            },
        )
        capture_preflight = next(
            item
            for item in payloads
            if item["event"] == "capture_preflight_passed"
        )
        self.assertEqual(capture_preflight["details"]["detectedMarkerCount"], 2)
        self.assertEqual(capture_preflight["details"]["requiredMarkerCount"], 2)
        self.assertEqual(
            [item["event"] for item in payloads if item["stage"] == "queue"],
            ["job_queued"],
        )
        self.assertEqual(
            [
                (item["stage"], item["event"])
                for item in payloads
                if item["stage"]
                in {
                    "capture-preflight",
                    "dorsal-measurement-preflight",
                    "queue",
                }
            ],
            [
                ("capture-preflight", "capture_preflight_started"),
                ("capture-preflight", "capture_preflight_passed"),
                (
                    "dorsal-measurement-preflight",
                    "measurement_preflight_started",
                ),
                (
                    "dorsal-measurement-preflight",
                    "measurement_preflight_passed",
                ),
                ("queue", "job_queued"),
            ],
        )
        queued = next(item for item in payloads if item["event"] == "job_queued")
        self.assertEqual(
            queued["message"],
            "사진 분석 요청을 접수했습니다. 백그라운드 분석 대기열에 등록했습니다.",
        )
        self.assertTrue(
            all(
                record.getMessage().startswith(
                    "[사진 분석][세션 34][요청 queued-r] "
                )
                for record in captured.records
            )
        )

    def test_capture_then_measurement_then_queue_before_any_main_analysis(self) -> None:
        trace: list[str] = []
        preflight_result = self.preflight_validator.return_value

        def capture_preflight(*args) -> object:
            self.assertEqual(
                args,
                (
                    VALID_DORSAL_CAPTURE,
                    VALID_LEFT_PLANTAR_CAPTURE,
                    VALID_RIGHT_PLANTAR_CAPTURE,
                ),
            )
            trace.append("capture-preflight")
            return preflight_result

        def measurement_preflight(*args) -> FootGeometryResult:
            self.assertEqual(trace, ["capture-preflight"])
            self.assertEqual(args, (VALID_DORSAL_CAPTURE, "dorsal.jpg"))
            trace.append("dorsal-measurement-preflight")
            return self.prepared_geometry

        async def confirm_passed(**kwargs):
            self.assertEqual(trace, ["capture-preflight", "dorsal-measurement-preflight"])
            trace.append("capture-status-transition")
            return await _confirm_photo_capture_passed(**kwargs)

        def enqueue(*args, **kwargs) -> None:
            self.assertEqual(
                trace,
                ["capture-preflight", "dorsal-measurement-preflight", "capture-status-transition"],
            )
            self.assertIs(args[0], route._run_photo_analysis_job)
            self.assertIs(args[1].prepared_geometry, self.prepared_geometry)
            trace.append("queue")

        self.preflight_validator.side_effect = capture_preflight
        self.measurement_preflight_validator.side_effect = measurement_preflight
        self.capture_passed_callback.side_effect = confirm_passed
        with (
            patch.object(route.BackgroundTasks, "add_task", side_effect=enqueue),
            patch.object(route, "analyze_integrated_foot_photo") as integrated,
            patch.object(route, "analyze_prepared_foot_geometry") as prepared,
            patch.object(route, "analyze_plantar_image") as plantar,
            patch.object(route, "build_plantar_skin_result") as skin,
            patch.object(
                route,
                "generate_full_photo_analysis_texts",
                new=AsyncMock(),
            ) as generate,
        ):
            response = self._post()

        self.assertEqual(response.status_code, 202, response.text)
        self.assertTrue(response.json()["mayStartPressure"])
        self.assertEqual(
            trace,
            ["capture-preflight", "dorsal-measurement-preflight", "capture-status-transition", "queue"],
        )
        integrated.assert_not_called()
        prepared.assert_not_called()
        plantar.assert_not_called()
        skin.assert_not_called()
        generate.assert_not_awaited()

    def test_capture_preflight_rejections_notify_recapture_only_for_aruco(self) -> None:
        wrong_size = cv2.imdecode(
            np.frombuffer(VALID_DORSAL_CAPTURE, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(wrong_size)
        wrong_size = cv2.resize(wrong_size, (640, 360))
        dorsal_without_markers = np.full((720, 1280, 3), 235, dtype=np.uint8)
        cv2.ellipse(
            dorsal_without_markers,
            (640, 360),
            (180, 270),
            0,
            0,
            360,
            (60, 80, 100),
            -1,
        )
        blank_plantar = _encode_jpeg(
            np.full((640, 480, 3), 255, dtype=np.uint8)
        )
        cases = (
            (
                "decode",
                {"dorsal_bytes": b"not-a-decodable-image"},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="DORSAL_DECODE_FAILED",
                    image_field="dorsalFootImage",
                ),
            ),
            (
                "dimensions",
                {"dorsal_bytes": _encode_jpeg(wrong_size)},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="DORSAL_DIMENSIONS_MISMATCH",
                    image_field="dorsalFootImage",
                    actual_width=640,
                    actual_height=360,
                    expected_width=1280,
                    expected_height=720,
                ),
            ),
            (
                "markers",
                {"dorsal_bytes": _encode_jpeg(dorsal_without_markers)},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="DORSAL_MARKERS_INSUFFICIENT",
                    image_field="dorsalFootImage",
                    actual_width=1280,
                    actual_height=720,
                    detected_marker_count=2,
                    required_marker_count=2,
                ),
            ),
            (
                "dorsal-feet",
                {},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="DORSAL_FEET_NOT_DETECTED",
                    image_field="dorsalFootImage",
                    actual_width=1280,
                    actual_height=720,
                    detected_foot_count=1,
                    required_foot_count=2,
                ),
            ),
            (
                "left-plantar-blank",
                {"left_plantar_bytes": blank_plantar},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="PLANTAR_CAPTURE_NEAR_WHITE",
                    image_field="leftPlantarFootImage",
                    actual_width=480,
                    actual_height=640,
                ),
            ),
            (
                "right-plantar-blank",
                {"right_plantar_bytes": blank_plantar},
                preflight_service.PhotoCapturePreflightError(
                    diagnostic_code="PLANTAR_CAPTURE_NEAR_WHITE",
                    image_field="rightPlantarFootImage",
                    actual_width=480,
                    actual_height=640,
                ),
            ),
        )

        for label, upload_override, preflight_error in cases:
            with self.subTest(case=label):
                route._active_photo_analysis_sessions.clear()
                with (
                    patch.object(route.BackgroundTasks, "add_task") as add_task,
                    patch.object(
                        route, "notify_measurement_recapture_required",
                        new=AsyncMock(return_value=route.MeasurementRecaptureOutcome(
                            confirmed=True, status="WAITING_FOR_RECAPTURE",
                            recapture_count=0, max_recaptures=3,
                        )),
                    ) as recapture_notifier,
                    patch.object(
                        route,
                        "_notify_photo_failure",
                        new=AsyncMock(),
                    ) as notifier,
                    patch.object(
                        route,
                        "_reserve_photo_analysis_session",
                        new=AsyncMock(),
                    ) as reserve,
                    patch.object(
                        route,
                        "_release_photo_analysis_session",
                        new=AsyncMock(),
                    ) as release,
                    patch.object(
                        route,
                        "validate_photo_capture_preflight",
                        side_effect=preflight_error,
                    ),
                    patch.object(
                        route,
                        "analyze_integrated_foot_photo",
                    ) as integrated,
                    patch.object(
                        route,
                        "analyze_prepared_foot_geometry",
                    ) as prepared,
                    patch.object(route, "analyze_plantar_image") as plantar,
                    patch.object(route, "build_plantar_skin_result") as skin,
                    patch.object(
                        route,
                        "generate_full_photo_analysis_texts",
                        new=AsyncMock(),
                    ) as generate,
                ):
                    response = self._post(**upload_override)

                self.assertEqual(response.status_code, 422, response.text)
                detail = response.json()["detail"]
                self.assertEqual(detail["stage"], "capture-preflight")
                self.assertEqual(
                    detail["diagnosticCode"],
                    preflight_error.diagnostic_code,
                )
                self.assertEqual(detail["imageField"], preflight_error.image_field)
                self.assertEqual(
                    detail.get("detectedFootCount"),
                    preflight_error.detected_foot_count,
                )
                self.assertEqual(
                    detail.get("requiredFootCount"),
                    preflight_error.required_foot_count,
                )
                self.assertRegex(detail["diagnosticCode"], r"^[A-Z][A-Z0-9_]+$")
                self.assertTrue(detail["recaptureRequired"])
                self.assertFalse(detail["mayStartPressure"])
                is_aruco = preflight_error.diagnostic_code in route.ARUCO_RECAPTURE_DIAGNOSTIC_CODES
                self.assertEqual(detail["retryable"], is_aruco)
                guidance = route.photo_failure_guidance(
                    preflight_error.diagnostic_code,
                    preflight_error.image_field,
                )
                self.assertEqual(detail["message"], guidance.user_message)
                self.assertEqual(
                    detail["recommendedAction"],
                    guidance.recommended_action,
                )
                self.assertFalse(detail["partialSave"])
                self.assertEqual(detail["reports"], {})
                self.assertEqual(
                    response.headers["x-feetfit-pipeline"],
                    route.PIPELINE_VERSION,
                )
                self.assertTrue(response.headers["x-feetfit-request-id"])
                add_task.assert_not_called()
                self.measurement_preflight_validator.assert_not_called()
                integrated.assert_not_called()
                prepared.assert_not_called()
                plantar.assert_not_called()
                skin.assert_not_called()
                generate.assert_not_awaited()
                if is_aruco:
                    notifier.assert_not_awaited()
                    recapture_notifier.assert_awaited_once_with(
                        measurement_session_id=34,
                        photo_capture_attempt=0,
                        authorization="Bearer photo-token",
                        diagnostic_code=preflight_error.diagnostic_code,
                        request_id=unittest.mock.ANY,
                    )
                    self.assertEqual(detail["measurementStatus"], "WAITING_FOR_RECAPTURE")
                    self.assertEqual(detail["nextMeasurementStatus"], "READY_FOR_RECAPTURE")
                else:
                    recapture_notifier.assert_not_awaited()
                    notifier.assert_awaited_once_with(
                        measurement_session_id=34,
                        photo_capture_attempt=0,
                        authorization="Bearer photo-token",
                        request_id=unittest.mock.ANY,
                        failure_reason=(
                            "INVALID_CAPTURE_DATA" if preflight_error.diagnostic_code in {
                                "DORSAL_MARKERS_INSUFFICIENT", "DORSAL_ARUCO_GEOMETRY_INVALID",
                            } else "CAMERA_ERROR"
                        ),
                        stage="capture-preflight",
                        error_type="PhotoCapturePreflightError",
                        diagnostic_code=preflight_error.diagnostic_code,
                        image_field=preflight_error.image_field,
                    )

                reserve.assert_awaited_once_with(
                    measurement_session_id=34,
                    request_id=unittest.mock.ANY,
                    completed_stages=list(_INPUT_COMPLETED_STAGES[:-3]),
                )
                release.assert_awaited_once_with(34)
                self.assertEqual(route._active_photo_analysis_sessions, set())
                self.assertNotIn("photo-token", response.text)
                self.assertNotIn("not-a-decodable-image", response.text)

    def test_measurement_readiness_failure_is_rejected_before_202(self) -> None:
        failure = preflight_service.PhotoCapturePreflightError(
            diagnostic_code="DORSAL_FOOT_POSITION_INVALID",
            image_field="dorsalFootImage",
        )
        with (
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as notifier,
            patch.object(
                route,
                "validate_dorsal_measurement_preflight",
                side_effect=failure,
            ),
            patch.object(route, "analyze_integrated_foot_photo") as integrated,
            patch.object(route, "analyze_prepared_foot_geometry") as prepared,
            patch.object(route, "analyze_plantar_image") as plantar,
            patch.object(route, "build_plantar_skin_result") as skin,
            patch.object(
                route,
                "generate_full_photo_analysis_texts",
                new=AsyncMock(),
            ) as generate,
        ):
            response = self._post(measurement_session_id=83)

        self.assertEqual(response.status_code, 422, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["stage"], "dorsal-measurement-preflight")
        self.assertEqual(
            detail["diagnosticCode"],
            "DORSAL_FOOT_POSITION_INVALID",
        )
        self.assertTrue(detail["recaptureRequired"])
        self.assertFalse(detail["mayStartPressure"])
        add_task.assert_not_called()
        self.preflight_validator.assert_called_once_with(
            VALID_DORSAL_CAPTURE,
            VALID_LEFT_PLANTAR_CAPTURE,
            VALID_RIGHT_PLANTAR_CAPTURE,
        )
        integrated.assert_not_called()
        prepared.assert_not_called()
        plantar.assert_not_called()
        skin.assert_not_called()
        generate.assert_not_awaited()
        notifier.assert_awaited_once_with(
            measurement_session_id=83,
            photo_capture_attempt=0,
            authorization="Bearer photo-token",
            request_id=unittest.mock.ANY,
            failure_reason="CAMERA_ERROR",
            stage="dorsal-measurement-preflight",
            error_type="PhotoCapturePreflightError",
            diagnostic_code="DORSAL_FOOT_POSITION_INVALID",
            image_field="dorsalFootImage",
        )

    def test_empty_or_oversized_upload_never_schedules_a_job(self) -> None:
        with (
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as empty_notifier,
            self.assertLogs(route.progress_logger.name, level="INFO") as empty_logs,
        ):
            empty_response = self._post(dorsal_bytes=b"")
        self.assertEqual(empty_response.status_code, 400, empty_response.text)
        empty_detail = empty_response.json()["detail"]
        self.assertEqual(empty_detail["diagnosticCode"], "PHOTO_UPLOAD_EMPTY")
        self.assertEqual(empty_detail["message"], "발등 사진 데이터가 비어 있습니다.")
        add_task.assert_not_called()
        empty_notifier.assert_awaited_once_with(
            measurement_session_id=34,
            photo_capture_attempt=0,
            authorization="Bearer photo-token",
            request_id=unittest.mock.ANY,
            failure_reason="CAMERA_ERROR",
            stage="input-read-dorsal",
            error_type="UploadValidationError",
            diagnostic_code="PHOTO_UPLOAD_EMPTY",
            image_field="dorsalFootImage",
        )
        self.assertEqual(route._active_photo_analysis_sessions, set())
        empty_log_text = "\n".join(record.getMessage() for record in empty_logs.records)
        self.assertIn('"event":"upload_rejected"', empty_log_text)
        self.assertIn("발등 사진 데이터가 비어 있습니다.", empty_log_text)

        with (
            patch.object(route.settings, "max_upload_size_bytes", 3),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as oversized_notifier,
            self.assertLogs(route.progress_logger.name, level="INFO") as oversized_logs,
        ):
            oversized_response = self._post(dorsal_bytes=b"four")
        self.assertEqual(
            oversized_response.status_code,
            413,
            oversized_response.text,
        )
        oversized_detail = oversized_response.json()["detail"]
        self.assertEqual(
            oversized_detail["diagnosticCode"],
            "PHOTO_UPLOAD_TOO_LARGE",
        )
        self.assertEqual(
            oversized_detail["message"],
            "발등 사진 파일 용량이 허용 범위를 초과했습니다.",
        )
        add_task.assert_not_called()
        oversized_notifier.assert_awaited_once_with(
            measurement_session_id=34,
            photo_capture_attempt=0,
            authorization="Bearer photo-token",
            request_id=unittest.mock.ANY,
            failure_reason="CAMERA_ERROR",
            stage="input-read-dorsal",
            error_type="UploadValidationError",
            diagnostic_code="PHOTO_UPLOAD_TOO_LARGE",
            image_field="dorsalFootImage",
        )
        self.assertEqual(route._active_photo_analysis_sessions, set())
        oversized_log_text = "\n".join(
            record.getMessage() for record in oversized_logs.records
        )
        self.assertIn('"event":"upload_rejected"', oversized_log_text)
        self.assertIn(
            "발등 사진 파일 용량이 허용 범위를 초과했습니다.",
            oversized_log_text,
        )

    def test_unexpected_upload_read_failure_notifies_ai_server_error(self) -> None:
        with (
            patch.object(
                route,
                "_read_upload",
                new=AsyncMock(side_effect=OSError("private raw message")),
            ),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as notifier,
        ):
            response = self._post()

        self.assertEqual(response.status_code, 500, response.text)
        add_task.assert_not_called()
        notifier.assert_awaited_once_with(
            measurement_session_id=34,
            photo_capture_attempt=0,
            authorization="Bearer photo-token",
            request_id=unittest.mock.ANY,
            failure_reason="AI_SERVER_ERROR",
            stage="input-read-dorsal",
            error_type="OSError",
        )

    def test_background_schedule_failure_notifies_once_and_preserves_500(self) -> None:
        with (
            patch.object(
                route.BackgroundTasks,
                "add_task",
                side_effect=RuntimeError("private scheduling message"),
            ),
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as notifier,
        ):
            response = self._post(measurement_session_id=39)

        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["detail"]["stage"], "background-schedule")
        notifier.assert_awaited_once_with(
            measurement_session_id=39,
            photo_capture_attempt=0,
            authorization="Bearer photo-token",
            request_id=unittest.mock.ANY,
            failure_reason="AI_SERVER_ERROR",
            stage="background-schedule",
            error_type="RuntimeError",
        )
        self.assertNotIn(39, route._active_photo_analysis_sessions)

    def test_duplicate_active_session_returns_409_without_second_job(self) -> None:
        with (
            patch.object(
                route,
                "new_request_dir",
                side_effect=[
                    ("first-request", Path("photo-analysis-async-test-output/first")),
                    ("duplicate-request", Path("photo-analysis-async-test-output/duplicate")),
                ],
            ),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as notifier,
        ):
            first = self._post(measurement_session_id=41)
            duplicate = self._post(measurement_session_id=41)

        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        detail = duplicate.json()["detail"]
        self.assertEqual(detail["stage"], "request-reservation")
        self.assertEqual(detail["errorType"], "DuplicatePhotoAnalysisError")
        self.assertFalse(detail["retryable"])
        self.assertEqual(add_task.call_count, 1)
        notifier.assert_not_awaited()
        self.assertEqual(route._active_photo_analysis_sessions, {41})

    def test_fourth_reserved_job_returns_503_and_retry_after(self) -> None:
        request_dirs = [
            (f"queue-{index}", Path(f"photo-analysis-async-test-output/queue-{index}"))
            for index in range(1, 5)
        ]
        with (
            patch.object(route, "new_request_dir", side_effect=request_dirs),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
            patch.object(
                route,
                "_notify_photo_failure",
                new=AsyncMock(),
            ) as notifier,
        ):
            accepted = [
                self._post(measurement_session_id=session_id)
                for session_id in (51, 52, 53)
            ]
            full = self._post(measurement_session_id=54)

        self.assertEqual([item.status_code for item in accepted], [202, 202, 202])
        self.assertEqual(full.status_code, 503, full.text)
        self.assertEqual(full.headers["retry-after"], "5")
        detail = full.json()["detail"]
        self.assertEqual(detail["errorType"], "PhotoAnalysisQueueFullError")
        self.assertTrue(detail["retryable"])
        self.assertEqual(add_task.call_count, 3)
        notifier.assert_not_awaited()
        self.assertEqual(route._active_photo_analysis_sessions, {51, 52, 53})

    def test_background_runner_serializes_two_photo_pipelines(self) -> None:
        async def exercise() -> tuple[int, int, int]:
            active = 0
            max_active = 0

            async def execute(work_item):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                try:
                    await asyncio.sleep(0.01)
                    return JSONResponse(status_code=200, content={"ok": True})
                finally:
                    active -= 1

            sessions = {61, 62}
            with (
                patch.object(
                    route,
                    "_photo_analysis_worker_semaphore",
                    asyncio.Semaphore(1),
                ),
                patch.object(
                    route,
                    "_photo_analysis_reservation_lock",
                    threading.Lock(),
                ),
                patch.object(route, "_active_photo_analysis_sessions", sessions),
                patch.object(
                    route,
                    "_execute_photo_pipeline",
                    new=AsyncMock(side_effect=execute),
                ) as executor,
                patch.object(
                    route,
                    "_notify_photo_job_failure",
                    new=AsyncMock(),
                ) as notifier,
            ):
                await asyncio.gather(
                    route._run_photo_analysis_job(
                        _queued_work_item(61, request_id="serial-1")
                    ),
                    route._run_photo_analysis_job(
                        _queued_work_item(62, request_id="serial-2")
                    ),
                )
                return max_active, executor.await_count, notifier.await_count

        max_active, execution_count, notification_count = asyncio.run(exercise())
        self.assertEqual(max_active, 1)
        self.assertEqual(execution_count, 2)
        self.assertEqual(notification_count, 0)

    def test_background_runner_maps_terminal_outcomes_and_releases_session(self) -> None:
        cases = (
            (
                "success",
                JSONResponse(status_code=200, content={"ok": True}),
                None,
            ),
            (
                "capture-failure",
                HTTPException(
                    status_code=422,
                    detail={
                        "stage": "measurement-validation",
                        "errorType": "MeasurementValidationError",
                        "diagnosticCode": "DORSAL_MARKERS_INSUFFICIENT",
                        "imageField": "dorsalFootImage",
                    },
                ),
                {
                    "failure_reason": "INVALID_CAPTURE_DATA",
                    "stage": "measurement-validation",
                    "error_type": "MeasurementValidationError",
                    "diagnostic_code": "DORSAL_MARKERS_INSUFFICIENT",
                    "image_field": "dorsalFootImage",
                },
            ),
            (
                "transport-failure",
                HTTPException(
                    status_code=502,
                    detail={
                        "stage": "core-save-metrics",
                        "errorType": "ConnectError",
                    },
                ),
                {
                    "failure_reason": "NETWORK_ERROR",
                    "stage": "core-save-metrics",
                    "error_type": "ConnectError",
                },
            ),
            (
                "retryable-partial",
                JSONResponse(
                    status_code=207,
                    content={
                        "errors": [
                            {
                                "stage": "summary-save-HALLUX_VALGUS",
                                "errorType": "UpstreamHTTPError",
                                "retryable": True,
                            }
                        ]
                    },
                ),
                {
                    "failure_reason": "REPORT_SAVE_ERROR",
                    "stage": "summary-save-HALLUX_VALGUS",
                    "error_type": "UpstreamHTTPError",
                },
            ),
            (
                "nonretryable-partial",
                JSONResponse(
                    status_code=207,
                    content={
                        "errors": [
                            {
                                "stage": "core-save-metrics",
                                "errorType": "UpstreamHTTPError",
                                "retryable": False,
                            }
                        ]
                    },
                ),
                {
                    "failure_reason": "REPORT_SAVE_ERROR",
                    "stage": "core-save-metrics",
                    "error_type": "UpstreamHTTPError",
                },
            ),
            (
                "expired-token-partial",
                JSONResponse(
                    status_code=207,
                    content={
                        "errors": [
                            {
                                "stage": "core-save-hallux-valgus",
                                "errorType": "UpstreamHTTPError",
                                "retryable": False,
                                "statusCode": 422,
                            },
                            {
                                "stage": "summary-save-HALLUX_VALGUS",
                                "errorType": "UpstreamHTTPError",
                                "retryable": False,
                                "statusCode": 401,
                            },
                        ]
                    },
                ),
                {
                    "failure_reason": "TOKEN_EXPIRED",
                    "stage": "summary-save-HALLUX_VALGUS",
                    "error_type": "UpstreamHTTPError",
                },
            ),
            (
                "direct-expired-token",
                HTTPException(
                    status_code=401,
                    detail={
                        "stage": "core-save-metrics",
                        "errorType": "UpstreamHTTPError",
                    },
                ),
                {
                    "failure_reason": "TOKEN_EXPIRED",
                    "stage": "core-save-metrics",
                    "error_type": "UpstreamHTTPError",
                },
            ),
            (
                "nested-expired-token",
                HTTPException(
                    status_code=502,
                    detail={
                        "stage": "core-save-metrics",
                        "errorType": "ReportTransportError",
                        "reports": {
                            "metrics": {
                                "status": "http-error",
                                "statusCode": 401,
                            }
                        },
                    },
                ),
                {
                    "failure_reason": "TOKEN_EXPIRED",
                    "stage": "core-save-metrics",
                    "error_type": "ReportTransportError",
                },
            ),
            (
                "local-ai-failure",
                HTTPException(
                    status_code=500,
                    detail={
                        "stage": "gpt-report-text",
                        "errorType": "FullPhotoOutputValidationError",
                        "diagnosticCode": "GPT_REPORT_RESPONSE_INVALID",
                        "validationField": "metric_advice.HALLUX_VALGUS.observation",
                        "validationRule": "SENTENCE_COUNT_OUT_OF_RANGE",
                    },
                ),
                {
                    "failure_reason": "AI_SERVER_ERROR",
                    "stage": "gpt-report-text",
                    "error_type": "FullPhotoOutputValidationError",
                    "diagnostic_code": "GPT_REPORT_RESPONSE_INVALID",
                    "image_field": None,
                    "validation_field": "metric_advice.HALLUX_VALGUS.observation",
                    "validation_rule": "SENTENCE_COUNT_OUT_OF_RANGE",
                },
            ),
            (
                "unknown-http-status",
                HTTPException(
                    status_code=418,
                    detail={
                        "stage": "background-worker",
                        "errorType": "UnexpectedHTTPError",
                    },
                ),
                {
                    "failure_reason": "UNKNOWN",
                    "stage": "background-worker",
                    "error_type": "UnexpectedHTTPError",
                },
            ),
        )

        async def exercise(outcome, work_item):
            sessions = {work_item.measurement_session_id}
            executor = (
                AsyncMock(side_effect=outcome)
                if isinstance(outcome, BaseException)
                else AsyncMock(return_value=outcome)
            )
            notifier = AsyncMock()
            progress = Mock()
            with (
                patch.object(
                    route,
                    "_photo_analysis_worker_semaphore",
                    asyncio.Semaphore(1),
                ),
                patch.object(
                    route,
                    "_photo_analysis_reservation_lock",
                    threading.Lock(),
                ),
                patch.object(route, "_active_photo_analysis_sessions", sessions),
                patch.object(route, "_execute_photo_pipeline", new=executor),
                patch.object(route, "_notify_photo_job_failure", new=notifier),
                patch.object(route, "_log_photo_progress", new=progress),
            ):
                await route._run_photo_analysis_job(work_item)
                return sessions, notifier, progress

        for index, (label, outcome, expected_notification) in enumerate(cases):
            with self.subTest(label=label):
                work_item = _queued_work_item(
                    70 + index,
                    request_id=f"terminal-{index}",
                )
                sessions, notifier, progress = asyncio.run(
                    exercise(outcome, work_item)
                )
                self.assertEqual(sessions, set())
                progress_by_event = {
                    call.kwargs["event"]: call.kwargs["message"]
                    for call in progress.call_args_list
                }
                self.assertIn("시작합니다", progress_by_event["job_started"])
                if expected_notification is None:
                    notifier.assert_not_awaited()
                    self.assertIn(
                        "완료했습니다",
                        progress_by_event["job_succeeded"],
                    )
                else:
                    notifier.assert_awaited_once_with(
                        work_item,
                        **expected_notification,
                    )
                    self.assertIn("실패", progress_by_event["job_failed"])
                    if label == "capture-failure":
                        self.assertIn(
                            "ArUco 마커 ID 1 또는 ID 4를 찾지 못했습니다",
                            progress_by_event["job_failed"],
                        )


class PhotoAnalysisRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        capture_passed = patch.object(
            route, "notify_photo_capture_passed",
            new=AsyncMock(side_effect=_confirm_photo_capture_passed),
        )
        capture_passed.start()
        self.addCleanup(capture_passed.stop)
        inflammation_enabled = patch.object(route.settings, "tinea_inflammation_enabled", True)
        inflammation_enabled.start()
        self.addCleanup(inflammation_enabled.stop)
        self.client = TestClient(app)
        self.output_dir = Path("photo-analysis-test-output")
        self.integrated = _integrated()
        self.left_plantar = _plantar(self.output_dir, "left", 3, 0.42)
        self.right_plantar = _plantar(self.output_dir, "right", 2, 0.31)
        self.sole_left = _neutral_plantar_tinea("sole-left")
        self.sole_right = _neutral_plantar_tinea("sole-right")
        self.left_plantar.tinea_analysis_result = self.sole_left
        self.right_plantar.tinea_analysis_result = self.sole_right

    def tearDown(self) -> None:
        self.client.close()

    def _post(self):
        return execute_photo_pipeline_for_test(output_dir=self.output_dir)

    def _post_endpoint(
        self,
        *,
        measurement_session_id: int = 34,
        photo_capture_attempt: int = 0,
        dorsal_bytes: bytes = b"raw-dorsal",
        left_plantar_bytes: bytes = b"raw-left-plantar",
        right_plantar_bytes: bytes = b"raw-right-plantar",
    ) -> httpx.Response:
        return self.client.post(
            ENDPOINT,
            data={"measurementSessionId": str(measurement_session_id), "photoCaptureAttempt": str(photo_capture_attempt)},
            files={
                "dorsalFootImage": (
                    "dorsal.jpg",
                    dorsal_bytes,
                    "image/jpeg",
                ),
                "leftPlantarFootImage": (
                    "left.jpg",
                    left_plantar_bytes,
                    "image/jpeg",
                ),
                "rightPlantarFootImage": (
                    "right.jpg",
                    right_plantar_bytes,
                    "image/jpeg",
                ),
            },
            headers={"Authorization": "Bearer photo-token"},
        )

    def _patch_pipeline(self, backend: _Backend, *, integrated=None, generated_texts=None):
        integrated_result = integrated or self.integrated

        def read_png(path: Path, label: str, *, require_alpha: bool = False):
            if "footprint visualization" in label:
                return b"left-footprint" if "left" in label else b"right-footprint"
            return b"left-clean-sole" if "left" in label else b"right-clean-sole"

        def combine(left: bytes, right: bytes):
            pairs = {
                (b"dorsal-left-map", b"dorsal-right-map"): b"dorsal-map",
                (b"dorsal-left-overlay", b"dorsal-right-overlay"): b"dorsal-overlay",
                (b"dorsal-left-clean", b"dorsal-right-clean"): b"dorsal-clean",
                (b"sole-left-overlay", b"sole-right-overlay"): b"sole-overlay",
                (b"sole-left-clean", b"sole-right-clean"): b"sole-clean",
                (b"left-clean-sole", b"right-clean-sole"): b"sole-clean",
            }
            return pairs[(left, right)]

        async def generate(**kwargs):
            backend.analysis_done = True
            return generated_texts if generated_texts is not None else TEXTS

        return (
            patch.object(
                route,
                "analyze_integrated_foot_photo",
                return_value=integrated_result,
            ),
            patch.object(
                route,
                "new_request_dir",
                return_value=("photo-request", self.output_dir),
            ),
            patch.object(
                route,
                "analyze_plantar_image",
                side_effect=[self.left_plantar, self.right_plantar],
            ),
            patch.object(route, "_read_png", side_effect=read_png),
            patch.object(
                route,
                "_read_mask",
                side_effect=[
                    np.full((8, 8), 255, np.uint8),
                    np.full((8, 8), 255, np.uint8),
                ],
            ),
            patch.object(
                route,
                "build_plantar_skin_result",
                side_effect=[self.sole_left, self.sole_right],
            ),
            patch.object(
                route,
                "combine_png_images_side_by_side",
                side_effect=combine,
            ),
            patch.object(
                route,
                "generate_full_photo_analysis_texts",
                new=AsyncMock(side_effect=generate),
            ),
            patch.object(route.httpx, "AsyncClient", return_value=backend),
        )

    @staticmethod
    def _progress_payloads(captured_logs) -> list[dict[str, object]]:
        marker = "pipeline_progress "
        return [
            json.loads(message.split(marker, 1)[1])
            for record in captured_logs.records
            if marker in (message := record.getMessage())
        ]

    def test_openapi_registers_exact_single_request_fields(self) -> None:
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"][ENDPOINT]["post"]
        body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        schema_name = body["$ref"].split("/")[-1]
        properties = schema["components"]["schemas"][schema_name]["properties"]
        self.assertIn("photoCaptureAttempt", schema["components"]["schemas"][schema_name]["required"])
        self.assertEqual(properties["photoCaptureAttempt"]["minimum"], 0)
        self.assertEqual(properties["photoCaptureAttempt"]["maximum"], 3)
        self.assertNotIn("default", properties["photoCaptureAttempt"])
        self.assertEqual(
            set(properties),
            {
                "measurementSessionId",
                "photoCaptureAttempt",
                "dorsalFootImage",
                "leftPlantarFootImage",
                "rightPlantarFootImage",
            },
        )
        for status_code in ("400", "413", "422", "500"):
            documented = operation["responses"][status_code]
            self.assertEqual(
                set(documented["headers"]),
                {"X-Feetfit-Pipeline", "X-Feetfit-Request-Id"},
            )
            error_ref = documented["content"]["application/json"]["schema"][
                "$ref"
            ]
            self.assertTrue(error_ref.endswith("/PhotoAnalysisErrorResponse"))
        self.assertNotIn("200", operation["responses"])
        self.assertNotIn("207", operation["responses"])
        self.assertIn("502", operation["responses"])
        self.assertEqual(
            set(operation["responses"]["202"]["headers"]),
            {"X-Feetfit-Pipeline", "X-Feetfit-Request-Id"},
        )
        accepted_schema = operation["responses"]["202"]["content"][
            "application/json"
        ]["schema"]
        self.assertTrue(
            {
                "accepted",
                "status",
                "captureValidationStatus",
                "mayStartPressure",
                "measurementSessionId",
                "photoCaptureAttempt",
                "measurementStatus",
                "statusCallbackConfirmed",
                "pipeline",
                "requestId",
            }.issubset(accepted_schema["required"])
        )
        self.assertEqual(
            accepted_schema["properties"]["captureValidationStatus"]["enum"],
            ["PASSED"],
        )
        self.assertEqual(
            accepted_schema["properties"]["mayStartPressure"]["type"],
            "boolean",
        )
        self.assertEqual(
            accepted_schema["properties"]["mayStartPressure"]["enum"],
            [True],
        )
        for status_code in ("409", "503"):
            error_ref = operation["responses"][status_code]["content"][
                "application/json"
            ]["schema"]["$ref"]
            self.assertTrue(error_ref.endswith("/PhotoAnalysisErrorResponse"))
        self.assertEqual(
            set(operation["responses"]["409"]["headers"]),
            {"X-Feetfit-Pipeline", "X-Feetfit-Request-Id"},
        )
        self.assertEqual(
            set(operation["responses"]["503"]["headers"]),
            {
                "X-Feetfit-Pipeline",
                "X-Feetfit-Request-Id",
                "Retry-After",
            },
        )
        detail_schema = schema["components"]["schemas"][
            "PhotoAnalysisErrorDetail"
        ]
        self.assertTrue(
            {
                "pipeline",
                "stage",
                "completedStages",
                "requestId",
                "errorType",
                "retryable",
                "partialSave",
                "reports",
                "message",
                "diagnosticCode",
                "recaptureRequired",
                "mayStartPressure",
                "recommendedAction",
                "detectedFootCount",
                "requiredFootCount",
            }.issubset(detail_schema["properties"])
        )

    def test_preflight_prepared_inputs_reach_contact_and_wound_with_raw_bytes_preserved(self) -> None:
        left_prepared = SimpleNamespace(
            rotation_degrees_clockwise=90, brightness_gain=1.0
        )
        right_prepared = SimpleNamespace(
            rotation_degrees_clockwise=90, brightness_gain=1.0
        )
        preflight = preflight_service.PhotoCapturePreflightResult(
            dorsal_width=1280,
            dorsal_height=720,
            detected_marker_count=2,
            left_plantar_width=480,
            left_plantar_height=640,
            right_plantar_width=480,
            right_plantar_height=640,
            left_plantar_foot_confidence=0.91,
            right_plantar_foot_confidence=0.92,
            left_plantar_prepared_input=left_prepared,
            right_plantar_prepared_input=right_prepared,
        )
        backend = _Backend()
        patches = self._patch_pipeline(backend)
        with (
            patches[0] as original_dorsal_analyze,
            patches[1],
            patches[2] as plantar_analyze,
            patches[3],
            patches[4],
            patches[5] as sole_analyze,
            patches[6],
            patches[7] as generate,
            patches[8],
            patch.object(
                route, "validate_photo_capture_preflight", return_value=preflight
            ),
            patch.object(
                route,
                "validate_dorsal_measurement_preflight",
                return_value=self.integrated.geometry,
            ),
            patch.object(
                route,
                "analyze_prepared_foot_geometry",
                return_value=self.integrated,
            ),
            patch.object(route.BackgroundTasks, "add_task") as add_task,
        ):
            with self.assertLogs(route.progress_logger.name, level="INFO") as logs:
                accepted = self._post_endpoint()
            self.assertEqual(accepted.status_code, 202, accepted.text)
            add_task.assert_called_once()
            runner, work_item = add_task.call_args.args
            self.assertIs(runner, route._run_photo_analysis_job)
            self.assertIs(work_item.left_plantar_prepared_input, left_prepared)
            self.assertIs(work_item.right_plantar_prepared_input, right_prepared)
            self.assertEqual(work_item.left_plantar_bytes, b"raw-left-plantar")
            self.assertEqual(work_item.right_plantar_bytes, b"raw-right-plantar")
            try:
                response = asyncio.run(route._execute_photo_pipeline(work_item))
            finally:
                asyncio.run(route._release_photo_analysis_session(34))

        self.assertEqual(response.status_code, 200, response.body)
        preflight_progress = next(
            entry
            for entry in self._progress_payloads(logs)
            if entry["event"] == "capture_preflight_passed"
        )["details"]
        self.assertEqual(preflight_progress["leftPlantarRotationDegreesClockwise"], 90)
        self.assertEqual(preflight_progress["leftPlantarBrightnessGain"], 1.0)
        self.assertEqual(preflight_progress["rightPlantarRotationDegreesClockwise"], 90)
        self.assertEqual(preflight_progress["rightPlantarBrightnessGain"], 1.0)
        original_dorsal_analyze.assert_not_called()
        self.assertEqual(
            plantar_analyze.call_args_list,
            [
                call(
                    b"raw-left-plantar",
                    "left",
                    self.output_dir,
                    prepared_input=left_prepared,
                ),
                call(
                    b"raw-right-plantar",
                    "right",
                    self.output_dir,
                    prepared_input=right_prepared,
                ),
            ],
        )
        sole_analyze.assert_not_called()
        generate.assert_awaited_once()
        self.assertEqual(
            generate.await_args.kwargs["left_plantar_raw_image_bytes"],
            b"raw-left-plantar",
        )
        self.assertEqual(
            generate.await_args.kwargs["right_plantar_raw_image_bytes"],
            b"raw-right-plantar",
        )

    def test_fallback_saves_each_report_once_and_identifies_its_source(self) -> None:
        backend = _Backend()
        texts = replace(
            TEXTS, generation_source="DETERMINISTIC_FALLBACK",
            generation_failure_code="OUTPUT_VALIDATION_FAILED",
        )
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in self._patch_pipeline(backend, generated_texts=texts)]
            stack.enter_context(patch.object(route.settings, "openai_report_fallback_enabled", True))
            logs = stack.enter_context(self.assertLogs(route.progress_logger, level="INFO"))
            response = self._post()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["analysis"]["reportTextGeneration"], {
            "source": "DETERMINISTIC_FALLBACK", "model": None,
            "failureCode": "OUTPUT_VALIDATION_FAILED",
        })
        self.assertFalse(response.json()["partialSave"])
        self.assertEqual(Counter(url for url, _ in backend.calls), Counter({
            route.settings.hallux_valgus_report_endpoint: 1,
            route.settings.daily_metrics_report_endpoint: 1,
            route.settings.tinea_report_endpoint: 1,
            route.settings.plantar_footprint_report_endpoint: 1,
            route.settings.summary_report_endpoint: 3,
        }))
        mocks[7].assert_awaited_once()
        completed = next(row for row in self._progress_payloads(logs)
                         if row["event"] == "gpt_generation_completed")
        self.assertEqual(completed["details"]["generationSource"], "DETERMINISTIC_FALLBACK")
        self.assertEqual(completed["details"]["errorCode"], "OUTPUT_VALIDATION_FAILED")
        self.assertIn("서버 분석 사실", completed["message"])
        self.assertNotIn("GPT", completed["message"])

    def test_disabled_inflammation_uses_actual_clean_images_and_truthful_metadata(self) -> None:
        for global_enabled, persisted_mode in ((False, None), (True, "disabled_for_testing")):
            with self.subTest(global_enabled=global_enabled, persisted_mode=persisted_mode):
                integrated = self.integrated
                if persisted_mode is not None:
                    left = integrated.tinea["left"]
                    integrated = replace(integrated, tinea={
                        **integrated.tinea,
                        "left": replace(left, metrics={
                            **left.metrics,
                            "inflammation_analysis_enabled": False,
                            "inflammation_analysis_mode": persisted_mode,
                        }),
                    })
                texts = replace(
                    TEXTS,
                    skin_reaction_safety_description="사진에서 보이는 피부 표면의 상태를 살펴보았습니다.",
                    inflammation_analysis_enabled=False,
                    local_policy_fields=(),
                )
                backend = _Backend()
                with ExitStack() as stack:
                    mocks = [stack.enter_context(p) for p in self._patch_pipeline(
                        backend, integrated=integrated, generated_texts=texts,
                    )]
                    stack.enter_context(patch.object(
                        route.settings, "tinea_inflammation_enabled", global_enabled,
                    ))
                    response = self._post()

                self.assertEqual(response.status_code, 200, response.text)
                generator_kwargs = mocks[7].await_args.kwargs
                self.assertEqual(generator_kwargs["related_analysis_images"], {
                    "HALLUX_LEFT_RESULT": b"hallux-left-png",
                    "HALLUX_RIGHT_RESULT": b"hallux-right-png",
                    "TINEA_DORSAL_CLEAN_UNANNOTATED_PHOTO": b"dorsal-clean",
                    "LEFT_PLANTAR_FOOTPRINT": b"left-footprint",
                    "RIGHT_PLANTAR_FOOTPRINT": b"right-footprint",
                })
                self.assertEqual(generator_kwargs["dorsal_raw_image_bytes"], b"raw-dorsal")
                self.assertEqual(generator_kwargs["left_plantar_raw_image_bytes"], b"raw-left-plantar")
                self.assertEqual(generator_kwargs["right_plantar_raw_image_bytes"], b"raw-right-plantar")
                tinea_facts = generator_kwargs["facts"]["tinea_pedis"]
                self.assertEqual(tinea_facts["final_scores"], {
                    "fungal_suspicion_safety_score": 89,
                    "skin_reaction_safety_score": 91,
                })
                if persisted_mode is not None:
                    self.assertIs(tinea_facts["dorsal"]["left"]["inflammation_analysis_enabled"], False)
                    self.assertEqual(
                        tinea_facts["dorsal"]["left"]["inflammation_analysis_mode"],
                        "disabled_for_testing",
                    )
                self.assertEqual(response.json()["analysis"]["reportTextGeneration"], {
                    "source": "OPENAI",
                    "model": route.settings.openai_report_model,
                    "inflammationAnalysisEnabled": False,
                    "inflammationAnalysisMode": "image_based_with_detector_disabled",
                    "localPolicyFields": [],
                })
                tinea_upload = next(kwargs["files"] for url, kwargs in backend.calls
                                    if url == route.settings.tinea_report_endpoint)
                self.assertEqual(tinea_upload["suspiciousAreaMapImage"][1], b"dorsal-map")
                self.assertEqual(tinea_upload["originalFootImage"][1], b"dorsal-overlay")
                self.assertEqual(
                    json.loads(tinea_upload["request"][1])["skinReactionSafetyDescription"],
                    texts.skin_reaction_safety_description,
                )

    def test_success_saves_exact_reports_and_three_photo_summaries(self) -> None:
        # Distinct bytes catch accidentally uploading the clean plantar photo.
        self.sole_left = replace(self.sole_left, photo_overlay_png=b"sole-left-overlay")
        self.sole_right = replace(self.sole_right, photo_overlay_png=b"sole-right-overlay")
        backend = _Backend()
        patches = self._patch_pipeline(backend)
        with patches[0] as integrated_analyze, patches[1], patches[2] as plantar_analyze, patches[3], patches[4], patches[5] as sole_analyze, patches[6], patches[7] as generate, patches[8]:
            response = self._post()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(response.headers["x-feetfit-pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(response.headers["x-feetfit-request-id"], "photo-request")
        self.assertEqual(body["requestId"], "photo-request")
        self.assertFalse(body["partialSave"])
        self.assertEqual(body["failedStages"], [])
        self.assertEqual(body["errors"], [])
        self.assertEqual(
            body["analysis"]["reportTextGeneration"],
            {
                "source": "OPENAI",
                "model": route.settings.openai_report_model,
            },
        )
        self.assertEqual(
            body["analysis"]["tineaPedis"],
            {
                "fungalSuspicionSafetyScore": 89,
                "skinReactionSafetyScore": 91,
            },
        )
        expected_completed = (
            "input-read-dorsal",
            "input-read-left-plantar",
            "input-read-right-plantar",
            "capture-preflight",
            "dorsal-measurement-preflight",
            "capture-status-transition",
            "dorsal-integrated-analysis",
            "measurement-validation",
            "plantar-left-analysis",
            "plantar-right-analysis",
            "plantar-artifact-read",
            "analysis-image-composition",
            "analysis-fact-build",
            "gpt-report-text",
            "backend-payload-build",
            "core-save-hallux-valgus",
            "core-save-metrics",
            "core-save-tinea-pedis",
            "core-save-plantar-footprint",
            "summary-save-ATHLETES_FOOT",
            "summary-save-SKIN_IRRITATION",
            "summary-save-HALLUX_VALGUS",
        )
        self.assertEqual(tuple(body["completedStages"]), expected_completed)
        self.assertEqual(response.json()["skippedMetricTypes"], [
            "PRESSURE_BALANCE",
            "FOOT_ENVIRONMENT",
        ])
        integrated_analyze.assert_called_once_with(b"raw-dorsal", "dorsal.jpg")
        self.assertEqual(
            plantar_analyze.call_args_list,
            [
                call(b"raw-left-plantar", "left", self.output_dir),
                call(b"raw-right-plantar", "right", self.output_dir),
            ],
        )
        sole_analyze.assert_not_called()

        generate.assert_awaited_once()
        generator_kwargs = generate.await_args.kwargs
        self.assertEqual(generator_kwargs["dorsal_raw_image_bytes"], b"raw-dorsal")
        self.assertEqual(
            generator_kwargs["left_plantar_raw_image_bytes"],
            b"raw-left-plantar",
        )
        self.assertEqual(
            generator_kwargs["right_plantar_raw_image_bytes"],
            b"raw-right-plantar",
        )
        self.assertEqual(
            generator_kwargs["facts"]["plantar_footprint"]["left"],
            {"region_count": 3, "contact_area_ratio": 0.42, "status": "detected"},
        )
        self.assertEqual(
            generator_kwargs["facts"]["tinea_pedis"]["final_scores"],
            {
                "fungal_suspicion_safety_score": 89,
                "skin_reaction_safety_score": 91,
            },
        )
        self.assertNotIn("metrics", generator_kwargs["facts"])
        self.assertEqual(
            set(generator_kwargs["related_analysis_images"]),
            {
                "HALLUX_LEFT_RESULT",
                "HALLUX_RIGHT_RESULT",
                "DORSAL_TINEA_SUSPICION_MAP",
                "DORSAL_TINEA_PHOTO_OVERLAY",
                "LEFT_PLANTAR_FOOTPRINT",
                "RIGHT_PLANTAR_FOOTPRINT",
            },
        )

        self.assertEqual(len(backend.calls), 7)
        urls = [url for url, _ in backend.calls]
        self.assertEqual(urls.count(route.settings.summary_report_endpoint), 3)
        self.assertNotIn("PRESSURE_BALANCE", str(backend.calls))
        self.assertNotIn("FOOT_ENVIRONMENT", str(backend.calls))
        for _, kwargs in backend.calls:
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer photo-token")

        by_url = {
            url: kwargs
            for url, kwargs in backend.calls
            if url != route.settings.summary_report_endpoint
        }
        metrics_json = by_url[route.settings.daily_metrics_report_endpoint]["json"]
        self.assertEqual(
            metrics_json,
            {
                "measurementSessionId": 34,
                "measuredLeftFootSizeMm": 253.0,
                "measuredRightFootSizeMm": 248.0,
                "leftFootWidthMm": 85.0,
                "rightFootWidthMm": 70.0,
            },
        )
        hallux_kwargs = by_url[route.settings.hallux_valgus_report_endpoint]
        self.assertEqual(
            json.loads(hallux_kwargs["files"]["request"][1])["scoreAnalysisText"],
            TEXTS.hallux_score_analysis_text,
        )
        self.assertEqual(
            set(hallux_kwargs["files"]),
            {"request", "leftFootImage", "rightFootImage"},
        )
        self.assertEqual(hallux_kwargs["files"]["request"][2], "application/json")

        tinea_files = by_url[route.settings.tinea_report_endpoint]["files"]
        self.assertEqual(
            set(tinea_files),
            {
                "request",
                "suspiciousAreaMapImage",
                "originalFootImage",
                "soleSuspiciousAreaMapImage",
                "soleOriginalFootImage",
            },
        )
        tinea_json = json.loads(tinea_files["request"][1])
        self.assertEqual(tinea_json["fungalSuspicionSafetyScore"], 89)
        self.assertEqual(tinea_json["skinReactionSafetyScore"], 91)
        self.assertEqual(tinea_files["suspiciousAreaMapImage"][1], b"dorsal-map")
        self.assertEqual(tinea_files["originalFootImage"][1], b"dorsal-overlay")
        self.assertEqual(tinea_files["soleSuspiciousAreaMapImage"][1], b"sole-clean")
        self.assertEqual(tinea_files["soleOriginalFootImage"], (
            "sole_not_assessed_photo.png", b"sole-clean", "image/png",
        ))

        footprint_files = by_url[route.settings.plantar_footprint_report_endpoint]["files"]
        self.assertEqual(
            set(footprint_files),
            {"request", "leftPlantarFootprintImage", "rightPlantarFootprintImage"},
        )
        footprint_json = json.loads(footprint_files["request"][1])
        self.assertEqual(
            footprint_json["plantarFootprintAnalysisText"],
            TEXTS.plantar_footprint_analysis_text,
        )
        self.assertEqual(footprint_files["leftPlantarFootprintImage"][1], b"left-footprint")
        self.assertEqual(footprint_files["rightPlantarFootprintImage"][1], b"right-footprint")

        summary_json = {
            kwargs["json"]["metricType"]: kwargs["json"]
            for url, kwargs in backend.calls
            if url == route.settings.summary_report_endpoint
        }
        self.assertEqual(
            set(summary_json),
            {"HALLUX_VALGUS", "ATHLETES_FOOT", "SKIN_IRRITATION"},
        )
        self.assertEqual(summary_json["HALLUX_VALGUS"]["score"], 76.5)
        self.assertEqual(summary_json["ATHLETES_FOOT"]["score"], 89.0)
        self.assertEqual(summary_json["SKIN_IRRITATION"]["score"], 91.0)
        for metric_type, payload in summary_json.items():
            self.assertEqual(payload["measurementSessionId"], 34)
            self.assertEqual(payload["advice"], list(TEXTS.metric_advice[metric_type]))
            self.assertEqual(len(payload["advice"]), 2)

    def test_photo_summaries_are_posted_strictly_sequentially_in_fixed_order(self) -> None:
        class _SummarySequenceBackend(_Backend):
            def __init__(self) -> None:
                super().__init__()
                self.active_summary_posts = 0
                self.max_active_summary_posts = 0
                self.summary_events: list[tuple[str, str]] = []

            async def post(self, url: str, **kwargs):
                if url != route.settings.summary_report_endpoint:
                    return await super().post(url, **kwargs)

                metric_type = kwargs["json"]["metricType"]
                self.active_summary_posts += 1
                self.max_active_summary_posts = max(
                    self.max_active_summary_posts,
                    self.active_summary_posts,
                )
                self.summary_events.append(("start", metric_type))
                try:
                    # Yield so a concurrent implementation would overlap all
                    # summary requests and make max_active_summary_posts > 1.
                    await asyncio.sleep(0)
                    return await super().post(url, **kwargs)
                finally:
                    self.summary_events.append(("end", metric_type))
                    self.active_summary_posts -= 1

        backend = _SummarySequenceBackend()
        patches = self._patch_pipeline(backend)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            response = self._post()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["partialSave"])
        self.assertEqual(backend.max_active_summary_posts, 1)
        self.assertEqual(
            backend.summary_events,
            [
                ("start", "ATHLETES_FOOT"),
                ("end", "ATHLETES_FOOT"),
                ("start", "SKIN_IRRITATION"),
                ("end", "SKIN_IRRITATION"),
                ("start", "HALLUX_VALGUS"),
                ("end", "HALLUX_VALGUS"),
            ],
        )

    def test_unilateral_lesions_keep_the_lower_score_in_facts_core_report_and_summary(self) -> None:
        integrated = replace(self.integrated, tinea={
            "left": replace(self.integrated.tinea["left"], fungal_safety_score=65,
                            skin_reaction_safety_score=61),
            "right": replace(self.integrated.tinea["right"], fungal_safety_score=100,
                             skin_reaction_safety_score=100),
        })
        backend = _Backend()
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in self._patch_pipeline(backend, integrated=integrated)]
            response = self._post()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["analysis"]["tineaPedis"], {
            "fungalSuspicionSafetyScore": 65, "skinReactionSafetyScore": 61,
        })
        facts = mocks[7].await_args.kwargs["facts"]["tinea_pedis"]["final_scores"]
        self.assertEqual(facts, {"fungal_suspicion_safety_score": 65,
                                "skin_reaction_safety_score": 61})
        upload = next(kwargs["files"] for url, kwargs in backend.calls
                      if url == route.settings.tinea_report_endpoint)
        saved = json.loads(upload["request"][1])
        self.assertEqual(saved["fungalSuspicionSafetyScore"], 65)
        self.assertEqual(saved["skinReactionSafetyScore"], 61)
        summaries = {kwargs["json"]["metricType"]: kwargs["json"]["score"]
                     for url, kwargs in backend.calls
                     if url == route.settings.summary_report_endpoint}
        self.assertEqual(summaries["ATHLETES_FOOT"], 65)
        self.assertEqual(summaries["SKIN_IRRITATION"], 61)

    def test_success_emits_safe_progress_for_every_pipeline_area(self) -> None:
        upstream_sentinel = "UPSTREAM_BODY_MUST_NOT_BE_LOGGED"

        class _SensitiveBodyBackend(_Backend):
            async def post(self, url: str, **kwargs):
                response = await super().post(url, **kwargs)
                if url == route.settings.daily_metrics_report_endpoint:
                    return httpx.Response(
                        response.status_code,
                        json={
                            "privateUpstreamBody": upstream_sentinel,
                            "authorization": "Bearer upstream-secret",
                        },
                        headers={"content-type": "application/json"},
                    )
                return response

        backend = _SensitiveBodyBackend()
        patches = self._patch_pipeline(backend)
        with self.assertLogs(route.progress_logger.name, level="INFO") as captured:
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
                patches[8],
            ):
                response = self._post()

        self.assertEqual(response.status_code, 200, response.text)
        payloads = self._progress_payloads(captured)
        self.assertGreaterEqual(len(payloads), 26)
        required_keys = {
            "pipeline",
            "requestId",
            "measurementSessionId",
            "event",
            "stage",
            "message",
        }
        for payload in payloads:
            self.assertTrue(required_keys.issubset(payload))
            self.assertEqual(payload["pipeline"], route.PIPELINE_VERSION)
            self.assertEqual(payload["requestId"], "photo-request")
            self.assertEqual(payload["measurementSessionId"], 34)
            for key, value in payload.get("details", {}).items():
                self.assertIn(key, route._PROGRESS_DETAIL_KEYS)
                self.assertIsInstance(value, (bool, int, float, str))
                if isinstance(value, str):
                    self.assertIn(value, route._PROGRESS_DETAIL_ENUMS)

        def select(event: str, stage: str) -> dict[str, object]:
            matches = [
                item
                for item in payloads
                if item["event"] == event and item["stage"] == stage
            ]
            self.assertEqual(len(matches), 1, (event, stage, matches))
            return matches[0]

        # Upload intake now runs before the background worker is registered.
        # This direct executor test starts after those three stages.
        self.assertFalse(
            any(item["event"] == "input_read_completed" for item in payloads)
        )
        dorsal_metrics = select(
            "foot_metrics_analyzed", "dorsal-integrated-analysis"
        )["details"]
        self.assertEqual(dorsal_metrics["leftFootLengthMm"], 253.0)
        self.assertEqual(dorsal_metrics["rightFootWidthMm"], 70.0)
        aruco = select("aruco_analyzed", "dorsal-integrated-analysis")["details"]
        self.assertTrue(aruco["measurementValid"])
        self.assertEqual(aruco["detectedMarkerCount"], 2)
        hallux = select(
            "hallux_valgus_analyzed", "dorsal-integrated-analysis"
        )["details"]
        self.assertEqual(hallux["leftToeAngleDegree"], 12.0)
        self.assertEqual(hallux["rightToeAngleDegree"], 18.0)
        self.assertEqual(
            {
                item["stage"]
                for item in payloads
                if item["event"] == "dorsal_tinea_analyzed"
            },
            {"dorsal-tinea-left-analysis", "dorsal-tinea-right-analysis"},
        )
        self.assertEqual(
            {
                item["stage"]
                for item in payloads
                if item["event"] == "plantar_segmentation_completed"
            },
            {"plantar-left-analysis", "plantar-right-analysis"},
        )
        self.assertEqual(
            {
                item["stage"]
                for item in payloads
                if item["event"] == "sole_skin_analysis_completed"
            },
            set(),
        )
        gpt_started = select("gpt_generation_started", "gpt-report-text")
        self.assertIn("리포트 문구", gpt_started["message"])
        self.assertTrue(str(gpt_started["message"]).endswith("시작합니다."))
        self.assertIs(gpt_started["details"]["responseTimeoutDisabled"], False)
        self.assertEqual(
            gpt_started["details"]["generationTimeoutSeconds"],
            route.settings.openai_report_full_photo_timeout_seconds,
        )
        self.assertEqual(
            gpt_started["details"]["connectTimeoutSeconds"],
            route.settings.openai_report_connect_timeout_seconds,
        )
        self.assertEqual(
            gpt_started["details"]["poolTimeoutSeconds"],
            route.settings.openai_report_pool_timeout_seconds,
        )
        gpt_completed = select("gpt_generation_completed", "gpt-report-text")
        self.assertEqual(
            gpt_completed["message"],
            "GPT 리포트 문구 생성을 완료했습니다.",
        )
        self.assertEqual(gpt_completed["details"]["generationSource"], "OPENAI")

        core_results = {
            item["stage"]: item["details"]
            for item in payloads
            if item["event"] == "core_save_result"
        }
        self.assertEqual(set(core_results), set(route._CORE_SAVE_STAGES.values()))
        self.assertTrue(
            all(details["outcome"] == "saved" for details in core_results.values())
        )
        self.assertTrue(
            all(
                "저장을 완료했습니다" in str(item["message"])
                for item in payloads
                if item["event"] == "core_save_result"
            )
        )
        summary_results = {
            item["stage"]: item["details"]
            for item in payloads
            if item["event"] == "summary_save_result"
        }
        self.assertEqual(
            set(summary_results),
            {
                "summary-save-ATHLETES_FOOT",
                "summary-save-SKIN_IRRITATION",
                "summary-save-HALLUX_VALGUS",
            },
        )
        self.assertTrue(
            all(
                details["outcome"] == "saved"
                for details in summary_results.values()
            )
        )
        final_details = select("response_completed", "response")["details"]
        self.assertEqual(final_details["statusCode"], 200)
        self.assertTrue(final_details["allReportsOk"])
        self.assertFalse(final_details["partialSave"])

        for record in captured.records:
            operator_line = record.getMessage()
            self.assertTrue(
                operator_line.startswith(
                    "[사진 분석][세션 34][요청 photo-re] "
                )
            )
            self.assertIn(" | pipeline_progress ", operator_line)

        log_text = "\n".join(record.getMessage() for record in captured.records)
        for forbidden in (
            "photo-token",
            "upstream-secret",
            upstream_sentinel,
            "raw-dorsal",
            "raw-left-plantar",
            "raw-right-plantar",
            TEXTS.hallux_score_analysis_text,
            TEXTS.total_score_description,
        ):
            self.assertNotIn(forbidden, log_text)

    def test_207_logs_core_failure_summary_skips_and_final_status(self) -> None:
        backend = _Backend(
            status_by_url={route.settings.tinea_report_endpoint: 503}
        )
        patches = self._patch_pipeline(backend)
        with self.assertLogs(route.progress_logger.name, level="INFO") as captured:
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
                patches[8],
            ):
                response = self._post()

        self.assertEqual(response.status_code, 207, response.text)
        payloads = self._progress_payloads(captured)
        results = {
            item["stage"]: item["details"]
            for item in payloads
            if item["event"] in {"core_save_result", "summary_save_result"}
        }
        self.assertEqual(
            results["core-save-tinea-pedis"],
            {"outcome": "http-error", "retryable": True, "statusCode": 503},
        )
        failed_tinea_save = next(
            item
            for item in payloads
            if item["stage"] == "core-save-tinea-pedis"
        )
        self.assertEqual(
            failed_tinea_save["message"],
            "무좀·피부 분석 결과 저장에 실패했습니다. (HTTP 503)",
        )
        self.assertEqual(
            results["summary-save-ATHLETES_FOOT"], {"outcome": "skipped"}
        )
        self.assertEqual(
            results["summary-save-SKIN_IRRITATION"], {"outcome": "skipped"}
        )
        self.assertEqual(
            results["summary-save-HALLUX_VALGUS"]["outcome"], "saved"
        )
        final = next(
            item for item in payloads if item["event"] == "response_completed"
        )
        self.assertEqual(final["details"]["statusCode"], 207)
        self.assertFalse(final["details"]["allReportsOk"])
        self.assertTrue(final["details"]["partialSave"])
        self.assertEqual(final["details"]["attemptedSaveCount"], 5)
        self.assertEqual(final["details"]["successfulSaveCount"], 4)
        self.assertNotIn(
            "photo-token",
            "\n".join(record.getMessage() for record in captured.records),
        )

    def test_progress_allowlist_drops_sensitive_non_scalar_values(self) -> None:
        with patch.object(route.progress_logger, "info") as info:
            route._log_photo_progress(
                request_id="safe-request",
                measurement_session_id=34,
                stage="response",
                event="response_completed",
                message="사진 분석과 저장을 완료했습니다.",
                details={
                    "statusCode": 200,
                    "measurementStatus": "valid",
                    "authorization": "Bearer helper-secret",
                    "outcome": "Bearer helper-secret",
                    "rawImageBytes": b"raw-image-secret",
                    "statusBody": {"token": "nested-secret"},
                },
            )

        info.assert_called_once()
        self.assertEqual(
            info.call_args.args[0],
            "[사진 분석][세션 %s][요청 %s] %s | pipeline_progress %s",
        )
        self.assertEqual(
            info.call_args.args[1:4],
            (34, "safe-req", "사진 분석과 저장을 완료했습니다."),
        )
        payload = json.loads(info.call_args.args[4])
        self.assertEqual(
            payload["details"],
            {"measurementStatus": "valid", "statusCode": 200},
        )
        serialized = info.call_args.args[4]
        for forbidden in (
            "helper-secret",
            "raw-image-secret",
            "nested-secret",
            "authorization",
            "rawImageBytes",
            "statusBody",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_invalid_aruco_measurement_is_422_before_plantar_gpt_or_writes(self) -> None:
        invalid = _integrated(measurement_valid=False)
        with (
            patch.object(route, "analyze_integrated_foot_photo", return_value=invalid),
            patch.object(route, "analyze_plantar_image") as plantar,
            patch.object(
                route,
                "generate_full_photo_analysis_texts",
                new=AsyncMock(),
            ) as generate,
            patch.object(route.httpx, "AsyncClient") as backend,
        ):
            response = self._post()

        self.assertEqual(response.status_code, 422)
        plantar.assert_not_called()
        generate.assert_not_awaited()
        backend.assert_not_called()

    def test_integrated_analysis_error_classifies_input_and_internal_failures(self) -> None:
        cases = (
            ("required ArUco marker was not detected", 422),
            ("CUDA inference worker failed", 500),
        )
        for message, expected_status in cases:
            with self.subTest(message=message):
                with (
                    patch.object(
                        route,
                        "analyze_integrated_foot_photo",
                        side_effect=AnalysisError(message),
                    ),
                    patch.object(route, "analyze_plantar_image") as plantar,
                    patch.object(route.httpx, "AsyncClient") as backend,
                ):
                    response = self._post()

                self.assertEqual(response.status_code, expected_status)
                plantar.assert_not_called()
                backend.assert_not_called()

    def test_core_http_failure_returns_207_and_summarizes_every_attempt(self) -> None:
        backend = _Backend(
            status_by_url={route.settings.hallux_valgus_report_endpoint: 500}
        )
        patches = self._patch_pipeline(backend)
        with self.assertLogs(route.logger.name, level="WARNING") as captured_logs:
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
                response = self._post()

        self.assertEqual(response.status_code, 207, response.text)
        log_text = "\n".join(captured_logs.output)
        for expected_log_field in (
            "pipeline=full-photo-analysis-v1",
            "request_id=photo-request",
            "stage=core-save-hallux-valgus",
            "error_type=UpstreamHTTPError",
            "retryable=True",
            "partial_save=True",
            "status_code=500",
        ):
            self.assertIn(expected_log_field, log_text)
        self.assertNotIn("photo-token", log_text)
        body = response.json()
        self.assertTrue(body["partialSave"])
        self.assertEqual(body["failedStages"], ["core-save-hallux-valgus"])
        self.assertEqual(
            body["errors"],
            [
                {
                    "stage": "core-save-hallux-valgus",
                    "errorType": "UpstreamHTTPError",
                    "retryable": True,
                    "statusCode": 500,
                }
            ],
        )
        self.assertNotIn("core-save-hallux-valgus", body["completedStages"])
        self.assertIn("core-save-metrics", body["completedStages"])
        self.assertEqual(body["reports"]["halluxValgus"]["statusCode"], 500)
        self.assertEqual(
            body["reports"]["summaries"]["HALLUX_VALGUS"]["status"],
            "skipped",
        )
        summary_calls = [
            kwargs["json"]["metricType"]
            for url, kwargs in backend.calls
            if url == route.settings.summary_report_endpoint
        ]
        self.assertEqual(set(summary_calls), {"ATHLETES_FOOT", "SKIN_IRRITATION"})
        self.assertEqual(body["reports"]["summaries"]["ATHLETES_FOOT"]["statusCode"], 201)
        self.assertEqual(body["reports"]["summaries"]["SKIN_IRRITATION"]["statusCode"], 201)

    def test_transport_failure_is_502_and_does_not_leak_token(self) -> None:
        backend = _Backend()

        async def fail_post(url: str, **kwargs):
            raise httpx.ConnectError("Bearer photo-token must never leak")

        backend.post = fail_post
        patches = self._patch_pipeline(backend)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            response = self._post()

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("photo-token", response.text)
        self.assertEqual(response.headers["x-feetfit-pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(response.headers["x-feetfit-request-id"], "photo-request")
        detail = response.json()["detail"]
        self.assertEqual(detail["pipeline"], route.PIPELINE_VERSION)
        self.assertEqual(detail["requestId"], "photo-request")
        self.assertEqual(detail["stage"], "core-save-hallux-valgus")
        self.assertEqual(detail["errorType"], "ConnectError")
        self.assertTrue(detail["retryable"])
        self.assertFalse(detail["partialSave"])
        self.assertEqual(
            {
                name: report["status"]
                for name, report in detail["reports"].items()
                if name != "summaries"
            },
            {
                "halluxValgus": "transport-error",
                "metrics": "transport-error",
                "tineaPedis": "transport-error",
                "plantarFootprint": "transport-error",
            },
        )

    def test_tinea_core_failure_skips_both_dependent_health_summaries(self) -> None:
        backend = _Backend(
            status_by_url={route.settings.tinea_report_endpoint: 503}
        )
        patches = self._patch_pipeline(backend)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            response = self._post()

        self.assertEqual(response.status_code, 207, response.text)
        body = response.json()
        self.assertEqual(body["reports"]["tineaPedis"]["statusCode"], 503)
        summaries = body["reports"]["summaries"]
        self.assertEqual(summaries["ATHLETES_FOOT"]["status"], "skipped")
        self.assertEqual(summaries["SKIN_IRRITATION"]["status"], "skipped")
        summary_calls = [
            kwargs["json"]["metricType"]
            for url, kwargs in backend.calls
            if url == route.settings.summary_report_endpoint
        ]
        self.assertEqual(summary_calls, ["HALLUX_VALGUS"])

    def test_all_core_http_failures_return_complete_207_error_matrix(self) -> None:
        backend = _Backend(
            status_by_url={
                route.settings.hallux_valgus_report_endpoint: 503,
                route.settings.daily_metrics_report_endpoint: 422,
                route.settings.tinea_report_endpoint: 429,
                route.settings.plantar_footprint_report_endpoint: 500,
            }
        )
        patches = self._patch_pipeline(backend)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            response = self._post()

        self.assertEqual(response.status_code, 207, response.text)
        body = response.json()
        self.assertFalse(body["partialSave"])
        self.assertEqual(
            body["failedStages"],
            [
                "core-save-hallux-valgus",
                "core-save-metrics",
                "core-save-tinea-pedis",
                "core-save-plantar-footprint",
            ],
        )
        self.assertEqual(len(body["errors"]), 4)
        self.assertFalse(
            any(stage.startswith("core-save-") for stage in body["completedStages"])
        )
        self.assertEqual(
            [
                url
                for url, _ in backend.calls
                if url == route.settings.summary_report_endpoint
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
