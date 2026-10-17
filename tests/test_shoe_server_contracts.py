from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.api.routes import shoes as shoe_routes
from app.api.routes.shoes import _generate_and_save_summary
from app.main import app
from app.schemas.shoe_fit_comment import ShoeFitSummary
from app.schemas.shoe_server import ServerSavedRecommendation, ServerShoeCharacteristics
from app.schemas.shoes import (
    ShoeRecommendationForwardRequest,
    ShoeRecommendationItemPayload,
    ShoeRecommendationReasonPayload,
    ShoeSummaryForwardRequest,
    ShoeSummaryGenerateRequest,
    ShoeSummaryReasonPayload,
    ShoeSummaryTriggerRequest,
)
from app.services.shoe.shoe_recommendation import (
    ReasonFacts,
    ShoeFacts,
    ShoeRecommendationBatch,
    ShoeRecommendationBusyError,
)
from app.services.shoe.shoe_fit_comment_service import ShoeFitCommentError
from app.services.shoe.shoe_server_client import (
    ShoeServerClientError,
    ShoeServerConfigurationError,
)


def _recommendation_reasons() -> list[ShoeRecommendationReasonPayload]:
    return [
        ShoeRecommendationReasonPayload(
            reason_type=reason_type,
            title=f"{reason_type} title",
            risk_level="LOW",
            review_ids=[index],
        )
        for index, reason_type in enumerate(("FOREFOOT", "HEEL", "INSOLE"), start=1)
    ]


def _saved_recommendation() -> ServerSavedRecommendation:
    return ServerSavedRecommendation.model_validate(
        {
            "measurementSessionId": 30,
            "userId": 7,
            "shoeId": 101,
            "brandName": "Example Brand",
            "shoeName": "Example Shoe",
            "fitScore": 80.0,
            "pointSummary": None,
            "analyzedAt": "2026-08-23T12:30:00",
            "reasons": [
                {
                    "reasonType": reason_type,
                    "title": f"{reason_type} title",
                    "riskLevel": "LOW",
                    "reviewSummary": None,
                    "reviews": [
                        {
                            "reviewId": index,
                            "reviewText": f"서버에 저장된 실제 리뷰 {index}",
                            "source": "MUSINSA",
                        }
                    ],
                }
                for index, reason_type in enumerate(("FOREFOOT", "HEEL", "INSOLE"), start=1)
            ],
        }
    )


def _shoe_characteristics() -> ServerShoeCharacteristics:
    return ServerShoeCharacteristics.model_validate(
        {
            "shoeId": 101,
            "summary": "쿠션감은 낮은 편입니다.",
            "characteristics": [
                {
                    "type": "CUSHION",
                    "level": "LOW",
                    "value": "40.0",
                    "averageValue": "30.0",
                    "minValue": "10.0",
                    "maxValue": "50.0",
                    "unit": "HA",
                    "testedSize": None,
                }
            ],
        }
    )


def _shoe_55_saved_recommendation(
    *,
    missing_reason_types: set[str] | None = None,
    include_point_summary: bool = True,
) -> ServerSavedRecommendation:
    missing_reason_types = missing_reason_types or set()
    reason_data = {
        "FOREFOOT": {
            "title": "발볼 적합도 주의",
            "riskLevel": "HIGH",
            "reviewSummary": (
                "측정 결과 발볼 여유가 더 필요하고 전족부 쪽 하중이 높은 편입니다. "
                "이 제품은 발볼이 다소 슬림하게 느껴질 수 있어 구매 전에 앞쪽 압박 여부를 확인해 보세요."
            ),
            "reviews": [
                "발볼도 너무 칼발용도 아니라 완전 넓은볼이 아니라면 정사이즈가도 될거같습니다.",
                "발볼 크신분은 사이즈업 하셔야합니다! 런닝하기 좋습니다.",
                "와이드가 아니여서 발볼이 조금 좁은 것 같습니다.",
            ],
        },
        "HEEL": {
            "title": "뒤꿈치 적합도 보통",
            "riskLevel": "MEDIUM",
            "reviewSummary": (
                "측정 결과 뒤꿈치 쪽 하중이 높은 편입니다. "
                "장시간 착용 전 뒤꿈치가 안정적으로 잡히는지와 충격 부담을 확인해 보세요."
            ),
            "reviews": [
                "가볍고 쿠션감도 좋아서 오래 뛰어도 발목이 아프지 않고 좋습니다."
            ],
        },
        "INSOLE": {
            "title": "깔창 적합도 보통",
            "riskLevel": "MEDIUM",
            "reviewSummary": (
                "측정 결과 발 내부 습도가 높은 편이라 통기성이 중요합니다. "
                "쿠션이 부드럽다는 착화 경험은 반복되지만 장시간 착용감에는 개인차가 있을 수 있습니다."
            ),
            "reviews": [
                "너무 가볍고 쿠션감이 좋습니다.",
                "가볍고 쿠션감 좋고 발을 잘 잡아줘서 안정적입니다.",
                "발이 편하고 푹신합니다.",
            ],
        },
    }
    review_id = 100
    reasons = []
    for reason_type in ("FOREFOOT", "HEEL", "INSOLE"):
        data = reason_data[reason_type]
        reviews = []
        for review_text in data["reviews"]:
            review_id += 1
            reviews.append(
                {
                    "reviewId": review_id,
                    "reviewText": review_text,
                    "source": "MUSINSA",
                }
            )
        reasons.append(
            {
                "reasonType": reason_type,
                "title": data["title"],
                "riskLevel": data["riskLevel"],
                "reviewSummary": (
                    None
                    if reason_type in missing_reason_types
                    else data["reviewSummary"]
                ),
                "reviews": reviews,
            }
        )
    return ServerSavedRecommendation.model_validate(
        {
            "measurementSessionId": 55,
            "userId": 7,
            "shoeId": 90,
            "brandName": "호카",
            "shoeName": "(남성) 클리프톤 9 - 샌드 / 1127895-SSEG",
            "fitScore": 55.79,
            "pointSummary": (
                "발볼이 다소 슬림하게 느껴질 수 있어 발볼이 넓다면 여유 있는 사이즈를 함께 비교해 보세요. "
                "가볍고 쿠션이 부드럽다는 착화 경험이 반복됩니다."
                if include_point_summary
                else None
            ),
            "analyzedAt": "2026-09-06T12:30:00",
            "reasons": reasons,
        }
    )


def _shoe_55_characteristics() -> ServerShoeCharacteristics:
    return ServerShoeCharacteristics.model_validate(
        {
            "shoeId": 90,
            "summary": "발볼과 앞코 공간이 좁고 쿠션은 단단한 편입니다.",
            "characteristics": [
                {"type": "WIDTH_SPACE", "level": "LOW"},
                {"type": "TOEBOX_SPACE", "level": "LOW"},
                {"type": "CUSHION", "level": "LOW"},
            ],
        }
    )


def _assert_no_mechanical_summary_copy(
    test_case: unittest.TestCase,
    response_payload: dict,
) -> None:
    texts = [
        response_payload["pointSummary"],
        *(reason["reviewSummary"] for reason in response_payload["reasons"]),
    ]
    for text in texts:
        test_case.assertTrue(text)
        test_case.assertNotIn("정량 분석 기준", text)
        test_case.assertNotRegex(text, r"관련 착화 리뷰\s*\d+개")
        test_case.assertNotIn("정량 분석 결과만 안내", text)
    point_summary = response_payload["pointSummary"]
    test_case.assertNotRegex(
        point_summary,
        r"(?:발볼 공간|앞코 공간|쿠션감)은 "
        r"(?:낮은 편|보통 수준|높은 편)입니다",
    )
    test_case.assertNotIn("착화 시 가볍게 느껴지는 편입니다", point_summary)


class ShoePayloadContractTests(unittest.TestCase):
    def test_first_stage_payload_serializes_null_summaries_as_pending(self) -> None:
        payload = ShoeRecommendationForwardRequest(
            measurement_session_id=30,
            recommendations=[
                ShoeRecommendationItemPayload(
                    shoe_id=101,
                    fit_score=80.0,
                    reasons=_recommendation_reasons(),
                )
            ],
        )

        self.assertEqual(
            payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "recommendations": [
                    {
                        "shoeId": 101,
                        "fitScore": 80.0,
                        "pointSummary": None,
                        "reasons": [
                            {
                                "reasonType": "FOREFOOT",
                                "title": "FOREFOOT title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [1],
                            },
                            {
                                "reasonType": "HEEL",
                                "title": "HEEL title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [2],
                            },
                            {
                                "reasonType": "INSOLE",
                                "title": "INSOLE title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [3],
                            },
                        ],
                    }
                ],
            },
        )

    def test_first_stage_payload_requires_all_three_unique_reason_types(self) -> None:
        duplicate_reasons = _recommendation_reasons()
        duplicate_reasons[2] = ShoeRecommendationReasonPayload(
            reason_type="HEEL",
            title="duplicate",
            risk_level="MEDIUM",
            review_ids=[],
        )
        with self.assertRaises(ValidationError):
            ShoeRecommendationItemPayload(
                shoe_id=101,
                fit_score=80.0,
                reasons=duplicate_reasons,
            )

    def test_first_stage_payload_limits_evidence_to_three_unique_reviews(self) -> None:
        with self.assertRaises(ValidationError):
            ShoeRecommendationReasonPayload(
                reason_type="FOREFOOT",
                title="title",
                risk_level="LOW",
                review_ids=[1, 2, 3, 4],
            )
        with self.assertRaises(ValidationError):
            ShoeRecommendationReasonPayload(
                reason_type="FOREFOOT",
                title="title",
                risk_level="LOW",
                review_ids=[1, 1],
            )

    def test_first_stage_payload_rejects_an_empty_recommendation_batch(self) -> None:
        with self.assertRaises(ValidationError):
            ShoeRecommendationForwardRequest(
                measurement_session_id=30,
                recommendations=[],
            )

    def test_first_stage_payload_rejects_duplicate_shoe_ids(self) -> None:
        item = ShoeRecommendationItemPayload(
            shoe_id=101,
            fit_score=80.0,
            reasons=_recommendation_reasons(),
        )
        with self.assertRaisesRegex(ValidationError, "duplicate shoeId"):
            ShoeRecommendationForwardRequest(
                measurement_session_id=30,
                recommendations=[item, item],
            )

    def test_summary_callback_contract_contains_only_generated_text(self) -> None:
        payload = ShoeSummaryForwardRequest(
            measurement_session_id=30,
            point_summary="요약",
            reasons=[
                ShoeSummaryReasonPayload(
                    reason_type=reason_type,
                    review_summary=f"{reason_type} 요약",
                    review_ids=[index],
                )
                for index, reason_type in enumerate(
                    ("FOREFOOT", "HEEL", "INSOLE"), start=1
                )
            ],
        )
        self.assertEqual(
            payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "pointSummary": "요약",
                "reasons": [
                    {"reasonType": "FOREFOOT", "reviewSummary": "FOREFOOT 요약", "reviewIds": [1]},
                    {"reasonType": "HEEL", "reviewSummary": "HEEL 요약", "reviewIds": [2]},
                    {"reasonType": "INSOLE", "reviewSummary": "INSOLE 요약", "reviewIds": [3]},
                ],
            },
        )


class InternalTriggerSecurityTests(unittest.TestCase):
    def test_heavy_routes_reject_missing_or_wrong_internal_key_without_echo(self) -> None:
        routes = (
            ("/api/reports/shoe-recommendations", {"measurementSessionId": 30}),
            (
                "/api/shoes/summaries",
                {"shoeId": 101, "measurementSessionId": 30},
            ),
            (
                "/api/shoes/summaries/generate",
                {"shoeId": 101, "measurementSessionId": 30},
            ),
        )
        with patch.object(
            settings, "feetfit_server_internal_api_key", "super-secret-value"
        ):
            for route, body in routes:
                for supplied in (None, "wrong-key"):
                    headers = {"Authorization": "Bearer forwarded-token"}
                    if supplied is not None:
                        headers["X-Internal-Api-Key"] = supplied
                    with self.subTest(route=route, supplied=supplied):
                        response = TestClient(app).post(route, headers=headers, json=body)
                        self.assertEqual(response.status_code, 403)
                        self.assertNotIn("super-secret-value", response.text)
                        self.assertNotIn("wrong-key", response.text)


class ShoeRecommendationRouteContractTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(
            settings, "feetfit_server_internal_api_key", "test-internal-key"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_route_forwards_complete_batch_summaries(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(return_value=object())
        upstream_request = httpx.Request("POST", "http://server/api/shoes/recommendations")
        server_client.forward_recommendations = AsyncMock(
            return_value=httpx.Response(200, json={"saved": True}, request=upstream_request)
        )
        batch = ShoeRecommendationBatch(
            user_id=7,
            measurement_session_id=30,
            items=[
                ShoeFacts(
                    shoe_id=101,
                    fit_score=80.0,
                    point_summary="발볼이 넓다면 사이즈 업을 고려할 만합니다.",
                    reasons=[
                        ReasonFacts(
                            reason_type=reason_type,
                            score=80.0,
                            title="title",
                            risk_level="LOW",
                            review_ids=[],
                            review_summary=f"{reason_type} 개인화 설명",
                        )
                        for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
                    ],
                )
            ],
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client) as client_type,
            patch("app.api.routes.reports.compute_shoe_recommendations", return_value=batch) as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 200)
        client_type.assert_called_once_with("Bearer forwarded-token")
        server_client.fetch_recommendation_context.assert_awaited_once_with(30)
        compute.assert_called_once()
        forwarded = server_client.forward_recommendations.await_args.args[0]
        self.assertEqual(
            forwarded.recommendations[0].point_summary,
            "발볼이 넓다면 사이즈 업을 고려할 만합니다.",
        )
        self.assertEqual(
            [reason.review_summary for reason in forwarded.recommendations[0].reasons],
            [
                "FOREFOOT 개인화 설명",
                "HEEL 개인화 설명",
                "INSOLE 개인화 설명",
            ],
        )

    def test_internal_context_failure_does_not_fall_back_to_db_or_compute(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(
            side_effect=ShoeServerClientError("contract unavailable")
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 502)
        compute.assert_not_called()

    def test_local_server_configuration_failure_is_not_reported_as_bad_gateway(self) -> None:
        with (
            patch(
                "app.api.routes.reports.ShoeServerClient",
                side_effect=ShoeServerConfigurationError("internal API key is missing"),
            ),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "internal API key is missing")
        compute.assert_not_called()

    def test_internal_context_preserves_server_measurement_not_completed_400(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(
            side_effect=ShoeServerClientError(
                "measurement session is not completed",
                status_code=400,
            )
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "measurement session is not completed")
        compute.assert_not_called()

    def test_busy_recommendation_runtime_returns_conflict_not_generic_error(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(return_value=object())
        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch(
                "app.api.routes.reports.compute_shoe_recommendations",
                side_effect=ShoeRecommendationBusyError("BGE runtime is busy"),
            ),
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("BGE runtime is busy", response.json()["detail"])


class ShoeSummaryRouteContractTests(unittest.IsolatedAsyncioTestCase):
    def test_old_low_information_point_summary_is_not_preserved(self) -> None:
        self.assertFalse(
            shoe_routes._preservable_saved_summary(
                "디자인에 대한 만족도가 높은 편입니다. "
                "그 밖의 착용 특성은 여러 리뷰에서 같은 방향으로 반복된 정보가 "
                "충분하지 않아 단정하기 어렵습니다."
            )
        )
        self.assertTrue(
            shoe_routes._preservable_saved_summary(
                "전반적인 착화감은 편하게 느껴지는 편입니다."
            )
        )

    def test_old_thin_comfort_point_summary_is_not_preserved(self) -> None:
        self.assertFalse(
            shoe_routes._preservable_saved_summary(
                "쿠션감에 대한 만족도가 높고, 무게도 가볍게 느껴지는 편입니다. "
                "디자인에 대한 만족도가 높은 편입니다."
            )
        )

    def setUp(self) -> None:
        patcher = patch.object(
            settings, "feetfit_server_internal_api_key", "test-internal-key"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        shoe_routes._summary_inflight.clear()
        shoe_routes._summary_immediate_waiters.clear()
        shoe_routes._direct_summary_waiters.clear()
        self.addCleanup(shoe_routes._summary_inflight.clear)
        self.addCleanup(shoe_routes._summary_immediate_waiters.clear)
        self.addCleanup(shoe_routes._direct_summary_waiters.clear)

    async def test_direct_summary_returns_exact_save_contract_without_callback(
        self,
    ) -> None:
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(
            return_value=_saved_recommendation()
        )
        server_client.fetch_shoe_characteristics = AsyncMock(
            return_value=_shoe_characteristics()
        )
        server_client.save_summary = AsyncMock(return_value=None)
        generated = ShoeFitSummary(
            point_summary="처음부터 보여 줄 자연스러운 종합 요약",
            forefoot_summary="자연스러운 발볼 요약",
            heel_summary="자연스러운 뒤꿈치 요약",
            insole_summary="자연스러운 깔창 요약",
            forefoot_review_ids=[1],
            heel_review_ids=[2],
            insole_review_ids=[3],
        )

        with (
            patch(
                "app.api.routes.shoes.ShoeServerClient",
                return_value=server_client,
            ),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(return_value=generated),
            ) as llm,
            patch(
                "app.api.routes.shoes.build_immediate_shoe_summaries",
                return_value=generated,
            ) as immediate_builder,
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries/generate",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "measurementSessionId": 30,
                "pointSummary": "처음부터 보여 줄 자연스러운 종합 요약",
                "reasons": [
                    {
                        "reasonType": "FOREFOOT",
                        "reviewSummary": "자연스러운 발볼 요약",
                        "reviewIds": [1],
                    },
                    {
                        "reasonType": "HEEL",
                        "reviewSummary": "자연스러운 뒤꿈치 요약",
                        "reviewIds": [2],
                    },
                    {
                        "reasonType": "INSOLE",
                        "reviewSummary": "자연스러운 깔창 요약",
                        "reviewIds": [3],
                    },
                ],
            },
        )
        llm.assert_not_awaited()
        immediate_builder.assert_called_once()
        server_client.save_summary.assert_not_awaited()

    async def test_direct_summary_returns_non_null_immediate_copy_without_ollama(
        self,
    ) -> None:
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(
            return_value=_saved_recommendation()
        )
        server_client.fetch_shoe_characteristics = AsyncMock(
            return_value=_shoe_characteristics()
        )
        server_client.save_summary = AsyncMock(return_value=None)
        fallback = ShoeFitSummary(
            point_summary="검증 근거 기반 종합 요약",
            forefoot_summary="검증 근거 기반 발볼 요약",
            heel_summary="검증 근거 기반 뒤꿈치 요약",
            insole_summary="검증 근거 기반 깔창 요약",
            forefoot_review_ids=[1],
            heel_review_ids=[2],
            insole_review_ids=[3],
        )

        with (
            patch(
                "app.api.routes.shoes.ShoeServerClient",
                return_value=server_client,
            ),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(side_effect=ShoeFitCommentError("Ollama unavailable")),
            ) as llm,
            patch(
                "app.api.routes.shoes.build_immediate_shoe_summaries",
                return_value=fallback,
            ) as fallback_builder,
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries/generate",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["pointSummary"])
        self.assertTrue(
            all(reason["reviewSummary"] for reason in response.json()["reasons"])
        )
        fallback_builder.assert_called_once()
        llm.assert_not_awaited()
        server_client.save_summary.assert_not_awaited()

    async def test_direct_fallback_preserves_only_point_copy_and_rebuilds_reasons(
        self,
    ) -> None:
        saved = _shoe_55_saved_recommendation(missing_reason_types={"HEEL"})
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=saved)
        server_client.fetch_shoe_characteristics = AsyncMock(
            return_value=_shoe_55_characteristics()
        )
        server_client.save_summary = AsyncMock(return_value=None)

        with (
            patch(
                "app.api.routes.shoes.ShoeServerClient",
                return_value=server_client,
            ),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(side_effect=ShoeFitCommentError("Ollama unavailable")),
            ),
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries/generate",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"shoeId": 90, "measurementSessionId": 55},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        _assert_no_mechanical_summary_copy(self, payload)
        summaries = {
            reason["reasonType"]: reason["reviewSummary"]
            for reason in payload["reasons"]
        }
        self.assertEqual(payload["pointSummary"], saved.point_summary)
        self.assertNotIn("전족부 쪽 하중", summaries["FOREFOOT"])
        self.assertIn("발볼", summaries["FOREFOOT"])
        self.assertEqual(
            summaries["INSOLE"],
            "해당 영역과 관련된 실제 착용 리뷰가 부족하여 리뷰 기반 설명을 제공하기 어렵습니다.",
        )
        for unsupported_measurement_claim in ("측정 결과", "하중", "압력", "습도"):
            self.assertNotIn(
                unsupported_measurement_claim,
                summaries["HEEL"],
            )
        server_client.save_summary.assert_not_awaited()

    async def test_direct_fallback_without_saved_measurements_does_not_invent_them(
        self,
    ) -> None:
        saved = _shoe_55_saved_recommendation(
            missing_reason_types={"FOREFOOT", "HEEL", "INSOLE"},
            include_point_summary=False,
        )
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=saved)
        server_client.fetch_shoe_characteristics = AsyncMock(
            return_value=_shoe_55_characteristics()
        )
        server_client.save_summary = AsyncMock(return_value=None)

        with (
            patch(
                "app.api.routes.shoes.ShoeServerClient",
                return_value=server_client,
            ),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(side_effect=ShoeFitCommentError("Ollama unavailable")),
            ),
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries/generate",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"shoeId": 90, "measurementSessionId": 55},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        _assert_no_mechanical_summary_copy(self, payload)
        combined = " ".join(
            [
                payload["pointSummary"],
                *(reason["reviewSummary"] for reason in payload["reasons"]),
            ]
        )
        for unsupported_measurement_claim in ("측정 결과", "하중", "압력", "습도"):
            self.assertNotIn(unsupported_measurement_claim, combined)
        server_client.save_summary.assert_not_awaited()

    async def test_direct_summary_concurrent_requests_share_one_generation(
        self,
    ) -> None:
        payload = ShoeSummaryForwardRequest(
            measurement_session_id=30,
            point_summary="공유된 종합 요약",
            reasons=[
                ShoeSummaryReasonPayload(
                    reason_type=reason_type,
                    review_summary=f"{reason_type} 공유 요약",
                    review_ids=[index],
                )
                for index, reason_type in enumerate(
                    ("FOREFOOT", "HEEL", "INSOLE"), start=1
                )
            ],
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="forwarded-token"
        )

        async def delayed_result(*_args) -> ShoeSummaryForwardRequest:
            await asyncio.sleep(0.02)
            return payload

        with patch(
            "app.api.routes.shoes._generate_direct_summary_payload",
            new=AsyncMock(side_effect=delayed_result),
        ) as generate:
            first, second = await asyncio.gather(
                shoe_routes.generate_shoe_summary_for_detail(
                    ShoeSummaryGenerateRequest(
                        shoe_id=101, measurement_session_id=30
                    ),
                    credentials,
                    None,
                ),
                shoe_routes.generate_shoe_summary_for_detail(
                    ShoeSummaryGenerateRequest(
                        shoe_id=101, measurement_session_id=30
                    ),
                    credentials,
                    None,
                ),
            )

        self.assertEqual(first, payload)
        self.assertEqual(second, payload)
        generate.assert_awaited_once()
        self.assertNotIn((30, 101), shoe_routes._direct_summary_waiters)

    def test_direct_summary_request_rejects_unknown_legacy_switch(self) -> None:
        with self.assertRaises(ValidationError):
            ShoeSummaryGenerateRequest.model_validate(
                {
                    "shoeId": 101,
                    "measurementSessionId": 30,
                    "waitForImmediate": True,
                }
            )

    async def test_summary_trigger_requires_and_forwards_measurement_session(self) -> None:
        with patch(
            "app.api.routes.shoes._generate_and_save_summary",
            new=AsyncMock(return_value=None),
        ) as task:
            response = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 202)
        task.assert_awaited_once_with(101, 30, "Bearer forwarded-token")

        missing_session = TestClient(app).post(
            "/api/shoes/summaries",
            headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
            json={"shoeId": 101},
        )
        self.assertEqual(missing_session.status_code, 422)

    async def test_duplicate_summary_trigger_is_deduplicated_by_session_and_shoe(self) -> None:
        with patch(
            "app.api.routes.shoes._generate_and_save_summary",
            new=AsyncMock(return_value=None),
        ) as task:
            first = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )
            second = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(first.json(), {"accepted": True, "deduplicated": False})
        self.assertEqual(second.json(), {"accepted": True, "deduplicated": True})
        task.assert_awaited_once()

    async def test_wait_for_immediate_saves_before_accepting_then_only_refines_in_background(
        self,
    ) -> None:
        work = MagicMock()
        background_tasks = BackgroundTasks()
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="forwarded-token"
        )
        with (
            patch(
                "app.api.routes.shoes._prepare_and_save_immediate_summary",
                new=AsyncMock(return_value=work),
            ) as immediate,
            patch(
                "app.api.routes.shoes._generate_and_save_refinement_inner",
                new=AsyncMock(return_value=None),
            ) as refinement,
        ):
            response = await shoe_routes.trigger_shoe_summary_generation(
                ShoeSummaryTriggerRequest(
                    shoe_id=101,
                    measurement_session_id=30,
                    wait_for_immediate=True,
                ),
                background_tasks,
                credentials,
                None,
            )
            refinement.assert_not_awaited()
            self.assertEqual(len(background_tasks.tasks), 1)
            self.assertIn((30, 101), shoe_routes._summary_inflight)
            await background_tasks()

        self.assertEqual(
            response.model_dump(by_alias=True, exclude_none=True),
            {
                "accepted": True,
                "deduplicated": False,
                "immediateSaved": True,
            },
        )
        immediate.assert_awaited_once_with(101, 30, "Bearer forwarded-token")
        refinement.assert_awaited_once_with(101, 30, work)
        self.assertNotIn((30, 101), shoe_routes._summary_inflight)

    async def test_duplicate_wait_for_immediate_awaits_existing_save_outcome(self) -> None:
        reserved, future = shoe_routes._reserve_summary((30, 101))
        self.assertTrue(reserved)
        future.set_result(
            shoe_routes._ImmediateSummaryOutcome(
                saved=True,
                status_code=202,
                detail="Immediate shoe summary saved.",
            )
        )

        response = TestClient(app).post(
            "/api/shoes/summaries",
            headers={
                "Authorization": "Bearer forwarded-token",
                "X-Internal-Api-Key": "test-internal-key",
            },
            json={
                "shoeId": 101,
                "measurementSessionId": 30,
                "waitForImmediate": True,
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "accepted": True,
                "deduplicated": True,
                "immediateSaved": True,
            },
        )

    async def test_waiting_follower_does_not_duplicate_legacy_inflight_work(self) -> None:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="forwarded-token"
        )
        leader_tasks = BackgroundTasks()
        leader = await shoe_routes.trigger_shoe_summary_generation(
            ShoeSummaryTriggerRequest(
                shoe_id=101,
                measurement_session_id=30,
                wait_for_immediate=False,
            ),
            leader_tasks,
            credentials,
            None,
        )
        follower = asyncio.create_task(
            shoe_routes.trigger_shoe_summary_generation(
                ShoeSummaryTriggerRequest(
                    shoe_id=101,
                    measurement_session_id=30,
                    wait_for_immediate=True,
                ),
                BackgroundTasks(),
                credentials,
                None,
            )
        )
        await asyncio.sleep(0)

        self.assertFalse(follower.done())
        self.assertEqual(len(leader_tasks.tasks), 1)
        shoe_routes._publish_immediate_outcome(
            (30, 101),
            shoe_routes._ImmediateSummaryOutcome(
                saved=True,
                status_code=202,
                detail="Immediate shoe summary saved.",
            ),
        )
        follower_response = await follower

        self.assertEqual(
            leader.model_dump(by_alias=True, exclude_none=True),
            {"accepted": True, "deduplicated": False},
        )
        self.assertEqual(
            follower_response.model_dump(by_alias=True, exclude_none=True),
            {
                "accepted": True,
                "deduplicated": True,
                "immediateSaved": True,
            },
        )
        shoe_routes._release_summary((30, 101))

    async def test_wait_for_immediate_maps_upstream_failure_and_releases_inflight(
        self,
    ) -> None:
        with patch(
            "app.api.routes.shoes._prepare_and_save_immediate_summary",
            new=AsyncMock(side_effect=ShoeServerClientError("tunnel unavailable")),
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={
                    "shoeId": 101,
                    "measurementSessionId": 30,
                    "waitForImmediate": True,
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Feetfit_Server summary context or save request failed.",
        )
        self.assertNotIn((30, 101), shoe_routes._summary_inflight)

    async def test_wait_for_immediate_maps_prerequisite_and_generation_failures(
        self,
    ) -> None:
        failures = (
            (shoe_routes._MissingSavedRecommendationError("missing"), 409),
            (shoe_routes.ShoeFitCommentError("invalid evidence"), 422),
            (ShoeServerConfigurationError("missing key"), 500),
        )
        for error, expected_status in failures:
            with self.subTest(expected_status=expected_status), patch(
                "app.api.routes.shoes._prepare_and_save_immediate_summary",
                new=AsyncMock(side_effect=error),
            ):
                response = TestClient(app).post(
                    "/api/shoes/summaries",
                    headers={
                        "Authorization": "Bearer forwarded-token",
                        "X-Internal-Api-Key": "test-internal-key",
                    },
                    json={
                        "shoeId": 101,
                        "measurementSessionId": 30,
                        "waitForImmediate": True,
                    },
                )

            self.assertEqual(response.status_code, expected_status)
            self.assertNotIn((30, 101), shoe_routes._summary_inflight)

    async def test_refinement_scheduling_failure_releases_inflight(self) -> None:
        with (
            patch(
                "app.api.routes.shoes._prepare_and_save_immediate_summary",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.object(
                BackgroundTasks,
                "add_task",
                side_effect=RuntimeError("cannot schedule"),
            ),
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={
                    "shoeId": 101,
                    "measurementSessionId": 30,
                    "waitForImmediate": True,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "The Ollama refinement task could not be scheduled.",
        )
        self.assertNotIn((30, 101), shoe_routes._summary_inflight)
        self.assertNotIn((30, 101), shoe_routes._summary_immediate_waiters)

    async def test_wait_for_immediate_timeout_is_bounded_and_releases_inflight(
        self,
    ) -> None:
        cancelled: list[bool] = []

        async def never_finishes(*_args) -> None:
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(True)

        with (
            patch.object(shoe_routes, "_IMMEDIATE_WAIT_TIMEOUT_SECONDS", 0.01),
            patch(
                "app.api.routes.shoes._prepare_and_save_immediate_summary",
                new=AsyncMock(side_effect=never_finishes),
            ),
        ):
            response = TestClient(app).post(
                "/api/shoes/summaries",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={
                    "shoeId": 101,
                    "measurementSessionId": 30,
                    "waitForImmediate": True,
                },
            )

        self.assertEqual(response.status_code, 504)
        self.assertIn("0.01 seconds", response.json()["detail"])
        self.assertEqual(cancelled, [True])
        self.assertNotIn((30, 101), shoe_routes._summary_inflight)

    async def test_summary_uses_saved_server_context_and_callback_contract(self) -> None:
        reserved, immediate_future = shoe_routes._reserve_summary((30, 101))
        self.assertTrue(reserved)
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=_saved_recommendation())
        server_client.fetch_shoe_characteristics = AsyncMock(return_value=_shoe_characteristics())
        server_client.save_summary = AsyncMock(return_value=None)
        generated = ShoeFitSummary(
            point_summary="종합 요약",
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[1],
            heel_review_ids=[2],
            insole_review_ids=[3],
        )

        with (
            patch("app.api.routes.shoes.ShoeServerClient", return_value=server_client) as client_type,
            patch("app.api.routes.shoes.generate_shoe_summaries", new=AsyncMock(return_value=generated)) as llm,
            patch("app.api.routes.shoes.logger.info") as info,
        ):
            await _generate_and_save_summary(101, 30, "Bearer forwarded-token")

        client_type.assert_called_once_with("Bearer forwarded-token")
        server_client.fetch_saved_recommendation.assert_awaited_once_with(30, 101)
        server_client.fetch_shoe_characteristics.assert_awaited_once_with(101)
        llm.assert_awaited_once()
        self.assertEqual(llm.await_args.kwargs["characteristics"], _shoe_characteristics())
        llm_reasons = llm.await_args.kwargs["reasons"]
        self.assertEqual(llm_reasons[0].review_ids, [1])
        self.assertEqual(llm_reasons[0].review_texts, ["서버에 저장된 실제 리뷰 1"])
        self.assertEqual(server_client.save_summary.await_count, 2)
        immediate_payload = server_client.save_summary.await_args_list[0].args[1]
        self.assertIn("쿠션은 비교적 단단", immediate_payload.point_summary)
        saved_payload = server_client.save_summary.await_args.args[1]
        self.assertEqual(
            saved_payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "pointSummary": "종합 요약",
                "reasons": [
                    {"reasonType": "FOREFOOT", "reviewSummary": "발볼 요약", "reviewIds": [1]},
                    {"reasonType": "HEEL", "reviewSummary": "뒤꿈치 요약", "reviewIds": [2]},
                    {"reasonType": "INSOLE", "reviewSummary": "깔창 요약", "reviewIds": [3]},
                ],
            },
        )
        self.assertTrue(
            any(
                "Shoe summary saved to Feetfit_Server" in call.args[0]
                and call.args[1:3] == (101, 30)
                and len(call.args) == 7
                for call in info.call_args_list
            )
        )
        self.assertTrue(immediate_future.result().saved)
        self.assertNotIn((30, 101), shoe_routes._summary_inflight)
        self.assertNotIn((30, 101), shoe_routes._summary_immediate_waiters)

    async def test_unexpected_background_failure_is_logged_and_dropped(self) -> None:
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=_saved_recommendation())
        server_client.fetch_shoe_characteristics = AsyncMock(return_value=_shoe_characteristics())
        server_client.save_summary = AsyncMock(return_value=None)

        with (
            patch("app.api.routes.shoes.ShoeServerClient", return_value=server_client),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(side_effect=RuntimeError("unexpected Ollama failure")),
            ),
            patch("app.api.routes.shoes.logger.warning") as warning,
        ):
            await _generate_and_save_summary(101, 30, "Bearer forwarded-token")

        self.assertEqual(server_client.save_summary.await_count, 1)
        provisional = server_client.save_summary.await_args.args[1]
        self.assertIn("쿠션은 비교적 단단", provisional.point_summary)
        self.assertTrue(
            any(
                "예상하지 못한 오류" in call.args[0]
                for call in warning.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
