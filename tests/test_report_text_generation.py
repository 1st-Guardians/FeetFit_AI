from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.services import report_text_generation as service


def setUpModule() -> None:
    enabled = patch.object(service.settings, "tinea_inflammation_enabled", True)
    enabled.start()
    unittest.addModuleCleanup(enabled.stop)


def _result(
    fungal_score: int,
    skin_score: int,
    *,
    fungal_location: str | None = None,
    skin_location: str | None = None,
):
    fungal_regions = (
        [{"location": fungal_location, "area_pixels": 20}]
        if fungal_location
        else []
    )
    inflammation_regions = (
        [{"location": skin_location, "area_pixels": 12}]
        if skin_location
        else []
    )
    return SimpleNamespace(
        fungal_safety_score=fungal_score,
        skin_reaction_safety_score=skin_score,
        metrics={
            "fungal_score_label": "mild",
            "inflammation_score_label": "mild",
            "overall_health_label": "mild",
            "fungal_pixel_ratio": 0.02,
            "inflammation_pixel_ratio": 0.01,
            "max_fungal_prob": 0.91,
            "max_inflammation_prob": 0.88,
            "fungal_regions": fungal_regions,
            "inflammation_regions": inflammation_regions,
            "score_reliability": "normal",
            "foot_outline_found": True,
        },
    )


class DorsalOnlyReportTests(unittest.IsolatedAsyncioTestCase):
    def test_plantar_scores_and_findings_never_influence_fallback(self):
        left, right = _result(97, 96, fungal_location="발등 중앙부"), _result(98, 97)
        baseline = service.build_fallback_tinea_report_text(left, right)
        for extra in ({"sole_left_result": _result(0, 0, fungal_location="발바닥 뒤꿈치")},
                      {"sole_left_result": _result(0, 0), "sole_right_result": _result(0, 0)}):
            with self.subTest(extra=tuple(extra)):
                actual = service.build_fallback_tinea_report_text(left, right, **extra)
                self.assertEqual(actual, baseline)
                self.assertNotIn("발바닥", actual.fungal_suspicion_safety_description)
                self.assertNotIn("발바닥", actual.skin_reaction_safety_description)
                self.assertNotIn("발바닥", actual.total_score_description)

    async def test_legacy_plantar_arguments_never_reach_gpt(self):
        generated = {"fungal_suspicion_safety_description": "발등 무좀 의심 소견",
                     "skin_reaction_safety_description": "발등 피부 소견",
                     "total_score_description": "양발 발등 관리 안내"}
        create = AsyncMock(return_value=generated)
        with patch.object(service, "_openai_enabled", return_value=True), patch.object(service, "_create_structured_response", create):
            actual = await service.generate_tinea_report_text(
                _result(91, 88), _result(87, 93), suspicion_map_png=b"dorsal-map", photo_overlay_png=b"dorsal-overlay",
                sole_left_result=_result(0, 0), sole_right_result=_result(0, 0),
                sole_suspicion_map_png=b"excluded-sole-map", sole_photo_overlay_png=b"excluded-sole-overlay")
        create.assert_awaited_once()
        request = create.await_args.kwargs
        self.assertEqual(actual.fungal_suspicion_safety_description, generated["fungal_suspicion_safety_description"])
        self.assertEqual(request["user_payload"]["left_foot"]["fungal_safety_score"], 91)
        self.assertEqual(request["user_payload"]["right_foot"]["skin_reaction_safety_score"], 93)
        self.assertNotIn("surfaces", request["user_payload"])
        self.assertEqual([im for _, im in request["images"]], [b"dorsal-map", b"dorsal-overlay"])

    async def test_partial_legacy_sole_pair_still_reports_dorsal_only(self):
        create = AsyncMock(side_effect=RuntimeError("offline"))
        left, right = _result(90, 91), _result(92, 93)
        with patch.object(service, "_openai_enabled", return_value=True), patch.object(service, "_create_structured_response", create):
            actual = await service.generate_tinea_report_text(left, right, sole_right_result=_result(0, 0))
        create.assert_awaited_once()
        self.assertEqual(actual, service.build_fallback_tinea_report_text(left, right))


if __name__ == "__main__":
    unittest.main()
