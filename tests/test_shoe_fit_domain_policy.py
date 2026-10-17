from __future__ import annotations

import copy
import unittest

from app.schemas.shoe_server import ServerShoe, ServerUserFootState
from app.services.shoe.shoe_fit_policy import ShoeFitPolicyError, score_shoe_fit


def _metric(
    metric_id: int,
    characteristic: str,
    value: float,
    *,
    location: str | None = None,
    variant: str | None = None,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> dict:
    return {
        "metricId": metric_id,
        "canonicalCharacteristic": characteristic,
        "sourceMetricName": characteristic,
        "value": value,
        "averageValue": (minimum + maximum) / 2,
        "sourceMinValue": minimum,
        "sourceMaxValue": maximum,
        "unit": "score",
        "testedSize": "US 9",
        "methodName": "test",
        "methodVersion": "new method",
        "location": location,
        "variant": variant,
        "comparisonSampleCount": 100,
        "comparisonCohort": "running shoes",
        "rawValueText": str(value),
    }


def _shoe_payload() -> dict:
    return {
        "shoeId": 101,
        "brandName": "Example Brand",
        "shoeName": "Example Shoe",
        "modelCode": "EX-101",
        "musinsaUrl": "https://example.com/shoes/101",
        "price": 100000,
        "imageUrl": None,
        "overallRating": 4.5,
        "reviewCount": 1,
        "reviews": [
            {
                "reviewId": 1001,
                "rating": 5.0,
                "reviewText": "발볼이 편하고 쿠션이 좋습니다.",
                "source": "MUSINSA",
                "collectedAt": None,
            }
        ],
        "labMeasurements": [
            {
                "measurementId": 10100,
                "source": "RUNREPEAT",
                "sourceUrl": "https://runrepeat.com/example-shoe",
                "sourceBrandName": "Example Brand",
                "sourceShoeName": "Example Shoe",
                "sourceModelCode": None,
                "testedSize": "US 9",
                "capturedAt": "2026-09-01T00:00:00",
                "parserVersion": "test-v1",
                "internalLengthMm": 270.0,
                "widthMm": 108.0,
                "toeboxWidthMm": 78.0,
                "toeboxHeightMm": 28.0,
                "insoleThicknessMm": 5.0,
                "heelStackMm": 35.0,
                "forefootStackMm": 27.0,
                "rawMetrics": [
                    _metric(1, "WIDTH_SPACE", 52.0),
                    _metric(2, "TOEBOX_SPACE", 64.0, variant="width"),
                    _metric(3, "HEEL_HOLD", 90.0, location="HEEL"),
                    _metric(4, "SHOCK_ABSORPTION", 72.0, location="HEEL"),
                    _metric(5, "ENERGY_RETURN", 60.0, location="HEEL"),
                    _metric(6, "CUSHION", 30.0, variant="primary"),
                    _metric(7, "BREATHABILITY", 75.0),
                ],
            }
        ],
    }


def _state() -> ServerUserFootState:
    return ServerUserFootState.model_validate(
        {
            "dailyFootAnalysis": {
                "balanceScore": 70.0,
                "leftPressurePercent": 40.0,
                "rightPressurePercent": 60.0,
                "measuredLeftFootSizeMm": 265.0,
                "measuredRightFootSizeMm": 267.0,
                "leftFootWidthMm": 109.0,
                "rightFootWidthMm": 111.0,
                "avgTemperatureCelsius": 28.0,
                "avgHumidityPercent": 70.0,
                "typeText": None,
            },
            "tinaPedisAnalysis": None,
            "halluxValgusAnalysis": None,
            "staticPressureAnalyses": [],
            "pressureSensorReadings": [],
        }
    )


def _scores(payload: dict) -> dict[str, object]:
    shoe = ServerShoe.model_validate(payload)
    return {area.reason_type: area for area in score_shoe_fit(shoe, _state())}


def _set_metric(payload: dict, characteristic: str, value: float) -> None:
    metrics = payload["labMeasurements"][0]["rawMetrics"]
    target = next(
        metric
        for metric in metrics
        if metric["canonicalCharacteristic"] == characteristic
    )
    target["value"] = value
    target["rawValueText"] = str(value)


class ShoeFitDomainPolicyTests(unittest.TestCase):
    def test_each_area_uses_only_its_domain_characteristics(self) -> None:
        areas = _scores(_shoe_payload())

        self.assertEqual(
            tuple(component.name for component in areas["FOREFOOT"].components),
            ("WIDTH_SPACE", "TOEBOX_SPACE"),
        )
        self.assertEqual(
            tuple(component.name for component in areas["HEEL"].components),
            ("SHOCK_ABSORPTION", "ENERGY_RETURN", "CUSHION"),
        )
        self.assertEqual(
            tuple(component.name for component in areas["INSOLE"].components),
            ("BREATHABILITY",),
        )

    def test_excluded_heel_hold_and_top_level_width_do_not_change_scores(self) -> None:
        baseline_payload = _shoe_payload()
        changed_payload = copy.deepcopy(baseline_payload)
        _set_metric(changed_payload, "HEEL_HOLD", 0.0)
        changed_payload["labMeasurements"][0]["widthMm"] = 50.0
        changed_payload["labMeasurements"][0]["internalLengthMm"] = 400.0

        baseline = _scores(baseline_payload)
        changed = _scores(changed_payload)

        self.assertEqual(
            {key: (value.score, value.risk_level) for key, value in baseline.items()},
            {key: (value.score, value.risk_level) for key, value in changed.items()},
        )

    def test_metric_changes_are_isolated_to_their_declared_area(self) -> None:
        baseline = _scores(_shoe_payload())

        forefoot_payload = _shoe_payload()
        _set_metric(forefoot_payload, "WIDTH_SPACE", 100.0)
        _set_metric(forefoot_payload, "TOEBOX_SPACE", 100.0)
        forefoot_changed = _scores(forefoot_payload)
        self.assertNotEqual(baseline["FOREFOOT"].score, forefoot_changed["FOREFOOT"].score)
        self.assertEqual(baseline["HEEL"].score, forefoot_changed["HEEL"].score)
        self.assertEqual(baseline["INSOLE"].score, forefoot_changed["INSOLE"].score)

        heel_payload = _shoe_payload()
        _set_metric(heel_payload, "SHOCK_ABSORPTION", 0.0)
        _set_metric(heel_payload, "ENERGY_RETURN", 0.0)
        _set_metric(heel_payload, "CUSHION", 100.0)
        heel_changed = _scores(heel_payload)
        self.assertEqual(baseline["FOREFOOT"].score, heel_changed["FOREFOOT"].score)
        self.assertNotEqual(baseline["HEEL"].score, heel_changed["HEEL"].score)
        self.assertEqual(baseline["INSOLE"].score, heel_changed["INSOLE"].score)

        insole_payload = _shoe_payload()
        _set_metric(insole_payload, "BREATHABILITY", 0.0)
        insole_changed = _scores(insole_payload)
        self.assertEqual(baseline["FOREFOOT"].score, insole_changed["FOREFOOT"].score)
        self.assertEqual(baseline["HEEL"].score, insole_changed["HEEL"].score)
        self.assertNotEqual(baseline["INSOLE"].score, insole_changed["INSOLE"].score)

    def test_reviews_never_change_quantitative_scores_or_risks(self) -> None:
        baseline_payload = _shoe_payload()
        changed_payload = copy.deepcopy(baseline_payload)
        changed_payload["reviews"][0]["reviewText"] = (
            "발볼이 매우 좁고 뒤꿈치가 불편하며 쿠션과 통기성도 아쉽습니다."
        )

        baseline = _scores(baseline_payload)
        changed = _scores(changed_payload)

        self.assertEqual(
            {key: (value.score, value.risk_level) for key, value in baseline.items()},
            {key: (value.score, value.risk_level) for key, value in changed.items()},
        )

    def test_skin_reaction_changes_only_insole_breathability_need(self) -> None:
        safe_payload = _state().model_dump(by_alias=True)
        safe_payload["dailyFootAnalysis"]["avgHumidityPercent"] = 40.0
        safe_payload["tinaPedisAnalysis"] = {
            "fungalSuspicionSafetyScore": 100,
            "skinReactionSafetyScore": 100,
        }
        sensitive_payload = copy.deepcopy(safe_payload)
        sensitive_payload["tinaPedisAnalysis"]["skinReactionSafetyScore"] = 0

        shoe = ServerShoe.model_validate(_shoe_payload())
        safe = {
            area.reason_type: area
            for area in score_shoe_fit(
                shoe, ServerUserFootState.model_validate(safe_payload)
            )
        }
        sensitive = {
            area.reason_type: area
            for area in score_shoe_fit(
                shoe, ServerUserFootState.model_validate(sensitive_payload)
            )
        }

        self.assertEqual(safe["FOREFOOT"].score, sensitive["FOREFOOT"].score)
        self.assertEqual(safe["HEEL"].score, sensitive["HEEL"].score)
        self.assertNotEqual(safe["INSOLE"].score, sensitive["INSOLE"].score)

    def test_missing_breathability_does_not_fall_back_to_heel_characteristics(
        self,
    ) -> None:
        payload = _shoe_payload()
        metrics = payload["labMeasurements"][0]["rawMetrics"]
        payload["labMeasurements"][0]["rawMetrics"] = [
            metric
            for metric in metrics
            if metric["canonicalCharacteristic"] != "BREATHABILITY"
        ]

        with self.assertRaisesRegex(ShoeFitPolicyError, "No usable RunRepeat"):
            _scores(payload)


if __name__ == "__main__":
    unittest.main()
