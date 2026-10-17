from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from app.core.config import PROJECT_ROOT, Settings
from app.services import report_text_generation as report
from tests.test_report_text_generation import _result


class PortablePathsTests(unittest.TestCase):
    def test_paths_remain_project_relative_after_cwd_changes(self):
        previous = Path.cwd()
        directory = PROJECT_ROOT / ".build" / "cwd-policy-test"
        directory.mkdir(parents=True, exist_ok=True)
        with patch.dict(os.environ, {}, clear=True):
            try:
                os.chdir(directory)
                configured = Settings(_env_file=None, weights_dir="custom-models", backend_base_url="https://backend.example")
            finally:
                os.chdir(previous)
        self.assertEqual(configured.weights_dir, PROJECT_ROOT / "custom-models")
        self.assertEqual(configured.wound_segmentation_checkpoint.parent, configured.weights_dir)
        self.assertEqual(configured.plantar_contact_segformer_checkpoint.parent, configured.weights_dir)
        self.assertEqual(configured.aruco_camera_calibration.parent, configured.weights_dir)
        self.assertEqual(configured.aruco_source_dir, PROJECT_ROOT / "vendor/aruco")
        self.assertEqual(configured.sam_source_dir, PROJECT_ROOT)
        self.assertEqual(configured.tinea_report_endpoint, "https://backend.example/api/reports/tina-pedis")
        self.assertTrue(configured.measurement_status_endpoint_template.startswith("https://backend.example/"))

    def test_explicit_backend_endpoint_is_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            configured = Settings(_env_file=None, backend_base_url="https://main.example/", tinea_report_endpoint="https://separate.example/upload")
        self.assertEqual(configured.tinea_report_endpoint, "https://separate.example/upload")
        self.assertTrue(configured.pressure_heatmap_report_endpoint.startswith("https://main.example/api/"))


class OpenAIModelPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_repair_preserves_facts_and_uses_model_contract(self):
        responses = [
            report.httpx.Response(200, json={"status": "completed", "output_text": '{"wrong":"shape"}'}),
            report.httpx.Response(200, json={"status": "completed", "output_text": '{"value":"ok"}'}),
        ]
        post = AsyncMock(side_effect=responses)
        client = AsyncMock()
        client.__aenter__.return_value.post = post
        schema = {"type": "object", "properties": {"value": {"type": "string", "enum": ["ok"]}}, "required": ["value"], "additionalProperties": False}
        with patch.object(report.settings, "openai_api_key", "test-key"), patch.object(report.settings, "openai_report_model", "gpt-6-astra"), patch.object(report.httpx, "AsyncClient", return_value=client), patch.object(report.asyncio, "sleep", AsyncMock()):
            actual = await report._create_structured_response(system_prompt="Return the schema", user_payload={"measuredValue": 91}, schema_name="test_schema_repair", json_schema=schema)
        self.assertEqual(actual, {"value": "ok"})
        self.assertEqual(post.await_count, 2)
        first, second = [call.kwargs["json"] for call in post.await_args_list]
        self.assertEqual(first["input"][1], second["input"][1])
        self.assertNotEqual(first["input"][0], second["input"][0])
        self.assertFalse(second["store"])
        self.assertTrue(second["text"]["format"]["strict"])
        self.assertEqual(second["reasoning"], {"effort": "low"})
        self.assertGreaterEqual(second["max_output_tokens"], 8000)
        self.assertNotIn("temperature", second)

    async def test_standalone_gpt_cannot_reintroduce_plantar_skin_diagnosis(self):
        left, right = _result(94, 93), _result(92, 91)
        excluded = {"fungal_suspicion_safety_description": "발바닥에 무좀 의심 영역이 보입니다.", "skin_reaction_safety_description": "피부 관리를 해 주세요.", "total_score_description": "발을 관리해 주세요."}
        with patch.object(report, "_openai_enabled", return_value=True), patch.object(report, "_create_structured_response", AsyncMock(return_value=excluded)):
            result = await report.generate_tinea_report_text(left, right)
        self.assertEqual(result, report.build_fallback_tinea_report_text(left, right))
        self.assertNotIn("발바닥", result.fungal_suspicion_safety_description)


if __name__ == "__main__":
    unittest.main()
