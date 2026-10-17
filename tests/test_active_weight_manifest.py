from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights


class ActiveWeightManifestTests(unittest.TestCase):
    def test_manifest_requires_exactly_four_active_weights_and_calibration(self):
        manifest = json.loads((PROJECT_ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
        actual = [entry["filename"] for entry in manifest["files"]]
        active = [Path(path).name for path in (weights.foot_segmentation, weights.tinea_pedis_segmentation,
                  weights.wound_segmentation, weights.plantar_contact_segformer)]
        self.assertEqual(len(set(active)), 4)
        self.assertEqual(len(actual), 5)
        self.assertEqual(set(actual), {*active, Path(settings.aruco_camera_calibration).name})
        self.assertNotIn("sam_vit_b_01ec64.pth", actual)
        self.assertNotIn("tinea_pedis_best.pt", actual)

    def test_setup_check_needs_no_sam_weight_even_when_legacy_paths_are_missing(self):
        # File-size checks are simulated so this test also runs on a fresh clone
        # without any separately distributed binary checkpoints.
        manifest = json.loads((PROJECT_ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
        sizes = {entry["filename"]: entry["sizeBytes"] for entry in manifest["files"]}
        real_is_file, real_stat = Path.is_file, Path.stat
        checked = []
        def is_file(path):
            checked.append(path.name)
            if path.parent == weights.root and path.name in sizes:
                return True
            return real_is_file(path)
        def stat(path, *args, **kwargs):
            if path.parent == weights.root and path.name in sizes:
                return SimpleNamespace(st_size=sizes[path.name])
            return real_stat(path, *args, **kwargs)
        namespace = runpy.run_path(str(PROJECT_ROOT / "scripts/check_setup.py"))
        captured = StringIO()
        with patch.object(weights, "sam_checkpoint", PROJECT_ROOT / "missing_sam.pth"), \
             patch.object(weights, "plantar_sam", PROJECT_ROOT / "missing_plantar_sam.pth"), \
             patch.object(Path, "is_file", is_file), patch.object(Path, "stat", stat), \
             patch.object(sys, "argv", ["check_setup.py"]), redirect_stdout(captured):
            code = namespace["main"]()
        result = json.loads(captured.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["activeWeightCount"], 4)
        self.assertEqual(result["requiredFileCount"], 5)
        self.assertNotIn("missing_sam.pth", checked)
        self.assertNotIn("missing_plantar_sam.pth", checked)


if __name__ == "__main__":
    unittest.main()
