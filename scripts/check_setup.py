"""Check portable runtime files; optionally verify hashes and load every model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--load-models", action="store_true", help="Load all four active networks and camera calibration locally")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    from app.core.config import settings
    from app.core.weights import weights

    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    configured = {
        "foot_seg_yolo11n_best.pt": weights.foot_segmentation,
        "tinea_segformer_b2_teacher_student_best.pt": weights.tinea_pedis_segmentation,
        "wound_segformer_experiment_c_best.pt": weights.wound_segmentation,
        "plantar_contact_segformer_b0_real_finetune_brightness2_best.pt": weights.plantar_contact_segformer,
        "camera_calibration.npz": settings.aruco_camera_calibration,
    }
    checks, problems = [], []
    for entry in manifest["files"]:
        path = Path(configured[entry["filename"]])
        status = "ok"
        if not path.is_file():
            status = "missing"
        elif path.stat().st_size != entry["sizeBytes"]:
            status = "size_mismatch"
        elif args.verify_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                status = "sha256_mismatch"
        checks.append({"file": entry["filename"], "path": str(path), "status": status})
        if status != "ok":
            problems.append(f"{entry['filename']}: {status}")
    for path in (settings.aruco_source_dir / "aruco_foot_measure/core.py", ROOT / "segment_anything/__init__.py"):
        if not path.is_file():
            problems.append(f"Missing vendored source: {path}")
    loaded = []
    if args.load_models and not problems:
        settings.analysis_device = args.device
        settings.plantar_inference_device = args.device
        from ultralytics import YOLO
        from app.services.tinea_analysis import get_tinea_analyzer
        from app.services.wound_analysis import get_wound_analyzer
        from app.services.plantar_full_sam_analysis import get_plantar_contact_segmenter
        from app.services.integrated_foot_analysis import _load_aruco_runtime
        try:
            YOLO(str(weights.foot_segmentation))
            loaded.append("foot_yolo")
            get_tinea_analyzer()
            loaded.append("tinea_segformer_b2_student")
            get_wound_analyzer()
            loaded.append("wound_segformer")
            get_plantar_contact_segmenter(device=args.device)
            loaded.append("plantar_contact_segformer")
            runtime = _load_aruco_runtime()
            runtime.lens_module.load_lens_calibration(settings.aruco_camera_calibration)
            loaded.append("aruco_calibration")
        except Exception as exc:
            problems.append(f"Local model load failed: {type(exc).__name__}: {exc}")
    print(json.dumps({"projectRoot": str(ROOT), "activeWeightCount": 4, "requiredFileCount": len(checks), "weights": checks, "loaded": loaded,
                      "openaiModel": settings.openai_report_model,
                      "openaiKeyConfigured": bool(settings.openai_api_key),
                      "skinAnalysisScope": "dorsal_only", "problems": problems}, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
