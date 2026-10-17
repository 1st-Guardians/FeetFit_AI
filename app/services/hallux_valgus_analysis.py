from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from time import perf_counter

import cv2
import numpy as np

from app.core.config import settings
from app.core.weights import weights
from app.services.hallux_valgus_contour import (
    analyze_foot_mask,
    get_mask_bbox,
    visualization_crop_bbox,
    visualize_result,
)
from app.services.tinea_analysis import (
    AnalysisError,
    load_image,
    predict_foot_mask,
    validate_precomputed_foot_mask,
)


progress_logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class HalluxValgusAnalysisResult:
    analysis_png: bytes
    original_filename: str
    angle_degree: float
    analysis_text: str
    score: float
    metrics: dict


def angle_stage(angle: float) -> str:
    if angle < 15.0:
        return "normal or low visible-deviation range (<15 deg)"
    if angle < 20.0:
        return "mild visible-deviation range (15-20 deg)"
    if angle < 40.0:
        return "moderate visible-deviation range (20-40 deg)"
    return "severe visible-deviation range (>=40 deg)"


def side_analysis_text(foot_side: str, angle: float) -> str:
    side_label = "left foot" if foot_side == "left" else "right foot"
    return (
        f"The {side_label} visible toe alignment angle was measured as "
        f"{angle:.1f} degrees. This is in the {angle_stage(angle)}."
    )


def risk_score_from_angles(left_angle: float, right_angle: float) -> float:
    max_angle = max(left_angle, right_angle)
    min_angle = min(left_angle, right_angle)
    if max_angle < 15.0:
        score = 15.0 + (max_angle / 15.0) * 25.0
    elif max_angle < 20.0:
        score = 45.0 + ((max_angle - 15.0) / 5.0) * 20.0
    elif max_angle < 40.0:
        score = 70.0 + ((max_angle - 20.0) / 20.0) * 20.0
    else:
        score = 90.0 + min(10.0, (max_angle - 40.0) * 0.5)
    if min_angle >= 15.0:
        score += 4.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def score_analysis_text(left_angle: float, right_angle: float) -> str:
    left_abnormal = left_angle >= 15.0
    right_abnormal = right_angle >= 15.0
    if left_abnormal and right_abnormal:
        if left_angle >= right_angle:
            return (
                "양발 모두 무지외반 진행 가능성이 있으며, 특히 왼발의 "
                f"각도가 {left_angle:.1f}도로 더 높아 관리가 필요합니다."
            )
        return (
            "양발 모두 무지외반 진행 가능성이 있으며, 특히 오른발의 "
            f"각도가 {right_angle:.1f}도로 더 높아 관리가 필요합니다."
        )
    if left_abnormal:
        return (
            f"왼발의 무지외반 각도가 {left_angle:.1f}도로 관찰되어 "
            "주의와 관리가 필요합니다."
        )
    if right_abnormal:
        return (
            f"오른발의 무지외반 각도가 {right_angle:.1f}도로 관찰되어 "
            "주의와 관리가 필요합니다."
        )
    return "양발의 무지외반 각도가 정상 범위로 보여 현재 큰 위험 신호는 두드러지지 않습니다."


def _encode_analysis_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AnalysisError("Failed to encode hallux valgus analysis image.")
    return encoded.tobytes()


class HalluxValgusAnalyzer:
    """Run contour HVA v6 after the shared ChArUco/ArUco preparation stage."""

    def __init__(self) -> None:
        # The hardware flow supplies the rectified crop and shared mask. YOLO is
        # retained only for the legacy route that uploads one image per foot.
        self.foot_model = None
        self._foot_model_lock = threading.Lock()

    def get_foot_model(self):
        if self.foot_model is not None:
            return self.foot_model
        with self._foot_model_lock:
            if self.foot_model is None:
                if not weights.foot_segmentation.exists():
                    raise AnalysisError(
                        f"Foot segmentation model not found: {weights.foot_segmentation}"
                    )
                from ultralytics import YOLO

                self.foot_model = YOLO(str(weights.foot_segmentation))
        return self.foot_model

    def analyze(
        self,
        image_bytes: bytes,
        filename: str | None,
        foot_side: str,
        *,
        foot_mask: np.ndarray | None = None,
    ) -> HalluxValgusAnalysisResult:
        _, image_bgr = load_image(image_bytes)
        if foot_mask is None:
            analyzed_foot_mask, foot_found = predict_foot_mask(
                foot_model=self.get_foot_model(),
                image_bgr=image_bgr,
                conf=settings.hallux_seg_conf,
                min_component_area=settings.foot_min_component_area,
                iou_dup_threshold=settings.foot_iou_duplicate_threshold,
            )
            mask_source = "hallux_yolo"
            input_geometry = "legacy_single_foot_upload"
        else:
            analyzed_foot_mask = validate_precomputed_foot_mask(
                foot_mask,
                image_bgr.shape[:2],
            )
            foot_found = True
            mask_source = "shared_aruco_segmentation"
            input_geometry = (
                "shared_lens_corrected_aruco_rectified_aligned_rectangular_foot_crop"
            )

        if not foot_found:
            raise AnalysisError(
                f"No foot outline was detected for the {foot_side} hallux analysis."
            )

        try:
            contour_result = analyze_foot_mask(analyzed_foot_mask, foot_side)
        except (ValueError, cv2.error) as exc:
            raise AnalysisError(
                f"Contour-based hallux analysis failed for the {foot_side} foot: {exc}"
            ) from exc
        if contour_result is None:
            raise AnalysisError(
                f"Contour-based hallux analysis could not identify the required "
                f"toe and medial reference axes for the {foot_side} foot."
            )

        angle = float(contour_result["angle"])
        if not np.isfinite(angle):
            raise AnalysisError("Contour-based hallux angle calculation failed.")
        try:
            visualization = visualize_result(
                analyzed_foot_mask.shape,
                contour_result,
            )
        except (ValueError, cv2.error) as exc:
            raise AnalysisError(
                f"Failed to render the contour hallux visualization: {exc}"
            ) from exc

        analysis_score = round(
            min(
                float(contour_result["hallux_fit_inlier_ratio"]),
                float(contour_result["medial_fit_inlier_ratio"]),
            ),
            6,
        )
        bbox = get_mask_bbox(np.asarray(contour_result["cleaned_mask"]))
        visual_bbox = visualization_crop_bbox(
            analyzed_foot_mask.shape,
            np.asarray(contour_result["contour"], dtype=np.float64),
        )
        if visual_bbox is None:
            raise AnalysisError("Failed to resolve the hallux visualization crop.")
        visual_origin = np.asarray(visual_bbox[:2], dtype=np.float64)
        return HalluxValgusAnalysisResult(
            analysis_png=_encode_analysis_png(visualization),
            original_filename=Path(filename or "foot.jpg").name,
            angle_degree=round(angle, 1),
            analysis_text=side_analysis_text(foot_side, angle),
            score=analysis_score,
            metrics={
                "foot_outline_found": bool(foot_found),
                "foot_mask_source": mask_source,
                "input_geometry": input_geometry,
                "bbox": bbox,
                "score": analysis_score,
                "score_type": "minimum_axis_ransac_inlier_ratio",
                "algorithm": "contour_hva_v6",
                "angle_definition": (
                    "unsigned acute angle between hallux center axis and medial contour reference axis"
                ),
                "hallux_axis_point": np.asarray(
                    contour_result["hallux_point"]
                ).tolist(),
                "hallux_axis_direction": np.asarray(
                    contour_result["hallux_dir"]
                ).tolist(),
                "reference_axis_point": np.asarray(
                    contour_result["reference_point"]
                ).tolist(),
                "reference_axis_direction": np.asarray(
                    contour_result["reference_dir"]
                ).tolist(),
                "intersection": np.asarray(
                    contour_result["intersection"]
                ).tolist(),
                "geometry_coordinate_space": "rectified_foot_crop_pixels",
                "visualization_crop_bbox": visual_bbox,
                "visualization_coordinate_space": "output_png_pixels",
                "visualization_hallux_axis_point": (
                    np.asarray(contour_result["hallux_point"]) - visual_origin
                ).tolist(),
                "visualization_reference_axis_point": (
                    np.asarray(contour_result["reference_point"]) - visual_origin
                ).tolist(),
                "visualization_intersection": (
                    np.asarray(contour_result["intersection"]) - visual_origin
                ).tolist(),
                "foot_width_px": float(contour_result["foot_width"]),
                "hallux_fit_point_count": int(
                    contour_result["hallux_fit_point_count"]
                ),
                "medial_fit_point_count": int(
                    contour_result["medial_fit_point_count"]
                ),
                "hallux_fit_inlier_ratio": float(
                    contour_result["hallux_fit_inlier_ratio"]
                ),
                "medial_fit_inlier_ratio": float(
                    contour_result["medial_fit_inlier_ratio"]
                ),
                "visualization": (
                    "white_filled_foot_with_contour_and_axes_on_transparent_background"
                ),
                "visualization_width": int(visualization.shape[1]),
                "visualization_height": int(visualization.shape[0]),
            },
        )


_hallux_analyzer: HalluxValgusAnalyzer | None = None
_hallux_analyzer_lock = threading.Lock()


def get_hallux_valgus_analyzer() -> HalluxValgusAnalyzer:
    global _hallux_analyzer
    if _hallux_analyzer is not None:
        return _hallux_analyzer
    with _hallux_analyzer_lock:
        if _hallux_analyzer is None:
            _hallux_analyzer = HalluxValgusAnalyzer()
    return _hallux_analyzer


def analyze_hallux_valgus_image(
    image_bytes: bytes,
    filename: str | None,
    foot_side: str,
    *,
    foot_mask: np.ndarray | None = None,
) -> HalluxValgusAnalysisResult:
    if foot_side not in {"left", "right"}:
        raise AnalysisError(f"Unsupported foot side: {foot_side}")
    side_label = "왼발" if foot_side == "left" else "오른발"
    started_at = perf_counter()
    progress_logger.info(
        "[사진 분석][%s] 사진 입력을 받았습니다. 무지외반 분석을 시작합니다.",
        side_label,
    )
    analyzer = get_hallux_valgus_analyzer()
    result = analyzer.analyze(
        image_bytes,
        filename,
        foot_side,
        foot_mask=foot_mask,
    )
    progress_logger.info(
        "[사진 분석][%s] 무지외반 분석을 완료했습니다. | elapsedSeconds=%.2f",
        side_label,
        perf_counter() - started_at,
    )
    return result
