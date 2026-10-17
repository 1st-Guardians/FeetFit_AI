from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import cv2
import numpy as np

from .core import (
    MARKER_SIZE_MM,
    CalibrationCollector,
    CalibrationError,
    FootMeasurement,
    LockedCalibration,
    compare_marker_positions,
    contour_touches_roi,
    extract_foot_contours,
    make_foreground_mask,
    measure_foot_contour,
)
from .lens import (
    UndistortionMaps,
    apply_undistortion,
    build_undistortion_maps,
    load_lens_calibration,
)


@dataclass(frozen=True)
class MarkerImageQuality:
    marker_count: int
    minimum_marker_side_px: float | None
    median_marker_side_px: float | None
    minimum_pixels_per_module: float | None
    minimum_marker_sharpness: float | None
    median_marker_sharpness: float | None
    minimum_marker_contrast: float | None


@dataclass(frozen=True)
class DetectorResult:
    corners_by_id: dict[int, np.ndarray]
    duplicate_ids: tuple[int, ...]
    draw_corners: list[np.ndarray]
    draw_ids: np.ndarray | None
    preprocessing: str
    fallback_ids: tuple[int, ...]
    rejected_candidate_count: int
    rejected_candidate_count_by_profile: dict[str, int]
    dictionary_verified_ids: tuple[int, ...]
    code_hamming_distance_by_id: dict[int, int]
    ignored_decoded_ids: tuple[int, ...]
    bit_validation_rejected_count: int
    mapped_distorted_fallback_ids: tuple[int, ...]
    quality: MarkerImageQuality


@dataclass(frozen=True)
class _DetectorProfile:
    name: str
    detector: object | None
    parameters: object
    scale: float = 1.0


@dataclass(frozen=True)
class _DetectionPass:
    name: str
    grouped: dict[int, list[np.ndarray]]
    hamming_by_id: dict[int, list[int]]
    rejected_candidate_count: int
    bit_validation_rejected_count: int


class ArucoMarkerDetector:
    """Decode complete ArUco markers, with conservative soft-image passes.

    Every accepted marker must pass OpenCV's dictionary decoder *and* a second
    canonical payload check.  A black quadrilateral without the expected inner
    code is therefore never promoted to a marker.  The unmodified grayscale
    frame always wins; fallback passes only fill IDs that raw detection missed,
    so upscaling can never move a corner already measured in the source image.
    """

    def __init__(
        self,
        dictionary_name: str,
        *,
        expected_marker_ids: Iterable[int] | None = None,
        robust: bool = True,
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV ArUco module is missing. Install opencv-contrib-python."
            )
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.marker_count = int(self.dictionary.bytesList.shape[0])
        self.marker_bits = int(self.dictionary.markerSize)
        self._expected_bits = {
            marker_id: self._dictionary_bits(marker_id)
            for marker_id in range(self.marker_count)
        }
        self.expected_marker_ids = frozenset(
            int(value) for value in (expected_marker_ids or ())
        )
        self.robust = bool(robust)
        self._profiles = (
            self._make_profile("raw", pixels_per_cell=4),
            self._make_profile("dense_decode", pixels_per_cell=10),
            self._make_profile(
                "wide_threshold",
                pixels_per_cell=8,
                wide_threshold=True,
            ),
            self._make_profile(
                "small_marker_upscale",
                pixels_per_cell=8,
                wide_threshold=True,
                scale=1.5,
                aggressive_small_marker=True,
            ),
        )

    def _make_profile(
        self,
        name: str,
        *,
        pixels_per_cell: int,
        wide_threshold: bool = False,
        scale: float = 1.0,
        aggressive_small_marker: bool = False,
    ) -> _DetectorProfile:
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            if hasattr(parameters, "cornerRefinementMinAccuracy"):
                parameters.cornerRefinementMinAccuracy = 0.05
        if hasattr(parameters, "perspectiveRemovePixelPerCell"):
            parameters.perspectiveRemovePixelPerCell = pixels_per_cell
        if hasattr(parameters, "perspectiveRemoveIgnoredMarginPerCell"):
            parameters.perspectiveRemoveIgnoredMarginPerCell = 0.20
        # Metric measurement favours an exact payload over permissive ID
        # correction.  DICT_4X4_50 has only one correction bit; setting this to
        # 1.0 would greatly increase the number of random patterns accepted.
        if hasattr(parameters, "errorCorrectionRate"):
            parameters.errorCorrectionRate = 0.0
        if wide_threshold:
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 55
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.adaptiveThreshConstant = (
                3.0 if aggressive_small_marker else 5.0
            )
        if aggressive_small_marker:
            # A 0.01 perimeter rate admits physically unusable sub-module
            # candidates in a 1280px image.  0.02 still helps small markers
            # without weakening the internal dictionary check.
            parameters.minMarkerPerimeterRate = 0.02

        detector = (
            cv2.aruco.ArucoDetector(self.dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        return _DetectorProfile(name, detector, parameters, scale)

    def _dictionary_bits(self, marker_id: int) -> np.ndarray:
        """Return the canonical payload for one dictionary ID."""

        if hasattr(self.dictionary, "getMarkerBits"):
            bits = self.dictionary.getMarkerBits(int(marker_id))
            return np.asarray(bits, dtype=np.uint8).reshape(
                self.marker_bits, self.marker_bits
            )

        # Compatibility path for older opencv-contrib builds.  Generating the
        # official marker still uses the selected Dictionary, not a handmade
        # square or an inferred ID.
        cells = self.marker_bits + 2
        side = cells * 12
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(
                self.dictionary, int(marker_id), side, borderBits=1
            )
        else:
            marker = np.empty((side, side), dtype=np.uint8)
            cv2.aruco.drawMarker(
                self.dictionary, int(marker_id), side, marker, 1
            )
        cell = side // cells
        return np.asarray(
            [
                [
                    int(
                        np.mean(
                            marker[
                                (row + 1) * cell + cell // 4 :
                                (row + 2) * cell - cell // 4,
                                (column + 1) * cell + cell // 4 :
                                (column + 2) * cell - cell // 4,
                            ]
                        )
                        >= 127.5
                    )
                    for column in range(self.marker_bits)
                ]
                for row in range(self.marker_bits)
            ],
            dtype=np.uint8,
        )

    def _validate_decoded_payload(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        parameters: object,
    ) -> tuple[bool, int]:
        """Verify black border, canonical direction and inner dictionary bits.

        ``detectMarkers`` has already decoded the marker.  This independent
        sampling is deliberately a second gate: it confirms that the returned
        corner order rectifies the payload to the claimed ID.  It never assigns
        an ID to a rejected quadrilateral.
        """

        border_bits = max(1, int(getattr(parameters, "markerBorderBits", 1)))
        cells = self.marker_bits + 2 * border_bits
        pixels_per_cell = max(
            10, int(getattr(parameters, "perspectiveRemovePixelPerCell", 4))
        )
        side = cells * pixels_per_cell
        destination = np.float32(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]]
        )
        source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        transform = cv2.getPerspectiveTransform(source, destination)
        patch = cv2.warpPerspective(
            gray,
            transform,
            (side, side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        _, binary = cv2.threshold(
            patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        margin = max(1, int(round(pixels_per_cell * 0.20)))
        sampled = np.zeros((cells, cells), dtype=np.uint8)
        for row in range(cells):
            for column in range(cells):
                cell = binary[
                    row * pixels_per_cell + margin :
                    (row + 1) * pixels_per_cell - margin,
                    column * pixels_per_cell + margin :
                    (column + 1) * pixels_per_cell - margin,
                ]
                sampled[row, column] = int(np.mean(cell) >= 127.5)

        border_values = np.concatenate(
            (
                sampled[:border_bits].reshape(-1),
                sampled[-border_bits:].reshape(-1),
                sampled[border_bits:-border_bits, :border_bits].reshape(-1),
                sampled[border_bits:-border_bits, -border_bits:].reshape(-1),
            )
        )
        border_errors = int(np.count_nonzero(border_values))
        maximum_border_error_rate = float(
            getattr(parameters, "maxErroneousBitsInBorderRate", 0.35)
        )
        maximum_border_errors = int(
            math.floor(border_values.size * maximum_border_error_rate)
        )

        payload = sampled[
            border_bits : border_bits + self.marker_bits,
            border_bits : border_bits + self.marker_bits,
        ]
        expected = self._expected_bits[int(marker_id)]
        hamming = int(np.count_nonzero(payload != expected))
        maximum_hamming = int(
            math.floor(
                float(getattr(self.dictionary, "maxCorrectionBits", 0))
                * float(getattr(parameters, "errorCorrectionRate", 0.0))
            )
        )
        return (
            border_errors <= maximum_border_errors
            and hamming <= maximum_hamming,
            hamming,
        )

    def _detect_once(
        self, gray: np.ndarray, profile: _DetectorProfile
    ) -> _DetectionPass:
        working = gray
        if profile.scale != 1.0:
            working = cv2.resize(
                gray,
                None,
                fx=profile.scale,
                fy=profile.scale,
                interpolation=cv2.INTER_LANCZOS4,
            )
        if profile.detector is not None:
            corners, ids, rejected = profile.detector.detectMarkers(working)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                working,
                self.dictionary,
                parameters=profile.parameters,
            )

        grouped: dict[int, list[np.ndarray]] = {}
        hamming_by_id: dict[int, list[int]] = {}
        bit_validation_rejected_count = 0
        if ids is not None:
            for marker_corners, marker_id_array in zip(corners, ids):
                marker_id = int(np.asarray(marker_id_array).reshape(-1)[0])
                working_points = np.asarray(
                    marker_corners, dtype=np.float32
                ).reshape(4, 2)
                verified, hamming = self._validate_decoded_payload(
                    working, working_points, marker_id, profile.parameters
                )
                if not verified:
                    bit_validation_rejected_count += 1
                    continue
                points = working_points
                if profile.scale != 1.0:
                    points = points / np.float32(profile.scale)
                grouped.setdefault(marker_id, []).append(points)
                hamming_by_id.setdefault(marker_id, []).append(hamming)
        return _DetectionPass(
            profile.name,
            grouped,
            hamming_by_id,
            len(rejected),
            bit_validation_rejected_count,
        )

    @staticmethod
    def _marker_side(corners: np.ndarray) -> float:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        return float(
            np.median(
                [
                    np.linalg.norm(points[(index + 1) % 4] - points[index])
                    for index in range(4)
                ]
            )
        )

    @classmethod
    def _same_marker_location(
        cls, first: np.ndarray, second: np.ndarray
    ) -> bool:
        first_points = np.asarray(first, dtype=np.float32).reshape(4, 2)
        second_points = np.asarray(second, dtype=np.float32).reshape(4, 2)
        reference_side = max(
            1.0,
            min(cls._marker_side(first_points), cls._marker_side(second_points)),
        )
        centre_distance = float(
            np.linalg.norm(first_points.mean(axis=0) - second_points.mean(axis=0))
        )
        corner_rms = float(
            np.sqrt(np.mean(np.sum((first_points - second_points) ** 2, axis=1)))
        )
        return (
            centre_distance <= reference_side * 0.20
            and corner_rms <= reference_side * 0.05
        )

    @staticmethod
    def _quality(
        frame_bgr: np.ndarray, corners_by_id: Mapping[int, np.ndarray]
    ) -> MarkerImageQuality:
        if not corners_by_id:
            return MarkerImageQuality(0, None, None, None, None, None, None)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        destination = np.float32([[0, 0], [119, 0], [119, 119], [0, 119]])
        sides: list[float] = []
        sharpness: list[float] = []
        contrast: list[float] = []
        for corners in corners_by_id.values():
            points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            sides.append(ArucoMarkerDetector._marker_side(points))
            transform = cv2.getPerspectiveTransform(points, destination)
            patch = cv2.warpPerspective(gray, transform, (120, 120))
            sharpness.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
            low, high = np.percentile(patch, (10.0, 90.0))
            contrast.append(float(high - low))
        side_array = np.asarray(sides, dtype=np.float64)
        return MarkerImageQuality(
            marker_count=len(corners_by_id),
            minimum_marker_side_px=float(np.min(side_array)),
            median_marker_side_px=float(np.median(side_array)),
            minimum_pixels_per_module=float(np.min(side_array) / 6.0),
            minimum_marker_sharpness=float(np.min(sharpness)),
            median_marker_sharpness=float(np.median(sharpness)),
            minimum_marker_contrast=float(np.min(contrast)),
        )

    def _build_result(
        self,
        frame_bgr: np.ndarray,
        detection_passes: list[_DetectionPass],
    ) -> DetectorResult:
        raw = detection_passes[0]
        all_decoded_ids = {
            marker_id
            for detection_pass in detection_passes
            for marker_id in detection_pass.grouped
        }
        allowed_ids = (
            set(self.expected_marker_ids)
            if self.expected_marker_ids
            else all_decoded_ids
        )
        ignored_decoded_ids = all_decoded_ids.difference(allowed_ids)
        selected: dict[int, np.ndarray] = {
            marker_id: values[0]
            for marker_id, values in raw.grouped.items()
            if marker_id in allowed_ids and len(values) == 1
        }
        selected_hamming: dict[int, int] = {
            marker_id: raw.hamming_by_id[marker_id][0]
            for marker_id in selected
        }
        duplicate_ids = {
            marker_id
            for marker_id, values in raw.grouped.items()
            if marker_id in allowed_ids and len(values) > 1
        }
        fallback_ids: set[int] = set()
        fallback_sources: dict[int, str] = {}

        if self.expected_marker_ids:
            for marker_id in sorted(self.expected_marker_ids):
                if marker_id in selected or marker_id in duplicate_ids:
                    continue
                candidates: list[tuple[str, np.ndarray, int]] = []
                for detection_pass in detection_passes[1:]:
                    values = detection_pass.grouped.get(marker_id, [])
                    if len(values) == 1:
                        candidates.append(
                            (
                                detection_pass.name,
                                values[0],
                                detection_pass.hamming_by_id[marker_id][0],
                            )
                        )
                if not candidates:
                    continue
                first_name, first_corners, first_hamming = candidates[0]
                if any(
                    not self._same_marker_location(first_corners, corners)
                    for _, corners, _ in candidates[1:]
                ):
                    duplicate_ids.add(marker_id)
                    continue
                overlaps_other_id = any(
                    self._same_marker_location(first_corners, existing)
                    for existing_id, existing in selected.items()
                    if existing_id != marker_id
                )
                if overlaps_other_id:
                    continue
                selected[marker_id] = first_corners
                selected_hamming[marker_id] = first_hamming
                fallback_ids.add(marker_id)
                fallback_sources[marker_id] = first_name

        preprocessing = "raw"
        if fallback_ids:
            used_profiles = []
            for marker_id in sorted(fallback_ids):
                profile_name = fallback_sources[marker_id]
                if profile_name not in used_profiles:
                    used_profiles.append(profile_name)
            preprocessing += "+" + "+".join(used_profiles)

        ordered_ids = sorted(selected)
        draw_corners = [
            selected[marker_id].reshape(1, 4, 2) for marker_id in ordered_ids
        ]
        draw_ids = (
            np.asarray(ordered_ids, dtype=np.int32).reshape(-1, 1)
            if ordered_ids
            else None
        )
        rejected_by_profile = {
            item.name: item.rejected_candidate_count for item in detection_passes
        }
        return DetectorResult(
            corners_by_id=selected,
            duplicate_ids=tuple(sorted(duplicate_ids)),
            draw_corners=draw_corners,
            draw_ids=draw_ids,
            preprocessing=preprocessing,
            fallback_ids=tuple(sorted(fallback_ids)),
            rejected_candidate_count=sum(
                item.rejected_candidate_count for item in detection_passes
            ),
            rejected_candidate_count_by_profile=rejected_by_profile,
            dictionary_verified_ids=tuple(ordered_ids),
            code_hamming_distance_by_id={
                marker_id: int(selected_hamming[marker_id])
                for marker_id in ordered_ids
            },
            ignored_decoded_ids=tuple(sorted(ignored_decoded_ids)),
            bit_validation_rejected_count=sum(
                item.bit_validation_rejected_count for item in detection_passes
            ),
            mapped_distorted_fallback_ids=(),
            quality=self._quality(frame_bgr, selected),
        )

    def detect(
        self, frame_bgr: np.ndarray, *, robust: bool | None = None
    ) -> DetectorResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detection_passes = [self._detect_once(gray, self._profiles[0])]
        use_fallback = self.robust if robust is None else bool(robust)
        raw_known = set(detection_passes[0].grouped).intersection(
            self.expected_marker_ids
        )
        raw_duplicates = {
            marker_id
            for marker_id, values in detection_passes[0].grouped.items()
            if len(values) > 1
        }
        needs_fallback = (
            use_fallback
            and bool(self.expected_marker_ids)
            and (
                raw_known != self.expected_marker_ids
                or bool(raw_duplicates.intersection(self.expected_marker_ids))
            )
        )
        if needs_fallback:
            for profile in self._profiles[1:]:
                detection_passes.append(self._detect_once(gray, profile))
                unique_known = {
                    marker_id
                    for marker_id, values in detection_passes[-1].grouped.items()
                    if len(values) == 1 and marker_id in self.expected_marker_ids
                }
                if unique_known == self.expected_marker_ids:
                    break
        return self._build_result(frame_bgr, detection_passes)

    def merge_mapped_distorted_fallback(
        self,
        frame_bgr: np.ndarray,
        primary: DetectorResult,
        distorted: DetectorResult,
        corner_transform: Callable[[np.ndarray], np.ndarray],
    ) -> DetectorResult:
        """Fill missing IDs with decoded, point-corrected sensor corners.

        Nonlinear lens remapping interpolates pixels and can make an otherwise
        readable edge marker fail payload decoding.  The fallback decodes the
        complete ArUco pattern in the original sensor image, then maps only its
        four verified black outer corners into the corrected image.  A rejected
        quadrilateral or a plain black square is never assigned an ID here.
        """

        selected = {
            marker_id: np.asarray(corners, dtype=np.float32).reshape(4, 2)
            for marker_id, corners in primary.corners_by_id.items()
        }
        hamming = dict(primary.code_hamming_distance_by_id)
        duplicate_ids = set(primary.duplicate_ids).union(distorted.duplicate_ids)
        mapped_ids = set(primary.mapped_distorted_fallback_ids)

        for marker_id in sorted(self.expected_marker_ids):
            if marker_id not in distorted.corners_by_id:
                continue
            mapped = np.asarray(
                corner_transform(distorted.corners_by_id[marker_id]),
                dtype=np.float32,
            ).reshape(4, 2)
            if not np.isfinite(mapped).all():
                continue
            if marker_id in selected:
                if not self._same_marker_location(selected[marker_id], mapped):
                    duplicate_ids.add(marker_id)
                continue
            if marker_id in duplicate_ids:
                continue
            overlaps_other_id = any(
                self._same_marker_location(mapped, existing)
                for existing_id, existing in selected.items()
                if existing_id != marker_id
            )
            if overlaps_other_id:
                continue
            selected[marker_id] = mapped
            hamming[marker_id] = int(
                distorted.code_hamming_distance_by_id[marker_id]
            )
            mapped_ids.add(marker_id)

        ordered_ids = sorted(selected)
        draw_corners = [
            selected[marker_id].reshape(1, 4, 2) for marker_id in ordered_ids
        ]
        draw_ids = (
            np.asarray(ordered_ids, dtype=np.int32).reshape(-1, 1)
            if ordered_ids
            else None
        )
        rejected_by_profile = dict(primary.rejected_candidate_count_by_profile)
        rejected_by_profile.update(
            {
                f"distorted:{name}": count
                for name, count in distorted.rejected_candidate_count_by_profile.items()
            }
        )
        preprocessing = primary.preprocessing
        if mapped_ids:
            preprocessing += "+distorted_dictionary_decode_point_map"
        fallback_ids = set(primary.fallback_ids).union(mapped_ids)
        return DetectorResult(
            corners_by_id=selected,
            duplicate_ids=tuple(sorted(duplicate_ids)),
            draw_corners=draw_corners,
            draw_ids=draw_ids,
            preprocessing=preprocessing,
            fallback_ids=tuple(sorted(fallback_ids)),
            rejected_candidate_count=sum(rejected_by_profile.values()),
            rejected_candidate_count_by_profile=rejected_by_profile,
            dictionary_verified_ids=tuple(ordered_ids),
            code_hamming_distance_by_id={
                marker_id: int(hamming[marker_id]) for marker_id in ordered_ids
            },
            ignored_decoded_ids=tuple(
                sorted(
                    set(primary.ignored_decoded_ids).union(
                        distorted.ignored_decoded_ids
                    )
                )
            ),
            bit_validation_rejected_count=(
                primary.bit_validation_rejected_count
                + distorted.bit_validation_rejected_count
            ),
            mapped_distorted_fallback_ids=tuple(sorted(mapped_ids)),
            quality=self._quality(frame_bgr, selected),
        )


class RollingLength:
    def __init__(self, window_size: int, minimum_stable_frames: int) -> None:
        self.values: deque[float] = deque(maxlen=window_size)
        self.minimum_stable_frames = minimum_stable_frames

    def clear(self) -> None:
        self.values.clear()

    def add(self, value: float) -> tuple[float, float, bool]:
        if self.values:
            current_median = float(np.median(self.values))
            if abs(value - current_median) > 25.0:
                self.values.clear()
        self.values.append(float(value))
        array = np.asarray(self.values, dtype=np.float64)
        median = float(np.median(array))
        spread = float(1.4826 * np.median(np.abs(array - median)))
        stable = len(array) >= self.minimum_stable_frames and spread <= 3.0
        return median, spread, stable


@dataclass
class AppConfig:
    marker_ids: tuple[int, ...]
    marker_size_mm: float = MARKER_SIZE_MM
    calibration_frames: int = 30
    foreground_threshold: int = 24
    minimum_live_markers: int = 2
    maximum_camera_motion_px: float = 5.0
    maximum_marker_spread_mm: float = 10.0
    minimum_measurement_markers: int = 3
    minimum_foot_area_mm2: float = 3_000.0
    maximum_foot_area_mm2: float = 60_000.0
    minimum_length_mm: float = 100.0
    maximum_length_mm: float = 400.0
    minimum_width_mm: float = 30.0
    maximum_width_mm: float = 200.0
    rolling_window: int = 15
    minimum_stable_frames: int = 8
    minimum_marker_side_px: float = 60.0
    minimum_marker_sharpness: float = 12.0
    minimum_marker_contrast: float = 50.0
    swap_feet: bool = False


class FootMeasurementApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.calibration: LockedCalibration | None = None
        self.collector: CalibrationCollector | None = None
        self.status = "Remove feet and press B to calibrate"
        self.debug_mask = False
        self.last_mask: np.ndarray | None = None
        self.last_score: np.ndarray | None = None
        self.last_detection_quality: MarkerImageQuality | None = None
        self.last_detection_preprocessing = "raw"
        self.last_fallback_ids: tuple[int, ...] = ()
        self.last_measurements: dict[
            str, tuple[FootMeasurement, float, float, bool]
        ] = {}
        self.filters = {
            "LEFT": RollingLength(config.rolling_window, config.minimum_stable_frames),
            "RIGHT": RollingLength(config.rolling_window, config.minimum_stable_frames),
        }

    def start_calibration(self) -> None:
        self.collector = CalibrationCollector(
            self.config.marker_ids, target_frames=self.config.calibration_frames
        )
        self.calibration = None
        self._clear_filters()
        self.status = "Calibrating - keep the board empty and still"

    def reset(self) -> None:
        self.collector = None
        self.calibration = None
        self.last_mask = None
        self.last_score = None
        self.last_measurements = {}
        self._clear_filters()
        self.status = "Remove feet and press B to calibrate"

    def swap_labels(self) -> None:
        self.config.swap_feet = not self.config.swap_feet
        self._clear_filters()

    def _clear_filters(self) -> None:
        for rolling_filter in self.filters.values():
            rolling_filter.clear()

    def _label_for_side(self, side: str) -> str:
        image_left_is_left = not self.config.swap_feet
        if side == "image_left":
            return "LEFT" if image_left_is_left else "RIGHT"
        return "RIGHT" if image_left_is_left else "LEFT"

    def _finish_calibration_if_ready(self) -> None:
        if self.collector is None or not self.collector.complete:
            return
        try:
            self.calibration = self.collector.finalize(self.config.marker_size_mm)
            self.status = (
                f"Calibration locked with {len(self.calibration.reference_corners)} markers"
            )
        except CalibrationError as error:
            self.calibration = None
            self.status = f"Calibration failed: {error}"
        finally:
            self.collector = None

    def capture_quality_problem(
        self, detector_result: DetectorResult
    ) -> str | None:
        quality = detector_result.quality
        if quality.minimum_marker_side_px is None:
            return "no complete ArUco marker"
        if quality.minimum_marker_side_px < self.config.minimum_marker_side_px:
            return (
                f"small marker {quality.minimum_marker_side_px:.0f}px; "
                f"need {self.config.minimum_marker_side_px:.0f}px"
            )
        if (
            quality.minimum_marker_sharpness is not None
            and quality.minimum_marker_sharpness
            < self.config.minimum_marker_sharpness
        ):
            return (
                f"soft focus {quality.minimum_marker_sharpness:.1f}; "
                f"need {self.config.minimum_marker_sharpness:.1f}"
            )
        if (
            quality.minimum_marker_contrast is not None
            and quality.minimum_marker_contrast
            < self.config.minimum_marker_contrast
        ):
            return (
                f"low marker contrast {quality.minimum_marker_contrast:.0f}; "
                f"need {self.config.minimum_marker_contrast:.0f}"
            )
        return None

    def process_frame(
        self, frame_bgr: np.ndarray, detector_result: DetectorResult
    ) -> np.ndarray:
        self.last_detection_quality = detector_result.quality
        self.last_detection_preprocessing = detector_result.preprocessing
        self.last_fallback_ids = detector_result.fallback_ids
        annotated = frame_bgr.copy()
        if detector_result.draw_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                annotated, detector_result.draw_corners, detector_result.draw_ids
            )

        expected_current = {
            marker_id: corners
            for marker_id, corners in detector_result.corners_by_id.items()
            if marker_id in self.config.marker_ids
        }
        unexpected = sorted(
            set(detector_result.ignored_decoded_ids).union(
                set(detector_result.corners_by_id).difference(
                    self.config.marker_ids
                )
            )
        )

        if self.collector is not None:
            duplicate_expected = sorted(
                set(detector_result.duplicate_ids).intersection(self.config.marker_ids)
            )
            if duplicate_expected:
                self.status = (
                    "Duplicate marker IDs visible: "
                    + ",".join(str(value) for value in duplicate_expected)
                )
            elif len(expected_current) != len(self.config.marker_ids):
                missing = sorted(set(self.config.marker_ids).difference(expected_current))
                self.status = (
                    "Calibration paused - show all markers; missing "
                    + ",".join(str(value) for value in missing)
                )
            elif (quality_problem := self.capture_quality_problem(detector_result)):
                self.status = "Calibration paused - " + quality_problem
            else:
                try:
                    self.collector.add_frame(frame_bgr, expected_current)
                except CalibrationError as error:
                    self.collector = None
                    self.status = f"Calibration failed: {error}"
                    self._draw_header(annotated, len(expected_current))
                    return annotated
                self.status = (
                    f"Calibrating {self.collector.frame_count}/"
                    f"{self.collector.target_frames} - keep board empty"
                )
                self._finish_calibration_if_ready()

        measurements: dict[str, tuple[FootMeasurement, float, float, bool]] = {}
        if self.calibration is not None and self.collector is None:
            duplicate_expected = set(detector_result.duplicate_ids).intersection(
                self.config.marker_ids
            )
            if duplicate_expected:
                self.status = "Duplicate marker IDs - use six unique markers"
                self._clear_filters()
            else:
                measurements = self._measure_frame(frame_bgr, expected_current)
            self._draw_calibration_geometry(annotated)
            self._draw_measurements(annotated, measurements)
        self.last_measurements = measurements

        self._draw_header(annotated, len(expected_current))
        if unexpected:
            self._draw_line(
                annotated,
                f"Ignored marker IDs: {','.join(str(value) for value in unexpected)}",
                3,
                (0, 200, 255),
            )
        return annotated

    def _measure_frame(
        self, frame_bgr: np.ndarray, current_corners: Mapping[int, np.ndarray]
    ) -> dict[str, tuple[FootMeasurement, float, float, bool]]:
        assert self.calibration is not None
        self.last_mask = None
        self.last_score = None
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.calibration.frame_size:
            self.status = "Frame size changed - recalibrate"
            self._clear_filters()
            return {}

        motion = compare_marker_positions(
            self.calibration.reference_corners, current_corners
        )
        if motion.common_marker_count < self.config.minimum_live_markers:
            self.status = (
                f"Need {self.config.minimum_live_markers} visible markers; "
                f"found {motion.common_marker_count}"
            )
            self._clear_filters()
            return {}
        if motion.median_error_px > self.config.maximum_camera_motion_px:
            self.status = (
                f"Camera/board moved ({motion.median_error_px:.1f}px) - recalibrate"
            )
            self._clear_filters()
            return {}
        if motion.max_error_px > self.config.maximum_camera_motion_px * 1.5:
            self.status = (
                f"Marker corner moved ({motion.max_error_px:.1f}px) - recalibrate"
            )
            self._clear_filters()
            return {}

        try:
            mask, score = make_foreground_mask(
                frame_bgr,
                self.calibration,
                threshold=self.config.foreground_threshold,
            )
        except ValueError as error:
            self.status = str(error)
            self._clear_filters()
            return {}
        self.last_mask = mask
        self.last_score = score
        contours = extract_foot_contours(
            mask,
            self.calibration,
            minimum_area_mm2=self.config.minimum_foot_area_mm2,
            maximum_area_mm2=self.config.maximum_foot_area_mm2,
        )

        output: dict[str, tuple[FootMeasurement, float, float, bool]] = {}
        seen_labels: set[str] = set()
        problems: list[str] = []
        for side, contour in contours.items():
            label = self._label_for_side(side)
            seen_labels.add(label)
            if contour_touches_roi(contour, self.calibration.roi_polygon):
                problems.append(f"{label} foot touches measurement boundary")
                self.filters[label].clear()
                continue
            measurement = measure_foot_contour(
                contour,
                self.calibration.marker_homographies,
                minimum_length_mm=self.config.minimum_length_mm,
                maximum_length_mm=self.config.maximum_length_mm,
                minimum_width_mm=self.config.minimum_width_mm,
                maximum_width_mm=self.config.maximum_width_mm,
            )
            if measurement is None:
                problems.append(f"{label} foot shape is invalid")
                self.filters[label].clear()
                continue
            if measurement.marker_count < self.config.minimum_measurement_markers:
                problems.append(
                    f"{label} has only {measurement.marker_count} valid marker estimates"
                )
                self.filters[label].clear()
                continue
            if measurement.marker_spread_mm > self.config.maximum_marker_spread_mm:
                problems.append(
                    f"{label} marker disagreement {measurement.marker_spread_mm:.1f}mm"
                )
                self.filters[label].clear()
                continue
            filtered, temporal_spread, stable = self.filters[label].add(
                measurement.length_mm
            )
            output[label] = (measurement, filtered, temporal_spread, stable)

        for label in ("LEFT", "RIGHT"):
            if label not in seen_labels:
                self.filters[label].clear()

        if problems:
            self.status = "; ".join(problems)
        elif len(output) < 2:
            self.status = "Place both feet inside the marker boundary"
        elif all(item[3] for item in output.values()):
            self.status = "Stable measurement"
        else:
            self.status = "Hold both feet still"
        return output

    def _draw_calibration_geometry(self, image: np.ndarray) -> None:
        assert self.calibration is not None
        polygon = np.rint(self.calibration.roi_polygon).astype(np.int32)
        cv2.polylines(image, [polygon], True, (255, 180, 0), 2, cv2.LINE_AA)
        top = tuple(np.rint(self.calibration.divider_top).astype(int))
        bottom = tuple(np.rint(self.calibration.divider_bottom).astype(int))
        cv2.line(image, top, bottom, (255, 180, 0), 2, cv2.LINE_AA)

    def _draw_measurements(
        self,
        image: np.ndarray,
        measurements: Mapping[str, tuple[FootMeasurement, float, float, bool]],
    ) -> None:
        colours = {"LEFT": (60, 220, 60), "RIGHT": (255, 160, 40)}
        for label, (measurement, filtered, temporal_spread, stable) in measurements.items():
            colour = colours[label]
            endpoints = np.rint(measurement.image_endpoints).astype(int)
            cv2.line(
                image,
                tuple(endpoints[0]),
                tuple(endpoints[1]),
                colour,
                3,
                cv2.LINE_AA,
            )
            for endpoint in endpoints:
                cv2.circle(image, tuple(endpoint), 7, colour, -1, cv2.LINE_AA)
            centroid = tuple(np.rint(measurement.image_centroid).astype(int))
            stable_text = "STABLE" if stable else "HOLD"
            cv2.putText(
                image,
                f"{label}: {filtered:.1f} mm  {stable_text}",
                (centroid[0] - 100, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"marker MAD {measurement.marker_spread_mm:.1f} mm / time MAD {temporal_spread:.1f}",
                (centroid[0] - 100, centroid[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )

    def _draw_header(self, image: np.ndarray, visible_markers: int) -> None:
        height = 122
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
        cv2.putText(
            image,
            self.status,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            self._quality_header_text(visible_markers),
            (14, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"labels {'SWAPPED' if self.config.swap_feet else 'NORMAL'} | "
            "B calibrate | C capture | R reset | S swap | D mask | Q quit",
            (14, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    def _quality_header_text(self, visible_markers: int) -> str:
        quality = self.last_detection_quality
        if quality is None or quality.minimum_marker_side_px is None:
            return (
                f"markers {visible_markers}/{len(self.config.marker_ids)} | "
                f"detect {self.last_detection_preprocessing}"
            )
        fallback = (
            f" fallback {','.join(str(value) for value in self.last_fallback_ids)}"
            if self.last_fallback_ids
            else ""
        )
        return (
            f"markers {visible_markers}/{len(self.config.marker_ids)} | "
            f"min {quality.minimum_marker_side_px:.0f}px | "
            f"sharp min {quality.minimum_marker_sharpness:.1f} | "
            f"contrast {quality.minimum_marker_contrast:.0f} | "
            f"detect {self.last_detection_preprocessing}{fallback}"
        )

    @staticmethod
    def _draw_line(
        image: np.ndarray, text: str, line_index: int, colour: tuple[int, int, int]
    ) -> None:
        cv2.putText(
            image,
            text,
            (14, 30 + line_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )


def _parse_marker_ids(value: str) -> tuple[int, ...]:
    try:
        marker_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("marker IDs must be comma-separated integers") from error
    if len(marker_ids) != 6 or len(set(marker_ids)) != len(marker_ids):
        raise argparse.ArgumentTypeError("provide exactly six unique marker IDs")
    return marker_ids


def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _set_camera_property(
    capture: cv2.VideoCapture,
    property_id: int,
    value: float,
    label: str,
) -> None:
    if not capture.set(property_id, value):
        print(
            f"Camera warning: backend ignored {label}={value}",
            file=sys.stderr,
        )


def _write_capture(capture_dir: Path, frame_bgr: np.ndarray) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    milliseconds = (time.time_ns() // 1_000_000) % 1_000
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
    path = capture_dir / f"foot_capture_{timestamp}.png"
    ok, encoded = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise OSError("Could not encode captured PNG")
    encoded.tofile(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure left/right foot length with six 30 mm ArUco markers."
    )
    parser.add_argument(
        "--source", default="0", help="USB camera index or video path (default: 0)"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        help="Optional four-character camera codec such as MJPG",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Discard these initial camera frames while focus/exposure settles",
    )
    parser.add_argument(
        "--autofocus",
        choices=("keep", "on", "off"),
        default="keep",
        help="Request camera autofocus mode; backend support varies",
    )
    parser.add_argument(
        "--focus",
        type=float,
        help="Optional manual camera focus value; use with --autofocus off",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        help="Optional backend-specific manual exposure value",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip a mirrored live camera frame",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures"),
        help="C-key lossless PNG output directory",
    )
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--marker-ids", type=_parse_marker_ids, default=(0, 1, 2, 3, 4, 5)
    )
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=24)
    parser.add_argument("--min-live-markers", type=int, default=2)
    parser.add_argument("--max-camera-motion-px", type=float, default=5.0)
    parser.add_argument("--max-marker-spread-mm", type=float, default=10.0)
    parser.add_argument("--min-marker-side-px", type=float, default=60.0)
    parser.add_argument("--min-marker-sharpness", type=float, default=12.0)
    parser.add_argument("--min-marker-contrast", type=float, default=50.0)
    parser.add_argument("--swap-feet", action="store_true")
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help=(
            "Optional pinhole/fisheye NPZ from calibrate_wide_camera.py. "
            "Correction runs on each full frame before mirroring and ArUco."
        ),
    )
    parser.add_argument(
        "--undistort-balance",
        type=float,
        default=1.0,
        help="Lens correction crop/FOV balance from 0 to 1 (default: 1).",
    )
    parser.add_argument(
        "--allow-calibration-resize",
        action="store_true",
        help=(
            "Allow a known full-frame same-aspect resize of calibration "
            "intrinsics; never use for crop or another sensor mode."
        ),
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.marker_size_mm) or args.marker_size_mm <= 0:
        raise SystemExit("--marker-size-mm must be positive")
    if args.min_live_markers < 1:
        raise SystemExit("--min-live-markers must be positive")
    if args.min_live_markers > len(args.marker_ids):
        raise SystemExit("--min-live-markers cannot exceed the six marker IDs")
    if args.calibration_frames < 5:
        raise SystemExit("--calibration-frames must be at least 5")
    if not 1 <= args.threshold <= 254:
        raise SystemExit("--threshold must be between 1 and 254")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("--width, --height and --fps must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames cannot be negative")
    if args.fourcc is not None and len(args.fourcc) != 4:
        raise SystemExit("--fourcc must contain exactly four characters")
    if args.focus is not None and not math.isfinite(args.focus):
        raise SystemExit("--focus must be finite")
    if args.exposure is not None and not math.isfinite(args.exposure):
        raise SystemExit("--exposure must be finite")
    if (
        not math.isfinite(args.undistort_balance)
        or not 0.0 <= args.undistort_balance <= 1.0
    ):
        raise SystemExit("--undistort-balance must be between 0 and 1")
    if (
        not math.isfinite(args.max_camera_motion_px)
        or not math.isfinite(args.max_marker_spread_mm)
        or args.max_camera_motion_px <= 0
        or args.max_marker_spread_mm <= 0
    ):
        raise SystemExit("motion/spread limits must be positive")
    quality_limits = (
        args.min_marker_side_px,
        args.min_marker_sharpness,
        args.min_marker_contrast,
    )
    if any(not math.isfinite(value) or value <= 0 for value in quality_limits):
        raise SystemExit("marker quality limits must be positive")

    camera_calibration = (
        None
        if args.camera_calibration is None
        else load_lens_calibration(args.camera_calibration)
    )
    detector = ArucoMarkerDetector(
        args.dictionary, expected_marker_ids=args.marker_ids
    )
    invalid_ids = [
        marker_id
        for marker_id in args.marker_ids
        if marker_id < 0 or marker_id >= detector.marker_count
    ]
    if invalid_ids:
        raise SystemExit(
            f"Marker IDs outside {args.dictionary} range: "
            + ",".join(str(value) for value in invalid_ids)
        )
    config = AppConfig(
        marker_ids=args.marker_ids,
        marker_size_mm=args.marker_size_mm,
        calibration_frames=args.calibration_frames,
        foreground_threshold=args.threshold,
        minimum_live_markers=args.min_live_markers,
        maximum_camera_motion_px=args.max_camera_motion_px,
        maximum_marker_spread_mm=args.max_marker_spread_mm,
        minimum_marker_side_px=args.min_marker_side_px,
        minimum_marker_sharpness=args.min_marker_sharpness,
        minimum_marker_contrast=args.min_marker_contrast,
        swap_feet=args.swap_feet,
    )
    app = FootMeasurementApp(config)

    source = _parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        if args.fourcc is not None:
            _set_camera_property(
                capture,
                cv2.CAP_PROP_FOURCC,
                float(cv2.VideoWriter_fourcc(*args.fourcc.upper())),
                "fourcc",
            )
        _set_camera_property(
            capture, cv2.CAP_PROP_FRAME_WIDTH, float(args.width), "width"
        )
        _set_camera_property(
            capture, cv2.CAP_PROP_FRAME_HEIGHT, float(args.height), "height"
        )
        _set_camera_property(capture, cv2.CAP_PROP_FPS, float(args.fps), "fps")
        if args.autofocus != "keep" and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            _set_camera_property(
                capture,
                cv2.CAP_PROP_AUTOFOCUS,
                1.0 if args.autofocus == "on" else 0.0,
                "autofocus",
            )
        if args.focus is not None and hasattr(cv2, "CAP_PROP_FOCUS"):
            _set_camera_property(
                capture, cv2.CAP_PROP_FOCUS, float(args.focus), "focus"
            )
        if args.exposure is not None and hasattr(cv2, "CAP_PROP_EXPOSURE"):
            _set_camera_property(
                capture,
                cv2.CAP_PROP_EXPOSURE,
                float(args.exposure),
                "exposure",
            )
    if not capture.isOpened():
        print(f"Could not open camera/video source: {args.source}", file=sys.stderr)
        return 2

    if isinstance(source, int):
        for _ in range(args.warmup_frames):
            ok, warmup_frame = capture.read()
            if not ok or warmup_frame is None:
                capture.release()
                print("Camera warm-up frame read failed", file=sys.stderr)
                return 3
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        print(
            f"Camera active: {actual_width}x{actual_height} @ "
            f"{actual_fps:.1f} FPS (requested {args.width}x{args.height} @ "
            f"{args.fps})"
        )
        if actual_width != args.width or actual_height != args.height:
            print(
                "Camera warning: requested resolution was not applied",
                file=sys.stderr,
            )

    print(
        "Controls: B=calibrate empty board, C=capture PNG, R=reset, "
        "S=swap feet, D=mask, Q=quit"
    )
    previous_time = time.perf_counter()
    undistort_maps: UndistortionMaps | None = None
    window_name = "ArUco two-foot measurement"
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if isinstance(source, str):
                    break
                print("Camera frame read failed", file=sys.stderr)
                return 3
            # Preserve the untouched sensor-coordinate frame for C-key saves.
            # The live processing frame may be undistorted and mirrored below;
            # saving that version would make a later photo workflow apply the
            # same lens calibration twice.
            raw_frame = frame.copy()
            if camera_calibration is not None:
                if undistort_maps is None:
                    undistort_maps = build_undistortion_maps(
                        camera_calibration,
                        (frame.shape[1], frame.shape[0]),
                        balance=args.undistort_balance,
                        allow_same_aspect_resize=args.allow_calibration_resize,
                    )
                    print(
                        "Lens correction active: "
                        f"{camera_calibration.model}, valid pixels "
                        f"{undistort_maps.valid_fraction * 100.0:.1f}%"
                    )
                frame = apply_undistortion(frame, undistort_maps)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            detector_result = detector.detect(
                frame, robust=app.calibration is None
            )
            annotated = app.process_frame(frame, detector_result)
            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(
                annotated,
                f"{fps:.1f} FPS",
                (annotated.shape[1] - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)
            if app.debug_mask and app.last_mask is not None:
                cv2.imshow("Foreground mask", app.last_mask)
            else:
                try:
                    cv2.destroyWindow("Foreground mask")
                except cv2.error:
                    pass

            key = cv2.waitKey(1) & 0xFF
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("b"), ord("B")):
                app.start_calibration()
            elif key in (ord("c"), ord("C")):
                expected_detected = set(detector_result.corners_by_id).intersection(
                    args.marker_ids
                )
                missing = sorted(set(args.marker_ids).difference(expected_detected))
                duplicate_expected = sorted(
                    set(detector_result.duplicate_ids).intersection(args.marker_ids)
                )
                quality_problem = app.capture_quality_problem(detector_result)
                if duplicate_expected:
                    app.status = "Capture blocked - duplicate marker IDs"
                elif missing:
                    app.status = (
                        "Capture blocked - missing markers "
                        + ",".join(str(value) for value in missing)
                    )
                elif quality_problem:
                    app.status = "Capture blocked - " + quality_problem
                else:
                    try:
                        capture_path = _write_capture(
                            args.capture_dir, raw_frame
                        )
                    except OSError as error:
                        app.status = f"Capture failed: {error}"
                    else:
                        app.status = f"Captured raw frame {capture_path}"
                        print(f"Captured raw sensor photo: {capture_path}")
            elif key in (ord("r"), ord("R")):
                app.reset()
            elif key in (ord("s"), ord("S")):
                app.swap_labels()
            elif key in (ord("d"), ord("D")):
                app.debug_mask = not app.debug_mask
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
