from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PINHOLE_DISTORTION_LENGTHS = frozenset((4, 5, 8, 12, 14))
FISHEYE_DISTORTION_LENGTH = 4
SUPPORTED_LENS_MODELS = frozenset(("pinhole", "fisheye"))


class LensCalibrationError(ValueError):
    """Raised when lens calibration data cannot be applied safely."""


def _normalise_model(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise LensCalibrationError("lens_model must contain one string")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        try:
            scalar = scalar.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LensCalibrationError("lens_model is not valid UTF-8") from error
    model = str(scalar).strip().casefold()
    aliases = {
        "standard": "pinhole",
        "brown": "pinhole",
        "brown-conrady": "pinhole",
        "opencv": "pinhole",
        "fish-eye": "fisheye",
    }
    model = aliases.get(model, model)
    if model not in SUPPORTED_LENS_MODELS:
        supported = ", ".join(sorted(SUPPORTED_LENS_MODELS))
        raise LensCalibrationError(
            f"Unsupported lens model {model!r}; expected one of: {supported}"
        )
    return model


def _normalise_image_size(value: Any, *, field_name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.size != 2 or not np.issubdtype(array.dtype, np.number):
        raise LensCalibrationError(
            f"{field_name} must contain (width, height)"
        )
    numbers = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(numbers).all():
        raise LensCalibrationError(f"{field_name} values must be finite")
    rounded = np.rint(numbers)
    if not np.allclose(numbers, rounded, rtol=0.0, atol=1e-9):
        raise LensCalibrationError(f"{field_name} values must be integers")
    width, height = (int(value) for value in rounded)
    if width <= 0 or height <= 0:
        raise LensCalibrationError(f"{field_name} values must be positive")
    return width, height


def _normalise_camera_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise LensCalibrationError("camera_matrix must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise LensCalibrationError("camera_matrix values must be finite")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise LensCalibrationError("camera focal lengths must be positive")
    if not np.allclose(
        matrix[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-8
    ):
        raise LensCalibrationError(
            "camera_matrix last row must be [0, 0, 1]"
        )
    result = matrix.copy()
    result.setflags(write=False)
    return result


def _normalise_distortion(value: Any, model: str) -> np.ndarray:
    coefficients = np.asarray(value, dtype=np.float64).reshape(-1)
    if not coefficients.size or not np.isfinite(coefficients).all():
        raise LensCalibrationError("dist_coeffs must be non-empty and finite")
    if model == "fisheye":
        if coefficients.size != FISHEYE_DISTORTION_LENGTH:
            raise LensCalibrationError(
                "fisheye dist_coeffs must contain exactly four values"
            )
    elif coefficients.size not in PINHOLE_DISTORTION_LENGTHS:
        allowed = ", ".join(str(value) for value in sorted(PINHOLE_DISTORTION_LENGTHS))
        raise LensCalibrationError(
            f"pinhole dist_coeffs must contain {allowed} values"
        )
    result = coefficients.copy()
    result.setflags(write=False)
    return result


def _normalise_rms(value: Any | None) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise LensCalibrationError("rms must contain one value")
    rms = float(array.reshape(-1)[0])
    if not math.isfinite(rms) or rms < 0.0:
        raise LensCalibrationError("rms must be finite and non-negative")
    return rms


@dataclass(frozen=True)
class LensCalibration:
    """Intrinsic calibration in the distorted full-frame sensor coordinates."""

    model: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        model = _normalise_model(self.model)
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self, "camera_matrix", _normalise_camera_matrix(self.camera_matrix)
        )
        object.__setattr__(
            self,
            "dist_coeffs",
            _normalise_distortion(self.dist_coeffs, model),
        )
        object.__setattr__(
            self,
            "image_size",
            _normalise_image_size(self.image_size, field_name="image_size"),
        )
        object.__setattr__(self, "rms", _normalise_rms(self.rms))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))


def _npz_scalar(data: Any, keys: tuple[str, ...]) -> Any | None:
    present = [key for key in keys if key in data]
    if not present:
        return None
    first = np.asarray(data[present[0]])
    for key in present[1:]:
        other = np.asarray(data[key])
        if first.shape != other.shape or not np.array_equal(first, other):
            joined = ", ".join(present)
            raise LensCalibrationError(
                f"Conflicting calibration fields: {joined}"
            )
    return data[present[0]]


def load_lens_calibration(path: str | Path) -> LensCalibration:
    """Load and validate a pinhole/fisheye calibration NPZ.

    Required fields are ``lens_model`` (or ``model``), ``camera_matrix``,
    ``dist_coeffs`` and ``image_size`` in ``(width, height)`` order. Optional
    RMS aliases are ``rms``, ``calibration_rms`` and ``calibration_rms_px``.
    """

    calibration_path = Path(path)
    try:
        with np.load(calibration_path, allow_pickle=False) as data:
            model_value = _npz_scalar(data, ("lens_model", "model"))
            if model_value is None:
                raise LensCalibrationError(
                    "Calibration NPZ is missing lens_model"
                )
            required = ("camera_matrix", "dist_coeffs", "image_size")
            missing = [key for key in required if key not in data]
            if missing:
                raise LensCalibrationError(
                    "Calibration NPZ is missing: " + ", ".join(missing)
                )
            rms = _npz_scalar(
                data, ("rms", "calibration_rms", "calibration_rms_px")
            )
            return LensCalibration(
                model=_normalise_model(model_value),
                camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
                image_size=_normalise_image_size(
                    data["image_size"], field_name="image_size"
                ),
                rms=_normalise_rms(rms),
                source_path=str(calibration_path),
            )
    except LensCalibrationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise LensCalibrationError(
            f"Could not load lens calibration: {calibration_path}"
        ) from error


def _validate_balance(balance: float) -> float:
    try:
        value = float(balance)
    except (TypeError, ValueError) as error:
        raise LensCalibrationError("balance must be a number in [0, 1]") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise LensCalibrationError("balance must be between 0 and 1")
    return value


def _scaled_camera_matrix(
    calibration: LensCalibration,
    input_size: tuple[int, int],
    *,
    allow_same_aspect_resize: bool,
) -> tuple[np.ndarray, float, float, bool]:
    calibrated_width, calibrated_height = calibration.image_size
    input_width, input_height = input_size
    if input_size != calibration.image_size and not allow_same_aspect_resize:
        raise LensCalibrationError(
            "Input size differs from calibration; use the exact calibrated "
            "camera mode or explicitly allow a known full-frame resize"
        )
    # A different aspect ratio implies a crop, padding, or non-uniform resize.
    # None can be recovered from the final dimensions without extra metadata.
    if input_width * calibrated_height != input_height * calibrated_width:
        raise LensCalibrationError(
            "Input aspect ratio differs from calibration; apply lens correction "
            "to the uncropped full frame before cropping"
        )
    scale_x = input_width / calibrated_width
    scale_y = input_height / calibrated_height
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64).copy()
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    return matrix, float(scale_x), float(scale_y), input_size != calibration.image_size


@dataclass(frozen=True)
class UndistortionMaps:
    """Dense output-to-input remap for one calibrated image resolution."""

    model: str
    input_size: tuple[int, int]
    calibrated_image_size: tuple[int, int]
    camera_matrix: np.ndarray
    new_camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    valid_mask: np.ndarray
    balance: float
    scale_x: float
    scale_y: float
    intrinsics_scaled: bool
    same_aspect_resize_allowed: bool
    calibration_rms: float | None
    calibration_path: str | None
    pinhole_valid_roi: tuple[int, int, int, int] | None = None

    @property
    def valid_fraction(self) -> float:
        return float(np.count_nonzero(self.valid_mask) / self.valid_mask.size)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return reproducible, directly JSON-serializable preprocessing data."""

        result: dict[str, Any] = {
            "applied": True,
            "lens_model": self.model,
            "calibration_path": self.calibration_path,
            "calibrated_image_size": list(self.calibrated_image_size),
            "input_image_size": list(self.input_size),
            "same_aspect_resize_assumed": bool(self.intrinsics_scaled),
            "same_aspect_resize_allowed": bool(
                self.same_aspect_resize_allowed
            ),
            "intrinsics_scaled": bool(self.intrinsics_scaled),
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "balance": float(self.balance),
            "calibration_rms_px": (
                None
                if self.calibration_rms is None
                else float(self.calibration_rms)
            ),
            "camera_matrix_scaled": self.camera_matrix.astype(float).tolist(),
            "new_camera_matrix": self.new_camera_matrix.astype(float).tolist(),
            "distortion_coefficient_count": int(self.dist_coeffs.size),
            "valid_pixel_fraction": self.valid_fraction,
            "pinhole_valid_roi": (
                None
                if self.pinhole_valid_roi is None
                else list(self.pinhole_valid_roi)
            ),
        }
        # Catch accidental NumPy scalars/arrays if this property is extended.
        json.dumps(result)
        return result


def build_undistortion_maps(
    calibration: LensCalibration,
    input_size: tuple[int, int],
    balance: float = 1.0,
    *,
    allow_same_aspect_resize: bool = False,
) -> UndistortionMaps:
    """Create a nonlinear full-frame lens remap for ``(width, height)``.

    Exact calibration size is required by default. A caller may explicitly
    allow a known full-frame same-aspect resize; crops and differently padded
    images remain rejected because their principal-point shift is unknowable
    from image dimensions alone.
    """

    if not isinstance(calibration, LensCalibration):
        raise LensCalibrationError("calibration must be a LensCalibration")
    size = _normalise_image_size(input_size, field_name="input_size")
    balance_value = _validate_balance(balance)
    camera_matrix, scale_x, scale_y, scaled = _scaled_camera_matrix(
        calibration,
        size,
        allow_same_aspect_resize=bool(allow_same_aspect_resize),
    )
    dist_coeffs = np.asarray(calibration.dist_coeffs, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    valid_roi: tuple[int, int, int, int] | None = None

    try:
        if calibration.model == "fisheye":
            fisheye_coeffs = dist_coeffs.reshape(4, 1)
            new_camera_matrix = (
                cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    camera_matrix,
                    fisheye_coeffs,
                    size,
                    identity,
                    balance=balance_value,
                    new_size=size,
                    fov_scale=1.0,
                )
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix,
                fisheye_coeffs,
                identity,
                new_camera_matrix,
                size,
                cv2.CV_32FC1,
            )
        else:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                size,
                balance_value,
                size,
                centerPrincipalPoint=False,
            )
            valid_roi = tuple(int(value) for value in roi)
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                identity,
                new_camera_matrix,
                size,
                cv2.CV_32FC1,
            )
    except cv2.error as error:
        raise LensCalibrationError(
            f"OpenCV could not build {calibration.model} undistortion maps"
        ) from error

    new_camera_matrix = np.asarray(new_camera_matrix, dtype=np.float64)
    map1 = np.asarray(map1, dtype=np.float32)
    map2 = np.asarray(map2, dtype=np.float32)
    if (
        new_camera_matrix.shape != (3, 3)
        or map1.shape != (size[1], size[0])
        or map2.shape != map1.shape
        or not np.isfinite(new_camera_matrix).all()
    ):
        raise LensCalibrationError("OpenCV returned invalid undistortion maps")
    valid = (
        np.isfinite(map1)
        & np.isfinite(map2)
        & (map1 >= 0.0)
        & (map1 <= size[0] - 1.0)
        & (map2 >= 0.0)
        & (map2 <= size[1] - 1.0)
    )
    valid_mask = valid.astype(np.uint8) * 255
    if not np.any(valid):
        raise LensCalibrationError("Undistortion map contains no valid pixels")

    return UndistortionMaps(
        model=calibration.model,
        input_size=size,
        calibrated_image_size=calibration.image_size,
        camera_matrix=camera_matrix,
        new_camera_matrix=new_camera_matrix,
        dist_coeffs=dist_coeffs.copy(),
        map1=map1,
        map2=map2,
        valid_mask=valid_mask,
        balance=balance_value,
        scale_x=scale_x,
        scale_y=scale_y,
        intrinsics_scaled=scaled,
        same_aspect_resize_allowed=bool(allow_same_aspect_resize),
        calibration_rms=calibration.rms,
        calibration_path=calibration.source_path,
        pinhole_valid_roi=valid_roi,
    )


def apply_undistortion(
    image: np.ndarray,
    maps: UndistortionMaps,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply precomputed maps without cropping or changing image dimensions."""

    if not isinstance(maps, UndistortionMaps):
        raise LensCalibrationError("maps must be UndistortionMaps")
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise LensCalibrationError("image must be a non-empty 2D or 3D array")
    actual_size = (int(array.shape[1]), int(array.shape[0]))
    if actual_size != maps.input_size:
        raise LensCalibrationError(
            f"Image size {actual_size} does not match maps {maps.input_size}"
        )
    try:
        return cv2.remap(
            array,
            maps.map1,
            maps.map2,
            interpolation=int(interpolation),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except (cv2.error, TypeError, ValueError) as error:
        raise LensCalibrationError("Could not apply lens undistortion") from error


def undistort_points(points: np.ndarray, maps: UndistortionMaps) -> np.ndarray:
    """Map distorted input pixels into the remap's undistorted output pixels."""

    if not isinstance(maps, UndistortionMaps):
        raise LensCalibrationError("maps must be UndistortionMaps")
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0 or array.shape[-1:] != (2,) or not np.isfinite(array).all():
        raise LensCalibrationError("points must be a finite (..., 2) array")
    original_shape = array.shape
    source = array.reshape(-1, 1, 2)
    try:
        if maps.model == "fisheye":
            output = cv2.fisheye.undistortPoints(
                source,
                maps.camera_matrix,
                maps.dist_coeffs.reshape(4, 1),
                R=np.eye(3, dtype=np.float64),
                P=maps.new_camera_matrix,
            )
        else:
            output = cv2.undistortPoints(
                source,
                maps.camera_matrix,
                maps.dist_coeffs,
                R=np.eye(3, dtype=np.float64),
                P=maps.new_camera_matrix,
            )
    except cv2.error as error:
        raise LensCalibrationError("Could not undistort points") from error
    result = np.asarray(output, dtype=np.float64).reshape(original_shape)
    if not np.isfinite(result).all():
        raise LensCalibrationError("Point undistortion produced non-finite values")
    return result
