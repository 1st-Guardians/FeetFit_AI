from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .lens import LensCalibration, LensCalibrationError


@dataclass(frozen=True)
class HybridLensModel:
    """Physical lens model followed by a smooth learned point correction.

    The correction is stored on a coarse regular grid in the physical model's
    undistorted pixel coordinates.  Bilinear interpolation keeps the mapping
    deterministic and avoids inventing image content.  Measurement code should
    transform detected source-image points with :func:`correct_points` rather
    than detecting features again on a resampled image.
    """

    calibration: LensCalibration
    grid_x: np.ndarray
    grid_y: np.ndarray
    correction: np.ndarray
    training_view_count: int
    validation_rmse_px: float
    base_validation_rmse_px: float
    regularization: float
    source_path: str | None = None

    def __post_init__(self) -> None:
        gx = np.asarray(self.grid_x, dtype=np.float64).reshape(-1)
        gy = np.asarray(self.grid_y, dtype=np.float64).reshape(-1)
        field = np.asarray(self.correction, dtype=np.float64)
        if gx.size < 2 or gy.size < 2 or np.any(np.diff(gx) <= 0) or np.any(np.diff(gy) <= 0):
            raise LensCalibrationError("hybrid correction grid must be increasing")
        if field.shape != (gy.size, gx.size, 2) or not np.isfinite(field).all():
            raise LensCalibrationError("hybrid correction has an invalid shape")
        object.__setattr__(self, "grid_x", gx)
        object.__setattr__(self, "grid_y", gy)
        object.__setattr__(self, "correction", field)


def load_hybrid_lens_model(path: str | Path) -> HybridLensModel:
    model_path = Path(path)
    try:
        with np.load(model_path, allow_pickle=False) as data:
            calibration = LensCalibration(
                model=str(np.asarray(data["lens_model"]).reshape(-1)[0]),
                camera_matrix=data["camera_matrix"],
                dist_coeffs=data["dist_coeffs"],
                image_size=tuple(int(v) for v in data["image_size"].reshape(-1)),
                rms=float(np.asarray(data["calibration_rms"]).reshape(-1)[0]),
                source_path=str(model_path),
            )
            return HybridLensModel(
                calibration=calibration,
                grid_x=data["residual_grid_x"],
                grid_y=data["residual_grid_y"],
                correction=data["residual_correction"],
                training_view_count=int(data["training_view_count"]),
                validation_rmse_px=float(data["validation_rmse_px"]),
                base_validation_rmse_px=float(data["base_validation_rmse_px"]),
                regularization=float(data["residual_regularization"]),
                source_path=str(model_path),
            )
    except LensCalibrationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise LensCalibrationError(f"Could not load hybrid lens model: {model_path}") from error


def _base_undistort_points(points: np.ndarray, calibration: LensCalibration) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    matrix = calibration.camera_matrix
    try:
        if calibration.model == "fisheye":
            result = cv2.fisheye.undistortPoints(
                source, matrix, calibration.dist_coeffs.reshape(4, 1), P=matrix
            )
        else:
            result = cv2.undistortPoints(
                source, matrix, calibration.dist_coeffs, P=matrix
            )
    except cv2.error as error:
        raise LensCalibrationError("Could not apply physical point correction") from error
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


def interpolate_residual(points: np.ndarray, model: HybridLensModel) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    gx, gy = model.grid_x, model.grid_y
    x = np.clip(points[:, 0], gx[0], gx[-1])
    y = np.clip(points[:, 1], gy[0], gy[-1])
    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    tx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    ty = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
    field = model.correction
    return (
        field[iy, ix] * ((1.0 - tx) * (1.0 - ty))[:, None]
        + field[iy, ix + 1] * (tx * (1.0 - ty))[:, None]
        + field[iy + 1, ix] * ((1.0 - tx) * ty)[:, None]
        + field[iy + 1, ix + 1] * (tx * ty)[:, None]
    )


def correct_points(points: np.ndarray, model: HybridLensModel) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0 or array.shape[-1] != 2 or not np.isfinite(array).all():
        raise LensCalibrationError("points must be a finite (..., 2) array")
    shape = array.shape
    base = _base_undistort_points(array, model.calibration)
    corrected = base + interpolate_residual(base, model)
    return corrected.reshape(shape)
