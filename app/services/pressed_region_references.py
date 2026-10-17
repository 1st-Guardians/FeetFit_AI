from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_REFERENCE_EXAMPLES = (
    (
        "CAVUS",
        "2-IMG-20221109-WA0031_jpg.rf.576edecfcfb26b0aa14ba06ab0251e26",
    ),
    (
        "NORMAL",
        "9-WhatsApp-Image-2022-10-11-at-3-57-23-PM_jpeg_jpg.rf."
        "b488ae841bf036acb08d7d11eb04c40c",
    ),
    (
        "FLAT",
        "5-IMG-20221022-WA0043_jpg.rf.1ddcf9856ef91fd6adf37c045e17b03f",
    ),
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_REFERENCE_PANEL_SIZE = 224
_PRESS_BGR = (124, 0, 255)


def _read_color(path: Path) -> np.ndarray | None:
    """Read Unicode/decomposed Windows paths without relying on cv2.imread."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _is_exact_press_pair(source: np.ndarray, mask: np.ndarray) -> bool:
    if source.shape[:2] != mask.shape[:2]:
        return False
    colors = {
        tuple(int(channel) for channel in color)
        for color in np.unique(mask.reshape(-1, 3), axis=0)
    }
    # SegmentationClass is decoded by OpenCV as BGR. RGB press=(255, 0, 124).
    allowed = {(0, 0, 0), _PRESS_BGR}
    return colors.issubset(allowed) and allowed.issubset(colors)


def _press_binary(mask: np.ndarray) -> np.ndarray:
    return np.all(mask == np.asarray(_PRESS_BGR, dtype=np.uint8), axis=2).astype(
        np.uint8
    )


def _toe_up_rotation_degrees(mask: np.ndarray) -> float:
    """Estimate a deterministic rotation that puts toes above the plantar axis.

    The press annotations contain one dominant plantar component and one or more
    separate forefoot/toe components. PCA supplies the unsigned long axis; the
    centroid of every component except the largest resolves which end contains
    the toes. This works for the deliberately mixed orientations in the curated
    examples without inferring orientation from source-image colour.
    """
    binary = _press_binary(mask)
    y_coords, x_coords = np.nonzero(binary)
    if x_coords.size < 2:
        return 0.0

    points = np.column_stack((x_coords, y_coords)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count > 2:
        component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest_index = int(np.argmax(component_areas))
        auxiliary_indices = [
            index for index in range(component_areas.size) if index != largest_index
        ]
        auxiliary_areas = component_areas[auxiliary_indices]
        auxiliary_centers = centroids[1:][auxiliary_indices]
        toe_center = np.average(
            auxiliary_centers,
            axis=0,
            weights=auxiliary_areas,
        )
        toe_hint = toe_center - center
        if float(np.linalg.norm(toe_hint)) > 1.0 and np.dot(long_axis, toe_hint) < 0:
            long_axis = -long_axis
    elif long_axis[1] > 0:
        # This fallback is not used by the curated masks, but keeps a single
        # connected synthetic/legacy mask in a stable vertical orientation.
        long_axis = -long_axis

    # OpenCV's positive image rotation subtracts this angle from a vector's
    # image-coordinate angle. A toe-up vector has an angle of -90 degrees.
    axis_angle = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
    rotation = axis_angle + 90.0
    return (rotation + 180.0) % 360.0 - 180.0


def _canonicalize_reference_pair(
    source: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a registered 224px SOURCE/MASK pair with toes facing upward."""
    panel_size = (_REFERENCE_PANEL_SIZE, _REFERENCE_PANEL_SIZE)
    source_panel = cv2.resize(source, panel_size, interpolation=cv2.INTER_AREA)
    mask_panel = cv2.resize(mask, panel_size, interpolation=cv2.INTER_NEAREST)

    rotation = _toe_up_rotation_degrees(mask_panel)
    center = ((_REFERENCE_PANEL_SIZE - 1) / 2.0,) * 2
    transform = cv2.getRotationMatrix2D(center, rotation, 1.0)
    canonical_source = cv2.warpAffine(
        source_panel,
        transform,
        panel_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    canonical_mask = cv2.warpAffine(
        mask_panel,
        transform,
        panel_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return canonical_source, canonical_mask


def _select_reference_pairs(
    dataset_dir: Path,
    annotation_dir: Path,
    max_pairs: int,
) -> list[tuple[str, Path, Path]]:
    try:
        source_files = sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        mask_files = sorted(
            path
            for path in annotation_dir.rglob("*.png")
            if path.parent.name == "SegmentationClass"
        )
    except OSError:
        return []

    sources_by_stem: dict[str, Path] = {}
    for path in source_files:
        sources_by_stem.setdefault(path.stem, path)
    masks_by_stem: dict[str, Path] = {}
    for mask_path in mask_files:
        masks_by_stem.setdefault(mask_path.stem, mask_path)

    selected: list[tuple[str, Path, Path]] = []
    for label, stem in _REFERENCE_EXAMPLES[:max_pairs]:
        source_path = sources_by_stem.get(stem)
        mask_path = masks_by_stem.get(stem)
        if source_path is None or mask_path is None:
            continue
        source = _read_color(source_path)
        mask = _read_color(mask_path)
        if source is None or mask is None or not _is_exact_press_pair(source, mask):
            continue
        selected.append((label, source_path, mask_path))
    return selected


def _make_reference_montage(
    pairs: list[tuple[str, Path, Path]],
) -> bytes | None:
    rows: list[np.ndarray] = []
    for _label, source_path, mask_path in pairs:
        source = _read_color(source_path)
        raw_mask = _read_color(mask_path)
        if (
            source is None
            or raw_mask is None
            or not _is_exact_press_pair(source, raw_mask)
        ):
            continue

        source, raw_mask = _canonicalize_reference_pair(source, raw_mask)
        rows.append(np.hstack((source, raw_mask)))

    if not rows:
        return None
    ok, encoded = cv2.imencode(".png", np.vstack(rows))
    return encoded.tobytes() if ok else None


@lru_cache(maxsize=8)
def _cached_reference_montage(
    dataset_dir: str,
    annotation_dir: str,
    max_pairs: int,
) -> bytes | None:
    pairs = _select_reference_pairs(
        Path(dataset_dir),
        Path(annotation_dir),
        max_pairs,
    )
    montage = _make_reference_montage(pairs)
    if montage is None:
        logger.warning(
            "No paired plantar press-mask reference examples were found under %s and %s",
            dataset_dir,
            annotation_dir,
        )
    return montage


def build_reference_montage(
    dataset_dir: Path | str | None,
    annotation_dir: Path | str | None,
    max_pairs: int,
) -> bytes | None:
    """Return one in-memory few-shot montage; never write a derived file to disk."""
    if dataset_dir is None or annotation_dir is None or max_pairs <= 0:
        return None
    return _cached_reference_montage(
        str(dataset_dir),
        str(annotation_dir),
        int(max_pairs),
    )
