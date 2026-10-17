from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.figure import Figure
from matplotlib.image import imread, imsave
from matplotlib.path import Path as MatplotlibPath
from scipy.interpolate import Rbf, splprep, splev
from scipy.ndimage import distance_transform_edt, gaussian_filter


SENSOR_COUNT = 12
# Rendering below follows the user's red-detail reference. Right input remains
# in raw mux channel order; its channel-to-pad coordinates already handle wiring.
IMG_W, IMG_H = 448, 1106
SENSOR_CENTERS_PX = np.array(
    [
        [302, 124], [164, 124], [227, 249], [358, 322],
        [95, 323], [227, 425], [119, 508], [226, 601],
        [141, 698], [165, 835], [289, 886], [227, 1001],
    ],
    dtype=float,
)
RIGHT_PAD_CENTERS_PX = SENSOR_CENTERS_PX.copy()
RIGHT_PAD_CENTERS_PX[:, 0] = IMG_W - RIGHT_PAD_CENTERS_PX[:, 0]
SENSOR_RADII_PX = np.array(
    [48, 48, 49, 48, 48, 49, 48, 49, 48, 49, 48, 52], dtype=float,
)
# Raw right C0 is heel pad 11; raw right C11 is toe pad 0.
RIGHT_CHANNEL_TO_PAD = np.arange(11, -1, -1)
RIGHT_SENSOR_CENTERS_PX = RIGHT_PAD_CENTERS_PX[RIGHT_CHANNEL_TO_PAD]
RIGHT_SENSOR_RADII_PX = SENSOR_RADII_PX[RIGHT_CHANNEL_TO_PAD]
ARCH_ZERO_NODE_FRACTIONS = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=float)[:, None]
LEFT_ARCH_ENDPOINTS_PX = SENSOR_CENTERS_PX[[3, 10]]
LEFT_ARCH_ZERO_NODES_PX = LEFT_ARCH_ENDPOINTS_PX[0] + ARCH_ZERO_NODE_FRACTIONS * (
    LEFT_ARCH_ENDPOINTS_PX[1] - LEFT_ARCH_ENDPOINTS_PX[0]
)
RIGHT_ARCH_ZERO_NODES_PX = LEFT_ARCH_ZERO_NODES_PX.copy()
RIGHT_ARCH_ZERO_NODES_PX[:, 0] = IMG_W - RIGHT_ARCH_ZERO_NODES_PX[:, 0]
LARGE_FOOT_BOUNDARY_PX = np.array(
    [
        [125, 135], [145, 95], [175, 65], [210, 50], [240, 47],
        [270, 53], [300, 68], [325, 93], [345, 125], [360, 165],
        [374, 220], [384, 275], [388, 320], [389, 430], [380, 520],
        [358, 600], [336, 680], [322, 775], [315, 875], [318, 985],
        [306, 1035], [283, 1074], [248, 1092], [205, 1090], [166, 1075],
        [137, 1046], [119, 1005], [111, 930], [106, 850], [99, 765],
        [91, 680], [79, 590], [66, 510], [58, 425], [59, 340],
        [67, 265], [80, 205], [96, 160],
    ],
    dtype=float,
)
SMALL_FOOT_BOUNDARY_PX = np.array(
    [
        [88, 235], [105, 195], [128, 165], [158, 145], [195, 132],
        [230, 130], [265, 138], [298, 155], [325, 180], [347, 215],
        [363, 260], [375, 315], [388, 421], [388, 520], [358, 600],
        [336, 680], [322, 775], [315, 875], [318, 985], [306, 1035],
        [283, 1074], [248, 1092], [205, 1090], [166, 1075], [137, 1046],
        [119, 1005], [111, 930], [106, 850], [99, 765], [91, 680],
        [79, 590], [66, 510], [58, 425], [59, 340], [67, 270],
    ],
    dtype=float,
)
RIGHT_LARGE_FOOT_BOUNDARY_PX = LARGE_FOOT_BOUNDARY_PX.copy()
RIGHT_LARGE_FOOT_BOUNDARY_PX[:, 0] = IMG_W - RIGHT_LARGE_FOOT_BOUNDARY_PX[:, 0]
RIGHT_SMALL_FOOT_BOUNDARY_PX = SMALL_FOOT_BOUNDARY_PX.copy()
RIGHT_SMALL_FOOT_BOUNDARY_PX[:, 0] = IMG_W - RIGHT_SMALL_FOOT_BOUNDARY_PX[:, 0]
FOREFOOT_VALID_THRESHOLD = 500.0


def has_valid_forefoot_pressure(
    values, front_channel_indices, threshold=FOREFOOT_VALID_THRESHOLD,
):
    values = np.asarray(values, dtype=float)
    if values.shape != (12,):
        raise ValueError('values는 반드시 C0~C11 순서의 12개 값이어야 합니다.')
    front_values = values[np.asarray(front_channel_indices, dtype=int)]
    return bool(np.any(front_values >= threshold))


def smooth_closed_boundary(points, n=900, s=5.0):
    closed = np.vstack([points, points[0]])
    (tck, _) = splprep([closed[:, 0], closed[:, 1]], s=s, per=True, k=3)
    u = np.linspace(0, 1, n, endpoint=False)
    (x, y) = splev(u, tck)
    return np.column_stack([x, y])


def normalize_in_mask(field, mask):
    field = field.copy()
    valid = field[mask]
    if valid.size == 0:
        return field
    fmin = np.min(valid)
    fmax = np.max(valid)
    if fmax > fmin:
        field = (field - fmin) / (fmax - fmin)
    else:
        field[:] = 0.0
    return np.clip(field, 0, 1)


def smoothstep(x, edge0, edge1):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_red_detail_cmap():
    colors = [
        '#2242b9', '#2e6fd6', '#38bedd', '#6bdd90', '#d9ea5f', '#f6cd55',
        '#f79a48', '#f06445', '#ea3131', '#c70000', '#920000', '#5f0000',
    ]
    return LinearSegmentedColormap.from_list('pressure_red_detail', colors, N=256)


def _make_foot_heatmap(
    values,
    sensor_centers_px,
    sensor_radii_px,
    foot_boundary_px,
    rbf_zero_nodes_px,
    save_path='result.png',
):
    values = np.asarray(values, dtype=float)
    if values.shape != (12,):
        raise ValueError('values는 반드시 C0~C11 순서의 12개 값이어야 합니다.')
    if np.any(values < 0):
        raise ValueError('압력값에는 음수가 들어갈 수 없습니다.')
    vmax = values.max()
    if vmax > 0:
        weights = values / vmax
    else:
        weights = np.zeros_like(values)
    (X, Y) = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    boundary = smooth_closed_boundary(foot_boundary_px, n=900, s=5.0)
    foot_path = MatplotlibPath(boundary)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = foot_path.contains_points(points).reshape(IMG_H, IMG_W)
    xs = sensor_centers_px[:, 0]
    ys = sensor_centers_px[:, 1]
    rbf_zero_nodes_px = np.asarray(rbf_zero_nodes_px, dtype=float)
    if rbf_zero_nodes_px.ndim != 2 or rbf_zero_nodes_px.shape[1] != 2:
        raise ValueError('rbf_zero_nodes_px는 (N, 2) 형태의 좌표여야 합니다.')
    rbf_xs = np.concatenate([xs, rbf_zero_nodes_px[:, 0]])
    rbf_ys = np.concatenate([ys, rbf_zero_nodes_px[:, 1]])
    rbf_weights = np.concatenate([
        weights, np.zeros(len(rbf_zero_nodes_px), dtype=float),
    ])
    # Virtual zero nodes affect only RBF, keeping real sensor Gaussian peaks.
    rbf = Rbf(
        rbf_xs, rbf_ys, rbf_weights,
        function='multiquadric', epsilon=120, smooth=0.16,
    )
    rbf_field = rbf(X, Y)
    rbf_field = np.clip(rbf_field, 0, None)
    local_field = np.zeros((IMG_H, IMG_W), dtype=float)
    support_field = np.zeros((IMG_H, IMG_W), dtype=float)
    for ((cx, cy), r, w) in zip(sensor_centers_px, sensor_radii_px, weights):
        local_sigma = max(r * 0.68, 10.0)
        support_sigma = max(r * 2.1, 26.0)
        local_kernel = np.exp(-0.5 * (
            ((X - cx) / local_sigma) ** 2 + ((Y - cy) / local_sigma) ** 2
        ))
        support_kernel = np.exp(-0.5 * (
            ((X - cx) / support_sigma) ** 2 + ((Y - cy) / support_sigma) ** 2
        ))
        local_field += w * local_kernel
        support_field += support_kernel
    rbf_field = normalize_in_mask(rbf_field, mask)
    local_field = normalize_in_mask(local_field, mask)
    support_field = normalize_in_mask(support_field, mask)
    field = 0.74 * rbf_field + 0.26 * local_field
    field *= 0.32 + 0.68 * support_field
    dist_inside = distance_transform_edt(mask)
    edge_fade = np.clip(dist_inside / 36.0, 0.0, 1.0)
    field *= 0.45 + 0.55 * edge_fade
    field = gaussian_filter(field, sigma=2.1)
    pressed_boost = smoothstep(field, 0.34, 0.68)
    field = field + 0.16 * pressed_boost * (1.0 - field)
    peak_boost = smoothstep(field, 0.72, 0.92)
    field = field + 0.1 * peak_boost * (1.0 - field)
    field = np.clip(field, 0, 1)
    field = np.power(field, 0.83)
    masked_field = np.ma.array(field, mask=~mask)
    cmap = make_red_detail_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    fig = Figure(figsize=(4.48, 11.06), dpi=180)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        masked_field, origin='upper', cmap=cmap,
        norm=PowerNorm(gamma=0.88, vmin=0.0, vmax=1.0),
        interpolation='bicubic',
    )
    ax.set_xlim(20, IMG_W - 20)
    ax.set_ylim(IMG_H - 5, 15)
    ax.axis('off')
    fig.savefig(
        save_path, dpi=180, facecolor='none', transparent=True,
        bbox_inches=None, pad_inches=0,
    )
    return (fig, ax)


def make_left_foot_heatmap(values, save_path='result.png'):
    foot_boundary_px = (
        LARGE_FOOT_BOUNDARY_PX
        if has_valid_forefoot_pressure(values, [0, 1])
        else SMALL_FOOT_BOUNDARY_PX
    )
    return _make_foot_heatmap(
        values,
        sensor_centers_px=SENSOR_CENTERS_PX,
        sensor_radii_px=SENSOR_RADII_PX,
        foot_boundary_px=foot_boundary_px,
        rbf_zero_nodes_px=LEFT_ARCH_ZERO_NODES_PX,
        save_path=save_path,
    )


def make_right_foot_heatmap(values, save_path='result_right.png'):
    foot_boundary_px = (
        RIGHT_LARGE_FOOT_BOUNDARY_PX
        if has_valid_forefoot_pressure(values, [11, 10])
        else RIGHT_SMALL_FOOT_BOUNDARY_PX
    )
    return _make_foot_heatmap(
        values,
        sensor_centers_px=RIGHT_SENSOR_CENTERS_PX,
        sensor_radii_px=RIGHT_SENSOR_RADII_PX,
        foot_boundary_px=foot_boundary_px,
        rbf_zero_nodes_px=RIGHT_ARCH_ZERO_NODES_PX,
        save_path=save_path,
    )


def combine_foot_heatmap_images(
    left_image_path, right_image_path, save_path='result_both.png', gap_px=32,
):
    left_image = imread(left_image_path)
    right_image = imread(right_image_path)
    if left_image.shape != right_image.shape:
        raise ValueError('왼발과 오른발 이미지 크기가 같아야 합니다.')
    if not isinstance(gap_px, int) or gap_px < 0:
        raise ValueError('gap_px는 0 이상의 정수여야 합니다.')
    image_height = left_image.shape[0]
    image_channels = left_image.shape[2]
    transparent_gap = np.zeros((image_height, gap_px, image_channels), dtype=left_image.dtype)
    combined_image = np.concatenate([left_image, transparent_gap, right_image], axis=1)
    imsave(save_path, combined_image)
    return combined_image


class PressureHeatmapError(RuntimeError):
    pass


@dataclass(frozen=True)
class SavedPressureHeatmaps:
    left_path: Path
    right_path: Path
    left_png: bytes
    right_png: bytes
    combined_path: Path | None = None


def _validate_values(values: Iterable[float], side: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PressureHeatmapError(
            f"{side} pressure values must be numeric C0-C11 values."
        ) from exc
    if array.shape != (SENSOR_COUNT,):
        raise PressureHeatmapError(
            f"{side} pressure values must contain exactly 12 values in C0-C11 order."
        )
    if not np.all(np.isfinite(array)):
        raise PressureHeatmapError(f"{side} pressure values must all be finite.")
    if np.any(array < 0.0):
        raise PressureHeatmapError(f"{side} pressure values cannot be negative.")
    return array


def create_pressure_heatmaps(
    left_values: Iterable[float],
    right_values: Iterable[float],
    output_dir: Path,
) -> SavedPressureHeatmaps:
    """Save independently normalized, transparent-background red-detail heatmaps.

    Left/right values are raw C0-C11 inputs. The right channel-to-pad mapping
    is applied by make_right_foot_heatmap; no additional reversal is performed.
    Each 180-DPI PNG is 806 by 1990 pixels. A local combined PNG uses a 32-pixel
    transparent gap; the backend receives the two RGBA PNGs without flattening.
    """
    left = _validate_values(left_values, "Left")
    right = _validate_values(right_values, "Right")
    directory = Path(output_dir)
    left_path = directory / "left_pressure_heatmap.png"
    right_path = directory / "right_pressure_heatmap.png"
    combined_path = directory / "both_pressure_heatmap.png"
    figures: list[Figure] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        left_figure, _ = make_left_foot_heatmap(left, save_path=left_path)
        figures.append(left_figure)
        right_figure, _ = make_right_foot_heatmap(right, save_path=right_path)
        figures.append(right_figure)
        combine_foot_heatmap_images(left_path, right_path, save_path=combined_path)
        left_png = left_path.read_bytes()
        right_png = right_path.read_bytes()
    except OSError as exc:
        raise PressureHeatmapError("Could not save the pressure heatmap PNG files.") from exc
    except Exception as exc:
        raise PressureHeatmapError("Could not interpolate and render the pressure values.") from exc
    finally:
        # Figures use private Agg canvases, never pyplot's global window registry.
        for figure in figures:
            figure.clear()

    return SavedPressureHeatmaps(
        left_path=left_path,
        right_path=right_path,
        left_png=left_png,
        right_png=right_png,
        combined_path=combined_path,
    )
