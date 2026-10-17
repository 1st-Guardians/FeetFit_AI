from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.services.two_marker_foot_geometry import (
    REQUIRED_MARKER_IDS,
    RigGeometry,
    TwoMarkerGeometryError,
    fit_two_marker_homography,
    marker_world_corners,
    measure_foot_mask,
)


class RigGeometryContractTests(unittest.TestCase):
    def test_edge_gap_derives_center_spacing_and_id_world_positions(self) -> None:
        geometry = RigGeometry()

        self.assertEqual(REQUIRED_MARKER_IDS, (1, 4))
        self.assertEqual(geometry.marker_size_mm, 20.0)
        self.assertEqual(geometry.marker_edge_gap_mm, 175.0)
        self.assertEqual(geometry.marker_center_distance_mm, 195.0)
        self.assertEqual(geometry.world_points_mm[1], (181.0, 25.0))
        self.assertEqual(geometry.world_points_mm[4], (181.0, 220.0))
        self.assertEqual(geometry.reference_edge_y_mm, 230.0)
        self.assertEqual(geometry.heel_offset_mm, 114.0)
        self.assertEqual(geometry.heel_line_y_mm, 344.0)
        self.assertEqual(geometry.board_height_mm, 354.0)

        payload = geometry.to_dict()
        self.assertEqual(payload["marker_ids"], [1, 4])
        self.assertEqual(payload["marker_gap_semantics"], "black_edge_to_edge")
        self.assertEqual(payload["marker_center_distance_mm"], 195.0)

    def test_175_mm_is_not_treated_as_center_spacing(self) -> None:
        geometry = RigGeometry(marker_edge_gap_mm=175.0)

        self.assertNotEqual(geometry.marker_center_distance_mm, 175.0)
        self.assertEqual(
            geometry.marker_center_distance_mm,
            geometry.marker_edge_gap_mm + geometry.marker_size_mm,
        )


class TwoMarkerHomographyContractTests(unittest.TestCase):
    def test_all_eight_marker_corners_recover_projective_world_coordinates(self) -> None:
        geometry = RigGeometry()
        world_to_image = np.asarray(
            [
                [3.7, 0.4, 80.0],
                [0.1, 2.8, 35.0],
                [0.00015, 0.00045, 1.0],
            ],
            dtype=np.float64,
        )
        corners_by_id = {
            marker_id: cv2.perspectiveTransform(
                marker_world_corners(center, geometry.marker_size_mm).reshape(
                    1, -1, 2
                ),
                world_to_image,
            ).reshape(4, 2)
            for marker_id, center in geometry.world_points_mm.items()
        }

        fit = fit_two_marker_homography(corners_by_id, geometry)
        query_world = np.asarray(
            [[90.0, 96.0], [275.0, 123.0]], dtype=np.float64
        )
        query_image = cv2.perspectiveTransform(
            query_world.reshape(1, -1, 2), world_to_image
        )
        recovered_world = cv2.perspectiveTransform(
            query_image, fit.image_to_world
        ).reshape(-1, 2)

        np.testing.assert_allclose(recovered_world, query_world, atol=1e-4)
        self.assertLess(fit.world_fit_rmse_mm, 1e-4)
        self.assertEqual(set(fit.per_marker_world_fit_rmse_mm), {1, 4})

    def test_missing_required_marker_cannot_fit_homography(self) -> None:
        geometry = RigGeometry()
        id1_only = {
            1: marker_world_corners(
                geometry.world_points_mm[1], geometry.marker_size_mm
            )
        }

        with self.assertRaisesRegex(TwoMarkerGeometryError, "missing: 4"):
            fit_two_marker_homography(id1_only, geometry)


class FootMaskMeasurementContractTests(unittest.TestCase):
    pixels_per_mm = 4.0

    def _empty_mask(self) -> np.ndarray:
        geometry = RigGeometry()
        return np.zeros(
            (
                int(np.ceil(geometry.board_height_mm * self.pixels_per_mm)),
                int(np.ceil(geometry.board_width_mm * self.pixels_per_mm)),
            ),
            dtype=np.uint8,
        )

    def _rectangular_foot_mask(
        self,
        total_length_mm: float,
        *,
        width_mm: float = 90.0,
    ) -> np.ndarray:
        geometry = RigGeometry()
        mask = self._empty_mask()
        toe_y_mm = geometry.heel_line_y_mm - total_length_mm
        left_x_mm = 40.0
        right_x_mm = left_x_mm + width_mm
        cv2.rectangle(
            mask,
            (
                int(round(left_x_mm * self.pixels_per_mm)),
                int(round(toe_y_mm * self.pixels_per_mm)),
            ),
            (
                int(round(right_x_mm * self.pixels_per_mm)),
                int(round(geometry.reference_edge_y_mm * self.pixels_per_mm)),
            ),
            255,
            thickness=-1,
        )
        return mask

    def test_length_boundaries_add_114_mm_exactly_once(self) -> None:
        geometry = RigGeometry()

        for expected_length_mm in (180.0, 320.0):
            with self.subTest(expected_length_mm=expected_length_mm):
                measurement = measure_foot_mask(
                    self._rectangular_foot_mask(expected_length_mm),
                    geometry,
                    side="image_left",
                    pixels_per_mm=self.pixels_per_mm,
                )

                self.assertAlmostEqual(
                    measurement.foot_length_mm, expected_length_mm, places=6
                )
                self.assertAlmostEqual(
                    measurement.visible_to_reference_mm,
                    expected_length_mm - 114.0,
                    places=6,
                )
                self.assertEqual(measurement.fixed_heel_offset_mm, 114.0)
                self.assertAlmostEqual(
                    measurement.visible_to_reference_mm
                    + measurement.fixed_heel_offset_mm,
                    expected_length_mm,
                    places=6,
                )
                self.assertEqual(
                    measurement.to_dict()["formula_check_mm"],
                    expected_length_mm,
                )

    def test_ball_width_uses_32_to_36_percent_horizontal_band(self) -> None:
        geometry = RigGeometry()
        measurement = measure_foot_mask(
            self._rectangular_foot_mask(240.0, width_mm=90.0),
            geometry,
            side="image_left",
            pixels_per_mm=self.pixels_per_mm,
        )

        self.assertEqual(measurement.ball_region_fraction_from_toe, (0.32, 0.36))
        first, second = measurement.ball_line_world_mm
        self.assertEqual(first[1], second[1])
        self.assertAlmostEqual(measurement.ball_width_mm, 90.0, places=6)
        distance_from_toe_mm = first[1] - measurement.toe_world_mm[1]
        self.assertGreaterEqual(distance_from_toe_mm, 0.32 * 240.0)
        self.assertLessEqual(distance_from_toe_mm, 0.36 * 240.0)
        self.assertEqual(
            measurement.to_dict()["ball_axis"],
            "perpendicular_to_id1_id4_axis",
        )

    def test_empty_small_and_non_reaching_masks_are_rejected(self) -> None:
        geometry = RigGeometry()

        with self.assertRaisesRegex(TwoMarkerGeometryError, "No foot mask"):
            measure_foot_mask(
                self._empty_mask(),
                geometry,
                side="image_left",
                pixels_per_mm=self.pixels_per_mm,
            )

        too_small = self._empty_mask()
        cv2.rectangle(
            too_small,
            (int(60 * self.pixels_per_mm), int(156 * self.pixels_per_mm)),
            (
                int(70 * self.pixels_per_mm),
                int(geometry.reference_edge_y_mm * self.pixels_per_mm),
            ),
            255,
            thickness=-1,
        )
        with self.assertRaisesRegex(TwoMarkerGeometryError, "too small"):
            measure_foot_mask(
                too_small,
                geometry,
                side="image_left",
                pixels_per_mm=self.pixels_per_mm,
            )

        non_reaching = self._empty_mask()
        cv2.rectangle(
            non_reaching,
            (int(20 * self.pixels_per_mm), int(20 * self.pixels_per_mm)),
            (int(150 * self.pixels_per_mm), int(150 * self.pixels_per_mm)),
            255,
            thickness=-1,
        )
        with self.assertRaisesRegex(
            TwoMarkerGeometryError, "does not reach the ID 4 reference edge"
        ):
            measure_foot_mask(
                non_reaching,
                geometry,
                side="image_left",
                pixels_per_mm=self.pixels_per_mm,
            )


if __name__ == "__main__":
    unittest.main()
